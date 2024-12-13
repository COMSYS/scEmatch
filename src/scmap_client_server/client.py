import json
import logging
from multiprocessing import Pool

import math
from pathlib import Path
import numpy as np
import scanpy as sc
from Pyfhel import Pyfhel
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.scmap_client_server.host import Host
from scipy.stats import rankdata
from scipy.sparse import csr_matrix
import time
from numpy.typing import NDArray

from src.scmap_client_server.constants import *
import warnings

# Benchmarks
CELL_DATA_BYTES_CSR_REF = 'cell-data-bytes-csr-ref'
CELL_DATA_BYTES_NUMPY64_REF = 'cell-data-bytes-numpy64-ref'
COMPLETE_DATASET_BYTES_REF = 'h5ad-dataset-bytes-ref'
CELL_DATA_BYTES_CSR_QUERY = 'cell-data-bytes-csr-query'
CELL_DATA_BYTES_NUMPY64_QUERY = 'cell-data-bytes-numpy64-query'
COMPLETE_DATASET_BYTES_QUERY = 'h5ad-dataset-bytes-query'

READ_REF = 'read-ref'
PREPROCESS_REF = 'preprocess-ref'
ENCRYPT_REF = 'encrypt-ref'
WRITE_REF = 'write-ref'
BYTES_REF = 'bytes-ref'

READ_QUERY = 'read-query'
PREPROCESS_QUERY = 'preprocess-query'
ENCRYPT_QUERY = 'encrypt-query'
WRITE_QUERY = 'write-query'
BYTES_QUERY = 'bytes-query'

READ_PUBLIC_KEY = 'read-public-key'

WRITE_META_AGGREGATION_SERVER = 'write-meta-aggregation-server'
BYTES_META_AGGREGATION_SERVER = 'bytes-meta-aggregation-server'

WRITE_META_SIMILARITY_SERVER = 'write-meta-similarity-server'
BYTES_META_SIMILARITY_SERVER = 'bytes-meta-similarity-server'

COMPUTE_LABELS = 'compute-labels'
WRITE_LABELS = 'write-labels'
BYTES_LABELS = 'bytes-labels'

READ_RESULTS_ONE_WAY = 'read-results-one-way'
READ_RESULTS_1TO1_ONE_WAY = 'read-results-1to1-one-way'
READ_RESULTS_TWO_WAY = 'read-results-two-way'
READ_RESULTS_1TO1_TWO_WAY = 'read-results-1to1-two-way'


class Client(Host):

    def __init__(
            self,
            client_name: str,
            write_directory: Path,
            aggregation_server_write_directory: Path,
            adata_reference_path: Path,
            adata_query_path: Path,
            cluster_type_key: str,
            benchmark_collector,
            num_cells: int = -1,
            seed: int = 0

    ):
        """
        Parameters
        :param client_name: Name of the client, e.g., 'aachen'.
        :param write_directory: The directory to write the prepared datasets, simulating to send data.
        :param adata_reference_path: A path to the file containing the reference dataset in .h5ad format.
        :param adata_query_path: A path to the file containing query dataset in .h5ad.
        :param cluster_type_key: To access the correct labels from the AnnData object. (adata.obs[cluster_type_key])
        """
        super().__init__(client_name, write_directory, benchmark_collector)
        self.aggregation_server_write_directory = aggregation_server_write_directory
        self.cluster_type_key = cluster_type_key

        # The paths from where the data is read
        self.adata_reference_path = adata_reference_path
        self.adata_query_path = adata_query_path

        # Data is read in the corresponding run method.
        self.adata_ref = None
        self.adata_query = None

        self.num_cells = num_cells
        self.seed = seed

        # Centroids are computed in the corresponding run method.
        self.centroids_ref = None

        # Depends on the ckks parameters in config file. Standard 4096
        self.ciphertext_slots = None

        # Matching results as a dictionary. Key is the type of the matching result, and value is the set of tuples
        # representing the matches. Obtain manually from top level, after aggregation server run, with
        # read_matching_results(self)
        self.matching_results = dict()

        # The mapper stores the encoded cell types, i.e., "cell_type_1":1, "cell_type_2":2 ...
        # The cluster types are equal between both datasets (ref. and aligned query dataset),
        # thus, the cluster types from the adata_ref is used.
        # Initialized in the run method
        # The class mapper is used to encode the query cell labels.
        self.cell_mapper = Client.ClassMapper()

        # Variable to access the ckks encryption capabilities.
        self.pyfhel_he_object = None

        # Creates the write directory to write its data in. Other hosts access files from this directory.
        self.create_directory()

        self.logger = logging.getLogger(client_name)

    # --- ENTRY POINT --- #
    def run(self, strategy: str):
        """
        Entry point. Executes the main logic.
        (1.) Read the datasets (aligned query and reference datasets) from disk.
        (2.) Perform initial processing on the data (transform to float64 array)
        (3.) Compute the centroids on the reference dataset ``self.adata_ref``.
        (4.) Compute and write the anonymized labels to disk, such that the aggregation servers can access it.
        :param strategy: {'simple', 'homomorphic'} (plaintext or homomorphic variant)
        """

        # (0.) Read public key
        if strategy == S_HOMOMORPHIC:
            with self.benchmark(READ_PUBLIC_KEY):
                self.read_public_key()

        # (1.) Read datasets
        self.logger.info("reading dataset")
        start = time.time()
        self.adata_ref = sc.read_h5ad(self.adata_reference_path)
        sampling_load_time_ref = time.time() - start  # exclude this from total runtime

        self.add_benchmark_result(READ_REF, sampling_load_time_ref)
        self.add_benchmark_result(COMPLETE_DATASET_BYTES_REF, Path(self.adata_reference_path).stat().st_size)

        start = time.time()
        self.adata_query = sc.read_h5ad(self.adata_query_path)
        sampling_load_time_query = time.time() - start  # exclude this from total runtime

        # (1.a) subsampling if desired
        if self.num_cells != -1:
            start = time.time()
            self.logger.info("subsampling dataset to %d cells", self.num_cells)

            subsampled_query = self.get_stratified_sample(self.num_cells)
            reduced_dataset_path = self.write_directory / "reduced_dataset.h5ad"
            subsampled_query.write(filename=reduced_dataset_path, compression="gzip")
            subsampling_time = time.time() - start

            with self.benchmark(READ_QUERY):
                self.adata_query = sc.read_h5ad(reduced_dataset_path)
            self.add_benchmark_result(COMPLETE_DATASET_BYTES_QUERY, Path(self.adata_query_path).stat().st_size)

            # subtract redundant load from total runtime
            self.add_benchmark_result(COMPLETE_RUNTIME, -sampling_load_time_query - subsampling_time)
        else:
            self.logger.info("using full dataset")
            self.add_benchmark_result(READ_QUERY, sampling_load_time_query)
            self.add_benchmark_result(COMPLETE_DATASET_BYTES_QUERY, Path(self.adata_query_path).stat().st_size)

        self.add_benchmark_result("cell-count", self.adata_query.X.shape[0])
        self.add_benchmark_result("feature-count", self.adata_query.X.shape[1])
        self.add_benchmark_result("cluster-count", len(set(self.adata_query.obs[self.cluster_type_key])))
        self.logger.info(
            f"cell-count:{self.adata_query.X.shape[0]} "
            f"cluster-count:{len(set(self.adata_query.obs[self.cluster_type_key]))} ({self.name})")

        # (2.) Transform to np.array as float64 datatype (if in sparse format)
        # (2.1) Convert reference data
        self.logger.info("converting reference data")
        with self.benchmark(PREPROCESS_REF):
            if isinstance(self.adata_ref.X, csr_matrix):
                self.add_benchmark_result(CELL_DATA_BYTES_CSR_REF, self.adata_ref.X.data.nbytes)
                self.adata_ref.X = self.adata_ref.X.toarray()
            self.adata_ref.X = self.adata_ref.X.astype('float64')
        self.add_benchmark_result(CELL_DATA_BYTES_NUMPY64_REF, self.adata_ref.X.nbytes)

        # (2.2) Convert query data
        self.logger.info("converting query data")
        with self.benchmark(PREPROCESS_QUERY):
            if isinstance(self.adata_query.X, csr_matrix):
                self.add_benchmark_result(CELL_DATA_BYTES_CSR_QUERY, self.adata_query.X.data.nbytes)
                self.adata_query.X = self.adata_query.X.toarray()
            self.adata_query.X = self.adata_query.X.astype('float64')
        self.add_benchmark_result(CELL_DATA_BYTES_NUMPY64_QUERY, self.adata_query.X.nbytes)

        # (3.) Compute centroids
        self.logger.info("computing centroids")
        with self.benchmark(PREPROCESS_REF):
            self.centroids_ref = self.compute_centroids()

        if len(self.cell_mapper.classes) < len(set(self.adata_query.obs[self.cluster_type_key])):
            self.logger.info("Filter out cells from query dataset for which the cluster centroids are zero.")
            with self.benchmark("Filter out cells from query dataset for which the cluster centroids are zero."):
                self.adata_query = self.adata_query[
                    self.adata_query.obs[self.cluster_type_key].isin(self.cell_mapper.classes)].copy()

        # (4.1.) Compute anonymized labels for aggregation server
        self.logger.info("computing anonymized labels for aggregation server")
        with self.benchmark(COMPUTE_LABELS):
            self.compute_anonymized_labels()

        # (4.2) Write anonymized labels
        self.logger.info("writing anonymized labels")
        with self.benchmark(WRITE_LABELS):
            self.write_anonymized_labels()
        self.add_benchmark_result(BYTES_LABELS, self.get_size_of_files(
            self.write_directory, ['anonymized_labels.by']
        ))

        # Select chosen strategy
        if strategy == S_SIMPLE:
            self.logger.info("running plaintext variant")
            self.run_simple()
        elif strategy == S_HOMOMORPHIC:
            self.logger.info("running homomorphic variant")
            self.run_homomorphic()
        else:
            raise ValueError("Strategy not supported.")

        self.add_benchmark_result('centroid-count', self.centroids_ref.shape[0])
        self.add_benchmark_result('centroid-features', self.centroids_ref.shape[1])
        self.add_benchmark_result('query-cell-count', self.adata_query.X.shape[0])
        self.add_benchmark_result('query-cell-features', self.adata_query.X.shape[1])

    def run_simple(self):
        """
        Plaintext variant. No use of CKKS scheme.
        :return:
        """
        # (1.) Write reference data
        self.logger.info("writing reference data")
        with self.benchmark(WRITE_REF):
            # write centroids (reference data)
            np.save(
                file=Path(self.write_directory, REFERENCE_DATA_SUFFIX),
                arr=self.centroids_ref)
        self.add_benchmark_result(BYTES_REF, self.get_size_of_files(
            self.write_directory, ['{}.npy'.format(REFERENCE_DATA_SUFFIX)]))

        # (2.) Write query data
        self.logger.info("writing query data")

        with self.benchmark(WRITE_QUERY):
            # write query data
            np.save(
                file=Path(self.write_directory, QUERY_DATA_SUFFIX),
                arr=self.adata_query.X)
        self.add_benchmark_result(BYTES_QUERY, self.get_size_of_files(
            self.write_directory, ['{}.npy'.format(QUERY_DATA_SUFFIX)]))

    def run_homomorphic(self):
        """
        Runs scmap-cluster homomorphically using the CKKS scheme.

            (1.) Write metadata to disk, such that the servers can access it.
            (2.) Scale the query cell matrix and the centroids per method {COSINE, PEARSONR, SPEARMANR}, obtaining three
                 scaled query and centroid matrices.
            (3.) Append zeros to each cell vector and centroid vector until the next power of two.
            (4.) Pack as many cells into a ciphertext as possible.
            (5.) Duplicate each centroid in one ciphertext until one ciphertext is fully packed with a centroid.
            (6.) Write the packed ciphertext to disk, for each cell/centroid-method combination, that is,
                 three query cell matrix files and three centroid matrix files.

        Filling query and centroids vectors is performed independently of each other, as they can have different
        dimensions.
        Example: cell vectors have length 100 -> fill up to 128
                 centroid vectors have length 300 -> fill up to 512
        """

        # (1.) Write metadata.
        self.logger.info("writing metadata")
        with self.benchmark(WRITE_META_SIMILARITY_SERVER):
            with open(self.write_directory / 'meta_similarity_server.json', 'w') as f:
                metadata = {
                    M_CENTROID_LENGTH: 2 ** math.ceil(math.log2(self.adata_ref.shape[1]))
                }
                json.dump(metadata, f)
        self.add_benchmark_result(BYTES_META_SIMILARITY_SERVER, self.get_size_of_files(
            self.write_directory, ['meta_similarity_server.json']
        ))

        with self.benchmark(WRITE_META_AGGREGATION_SERVER):
            with open(self.write_directory / 'meta_aggregation_server.json', 'w') as f:
                metadata = {
                    M_QUERY_CELL_COUNT: self.adata_query.shape[0],
                    M_CENTROID_LENGTH: 2 ** math.ceil(math.log2(self.adata_ref.shape[1]))
                }
                json.dump(metadata, f)
        self.add_benchmark_result(BYTES_META_AGGREGATION_SERVER, self.get_size_of_files(
            self.write_directory, ['meta_aggregation_server.json']
        ))

        # (2.) Scale query cell x gene and centroids, and obtain both as sparse matrices
        # Here, the matrices are duplicated to 3 instances each (query and centroids),
        # where each instance is scaled correspondingly
        # (3.) Add zeros to next power of 2 length
        # (2.1) Scale reference data
        self.logger.info("preprocessing data")
        with self.benchmark(PREPROCESS_REF):
            scaled_centroids_matrices = {
                method: self.scale(self.centroids_ref, method=method)
                for method in [COSINE, PEARSONR, SPEARMANR]}
            # (3.1) Add zeroes to scaled reference data
            self.add_zeros(scaled_centroids_matrices)

        # (2.2) Scale query data
        with self.benchmark(PREPROCESS_QUERY):
            scaled_query_matrices = {
                method: self.scale(self.adata_query.X, method=method)
                for method in [COSINE, PEARSONR, SPEARMANR]}
            # (3.2) Add zeroes to scaled query data
            self.add_zeros(scaled_query_matrices)

        # (4.) Reshape query data to fit perfectly with ciphertext slot number
        with self.benchmark(PREPROCESS_QUERY):
            self.reshape_query_data(scaled_query_matrices)

        # (5.) Duplicate centroids (reference data)
        # Each centroid is duplicated and packed into 1 array for easier computations on similarity server.
        # We have far fewer centroids, thus this is not a big overhead, but more simple to use.
        with self.benchmark(PREPROCESS_REF):
            self.duplicate_centroids(scaled_centroids_matrices)

        # (6.) Encrypt and write the ciphertexts to disk.

        # (6.1) For reference matrices (centroid matrices)
        self.logger.info("encrypting reference matrices")
        start_time = time.time()
        with Pool(3) as p:
            params = [(self.write_directory, self.aggregation_server_write_directory, method_name, matrix,
                       REFERENCE_DATA_SUFFIX, ENCRYPT_REF) for
                      method_name, matrix in scaled_centroids_matrices.items()]
            runtimes = p.map(write_encrypted_matrix, params)
            encrypt_runtime = max(runtimes)
            write_runtime = time.time() - start_time - encrypt_runtime

        self.add_benchmark_result(ENCRYPT_REF, encrypt_runtime)
        self.add_benchmark_result(WRITE_REF, write_runtime)

        # (6.2) For query matrices (gene x cell matrices)
        self.logger.info("encrypting query matrices")
        start_time = time.time()
        with Pool(3) as p:
            params = [(self.write_directory, self.aggregation_server_write_directory, method_name, matrix,
                       QUERY_DATA_SUFFIX, ENCRYPT_QUERY) for
                      method_name, matrix in scaled_query_matrices.items()]
            runtimes = p.map(write_encrypted_matrix, params)
            encrypt_runtime = max(runtimes)
            write_runtime = time.time() - start_time - encrypt_runtime

        self.add_benchmark_result(ENCRYPT_QUERY, encrypt_runtime)
        self.add_benchmark_result(WRITE_QUERY, write_runtime)

        # At last step, get the size of written files and set in the benchmark dictionary
        # (1.) Measure reference data bytes
        file_list_ref = [
            '{}.by'.format(f) for f
            in [COSINE_REFERENCE_DATA_NAME, SPEARMANR_REFERENCE_DATA_NAME, PEARSONR_REFERENCE_DATA_NAME]]
        sent_bytes_ref = Client.get_size_of_files(self.write_directory, file_list_ref)
        self.add_benchmark_result(BYTES_REF, sent_bytes_ref)

        # (2.) Measure query data bytes
        file_list_query = [
            '{}.by'.format(f) for f
            in [COSINE_QUERY_DATA_NAME, PEARSONR_QUERY_DATA_NAME, SPEARMANR_QUERY_DATA_NAME]]
        sent_bytes_query = Client.get_size_of_files(self.write_directory, file_list_query)
        self.add_benchmark_result(BYTES_QUERY, sent_bytes_query)

    def compute_centroids(self) -> NDArray[np.float64]:
        """
        Computes the centroids of the cell cluster in the reference dataset.
        :return:
        """
        centroids = list()
        centroid_classes = list()
        # Sorted just for more easy comparability of similarity scores across modes and runs
        # Any order can be used (also random permutation)
        for class_ in sorted(set(self.adata_ref.obs[self.cluster_type_key])):
            # Filter for indices of the current class/cluster
            indices = self.adata_ref.obs[self.cluster_type_key] == class_
            # Filter for all row vectors for that class
            cluster_cell_data = self.adata_ref.X[indices]
            # Compute centroid
            centroid = np.median(cluster_cell_data, axis=0)
            if np.sum(centroid) != 0:
                centroids.append(centroid)
                centroid_classes.append(class_)
            else:
                warnings.warn("Computed zero centroid")
        # Compute cell mapper
        self.cell_mapper.encode(centroid_classes)
        # transform list of arrays to one array
        return np.array(centroids, dtype=np.float64)

    @staticmethod
    def scale(matrix: NDArray[np.float64], method: str):
        """
        Helper function that is invoked by scale_data()
        Scales the sparse matrix by the specified method.
        :param matrix: cell x gene matrix
        :param method: {'cosine', 'pearsonr', 'spearmanr'}
        :return: scaled sparse_matrix
        """
        matrix_copy = matrix.copy()  # don't alter the original matrix
        if method == PEARSONR:
            means = matrix_copy.mean(axis=1)
            matrix_copy = matrix_copy - np.expand_dims(means, 1)
        elif method == SPEARMANR:
            matrix_copy = rankdata(matrix_copy, axis=1, method='average')
            means = matrix_copy.mean(axis=1)
            matrix_copy = matrix_copy - np.expand_dims(means, 1)
        elif method == COSINE:
            # No precomputations needed for cosine
            pass
        # Compute a scaling factor per row (cell). scaling_factors is a column vector.
        # Inverse Euclidean norms
        scaling_factors_ = (1 / np.sqrt(np.sum(matrix_copy ** 2, axis=1))).reshape(-1, 1)
        # Scale matrix by scaling factors.
        return matrix_copy * scaling_factors_

    def compute_anonymized_labels(self):
        # We add a series to each adata object that stores the encoded labels, i.e., integers instead of cluster names.
        self.adata_query.obs['encoded_label'] = [
            self.cell_mapper.get_index_by_class(label) for label in self.adata_query.obs[self.cluster_type_key]]

    @staticmethod
    def add_zeros(matrices: dict[str, NDArray[np.float64]]):
        """
        Adds zeros to the matrices in place to the next power of two for simplified homomorphic operations when packing.
        Example [1,2,3,4,5] -> [1,2,3,4,5,0,0,0]
        :param matrices: dict of {method_identifier: NDArray[np.float64]}.
        :return:
        """
        for method_name, matrix in matrices.items():
            next_power = 2 ** math.ceil(math.log2(matrix.shape[1]))
            zeros_to_fill = next_power - matrix.shape[1]
            matrices[method_name] = np.hstack([
                matrix,
                np.zeros(shape=(matrix.shape[0], zeros_to_fill))])

    def reshape_query_data(self, scaled_query_matrices):
        """
        In place computations in the first argument ``scaled_query_matrices``.
        Reshapes query data to dimensions of ciphertexts. Assumes the query data is already scaled and zero filled, i.e,
        cell vectors are extended with zeros such that they have length of a power of 2.
        :param scaled_query_matrices: dict of {method_identifier: NDArray[np.float64]}. Reshaped in place.
        """

        for method_name, matrix in scaled_query_matrices.items():
            vectors_per_ciphertext = self.ciphertext_slots // matrix.shape[1]
            overflow_vector_count = (matrix.size % self.ciphertext_slots) // matrix.shape[1]
            # Missing rows, such that reshape works (Zero cell vectors are appended to the bottem)
            missing_rows = (vectors_per_ciphertext - overflow_vector_count) % vectors_per_ciphertext
            matrix = np.vstack([matrix, np.zeros((missing_rows, matrix.shape[1]))])
            # Reshape in place
            scaled_query_matrices[method_name] = matrix.reshape((-1, self.ciphertext_slots))

    def duplicate_centroids(self, scaled_centroids_matrices):
        """In place computations in the first argument ``scaled_centroid_matrices``."""
        duplications = self.ciphertext_slots // 2 ** math.ceil(math.log2(self.adata_ref.shape[1]))
        for method_name, matrix in scaled_centroids_matrices.items():
            scaled_centroids_matrices[method_name] = np.hstack(
                [scaled_centroids_matrices[method_name] for _ in range(duplications)])

    def write_anonymized_labels(self):
        with open(self.write_directory / 'anonymized_labels.by', 'wb') as f:
            f.write(np.array(self.adata_query.obs['encoded_label'], dtype=np.int64).tobytes())

    def read_public_key(self):
        with self.benchmark(READ_PUBLIC_KEY):
            self.pyfhel_he_object = Pyfhel()
            with open(self.aggregation_server_write_directory / 'public_key.by', 'rb') as f:
                for method in [
                    self.pyfhel_he_object.from_bytes_context,
                    self.pyfhel_he_object.from_bytes_public_key,
                ]:
                    length_byte_string = f.read(4)
                    length = int.from_bytes(length_byte_string, byteorder='big')
                    method(f.read(length))
            self.ciphertext_slots = self.pyfhel_he_object.get_nSlots()

    def read_matching_results(self, result_type):
        """
        After aggregation server has written the matching results, read the result.
        :return:
        """
        key = {
            R_ONE_WAY_MATCH: READ_RESULTS_ONE_WAY,
            R_1_TO_1_ONE_WAY_MATCH: READ_RESULTS_1TO1_ONE_WAY,
            R_TWO_WAY_MATCH: READ_RESULTS_TWO_WAY,
            R_1_TO_1_TOW_WAY_MATCH: READ_RESULTS_1TO1_TWO_WAY
        }.get(result_type)
        with self.benchmark(key):
            with open(self.aggregation_server_write_directory / f'{self.name}_{result_type}.by', 'rb') as f:
                result = np.frombuffer(f.read(), dtype=np.int64)
                # Reshape to array containing two columns, where each row represents the match between the two datasets.
                # If only one element is contained in the result, it is '-1' indicating no match, and thus no reshape.
                if result.size > 1:
                    result = result.reshape(-1, 2)
                self.matching_results[result_type] = result

    def get_stratified_sample(self, n):
        """If a subsample of the query dataset should be used for matching."""
        index = np.arange(0, self.adata_query.shape[0])
        _, sample_index, _, _ = train_test_split(
            index,
            self.adata_query.obs[self.cluster_type_key],
            test_size=n,
            shuffle=True,
            stratify=self.adata_query.obs[self.cluster_type_key],
            random_state=self.seed
        )
        adata = self.adata_query[sample_index, :].copy()
        return adata

    class ClassMapper:
        """
        Encodes classes (cluster labels) to integers and stores mapping between encoded classes
        (integers) and the actual class names.
        Stores the mapping by class name.
        """

        def __init__(self):
            self.class_to_index = None
            self.index_to_class = None
            self.classes = None
            self.encoded_classes = None

        def encode(self, classes):
            """
            Encodes classes to positive integers and stores mapping between encoded classes
            :param classes:
            """
            self.class_to_index = {
                class_: index for index, class_ in enumerate(classes)
            }
            self.index_to_class = {
                index: class_ for index, class_ in enumerate(classes)
            }
            self.classes = classes
            self.encoded_classes = [i for i, _ in enumerate(self.classes)]

        def get_class_by_index(self, i: int) -> str:
            """
            Returns the class name of a given encoded class.
            :param i: The encoded class index.
            :return: The name of the class.
            """
            return 'unassigned' if i == -1 else self.index_to_class[i]

        def get_index_by_class(self, class_: str) -> int:
            """
            Returns the index of a given class/cluster name.
            :param class_:
            :return: index of the class/cluster.
            """
            return -1 if class_ == 'unassigned' else self.class_to_index[class_]


def prepend_length(byte_string):
    length = len(byte_string).to_bytes(4, byteorder='big')
    return length + byte_string


def write_encrypted_matrix(params):
    write_directory, agg_dir, method_name, matrix, suffix, benchmark_key = params
    pyfhel_he_object = Pyfhel()
    with open(f'{agg_dir}/public_key.by', 'rb') as f:
        for method in [
            pyfhel_he_object.from_bytes_context,
            pyfhel_he_object.from_bytes_public_key,
        ]:
            length_byte_string = f.read(4)
            length = int.from_bytes(length_byte_string, byteorder='big')
            method(f.read(length))
    runtime = 0

    with open(Path(write_directory, "{}_{}.by".format(method_name, suffix)), 'wb') as f:
        for packed_vector in tqdm(matrix):
            start_time = time.time()
            ctxt = pyfhel_he_object.encryptFrac(packed_vector)
            runtime += time.time() - start_time

            serialized_vector = ctxt.to_bytes()
            f.write(prepend_length(serialized_vector))
    return runtime
