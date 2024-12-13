import json
import logging
import time
from pathlib import Path
from src.scmap_client_server import server
import numpy as np
from Pyfhel import Pyfhel, PyCtxt
from numpy.typing import NDArray
from collections import Counter
from src.scmap_client_server.constants import *

# Benchmark keys
READ_SIMILARITIES_C1_C2 = 'read-similarities-c1-c2'
DECRYPT_SIMILARITIES_C1_C2 = 'decrypt-similarities-c1-c2'
COMPUTE_CELL_LEVEL_MATCHES_C1_C2 = 'compute-cell-level-matches-c1-c2'
COMPUTE_ONE_WAY_MATCHES_C1_C2 = 'compute-one-way-matches-c1-c2'
COMPUTE_1_TO_1_ONE_WAY_MATCHES_C1_C2 = 'compute-1-to-1-one-way-matches-c1-c2'

READ_SIMILARITIES_C2_C1 = 'read-similarities-c2-c1'
DECRYPT_SIMILARITIES_C2_C1 = 'decrypt-similarities-c2-c1'
COMPUTE_CELL_LEVEL_MATCHES_C2_C1 = 'compute-cell-level-matches-c2-c1'
COMPUTE_ONE_WAY_MATCHES_C2_C1 = 'compute-one-way-matches-c2-c1'
COMPUTE_1_TO_1_ONE_WAY_MATCHES_C2_C1 = 'compute-1-to-1-one-way-matches-c2-c1'

COMPUTE_TWO_WAY_MATCHES = 'compute-two-way-matches'
COMPUTE_1_TO_1_TWO_WAY_MATCHES = 'compute-1-to-1-two-way-matches'

READ_META_C1 = 'read-meta-c1'
READ_META_C2 = 'read-meta-c2'

READ_LABELS_C1 = 'read-labels-c1'
READ_LABELS_C2 = 'read-labels-c2'

GENERATE_CKKS_KEY = 'generate-ckks-key'
# For clients
WRITE_PUBLIC_KEY = 'write-public-key'
BYTES_PUBLIC_KEY = 'bytes-public-key'
# For similarity server
WRITE_PUBLIC_AND_EVAL_KEYS = 'write-public-and-eval-keys'
BYTES_PUBLIC_AND_EVAL_KEYS = 'bytes-public-and-eval-keys'

WRITE_ONE_WAY_C1 = 'write-one-way-c1'
BYTES_ONE_WAY_C1 = 'bytes-one-way-c1'
WRITE_1_TO_1_ONE_WAY_C1 = 'write-1-to-1-one-way-c1'
BYTES_1_TO_1_ONE_WAY_C1 = 'bytes-1-to-1-one-way-c1'
WRITE_TWO_WAY_C1 = 'write-two-way-c1'
BYTES_TWO_WAY_C1 = 'bytes-two-way-c1'
WRITE_1_TO_1_TWO_WAY_C1 = 'write-1-to-1-two-way-c1'
BYTES_1_TO_1_TWO_WAY_C1 = 'bytes-1-to-1-two-way-c1'

WRITE_ONE_WAY_C2 = 'write-one-way-c2'
BYTES_ONE_WAY_C2 = 'bytes-one-way-c2'
WRITE_1_TO_1_ONE_WAY_C2 = 'write-1-to-1-one-way-c2'
BYTES_1_TO_1_ONE_WAY_C2 = 'bytes-1-to-1-one-way-c2'
WRITE_TWO_WAY_C2 = 'write-two-way-c2'
BYTES_TWO_WAY_C2 = 'bytes-two-way-c2'
WRITE_1_TO_1_TWO_WAY_C2 = 'write-1-to-1-two-way-c2'
BYTES_1_TO_1_TWO_WAY_C2 = 'bytes-1-to-1-two-way-c2'


class AggregationServer(server.Server):

    def __init__(
            self,
            server_name: str,
            write_directory: Path,
            similarity_server_write_directory: Path,
            client1_name: str,
            client1_write_directory: Path,
            client2_name: str,
            client2_write_directory: Path,
            benchmark_collector,
            ciphertext_slots: int = None,
            scmap_threshold: float = 0.7,
            cluster_level_threshold: float = 0.3,
    ):
        super().__init__(
            server_name=server_name,
            write_directory=write_directory,
            client1_name=client1_name,
            client1_write_directory=client1_write_directory,
            client2_name=client2_name,
            client2_write_directory=client2_write_directory,
            benchmark_collector=benchmark_collector
        )
        self.similarity_server_write_directory = similarity_server_write_directory

        # Meta information of both clients
        self.client1_centroid_length = None
        self.client1_query_cell_count = None
        self.client2_centroid_length = None
        self.client2_query_cell_count = None

        # Anonymous labels
        self.labels_client1 = None
        self.labels_client2 = None

        # CKKS information
        self.pyfhel_he_object = None
        self.ciphertext_slots = None

        # scE(match) configuration
        self.scmap_threshold: float = scmap_threshold
        self.cluster_level_threshold: float = cluster_level_threshold

        # Creates the write directory to write its data in. Other hosts access files from this directory.
        self.create_directory()

        self.logger = logging.getLogger(server_name)

        # similarity_values: key1 is matching_direction, key2 is the method {COSINE, PEARSONR, SPEARMANR}
        self.similarity_values: {str: {str: NDArray[np.float64]}} = dict()
        # cell_level_matches: key is the matching_direction
        self.cell_level_matches: {str: NDArray[np.int64]} = dict()

    # --- ENTRY POINT --- #
    def run(self, strategy: str):
        """
        Entry point. Executes the main logic.
        (1.) Read the anonymized query cell labels from each client.
        :param strategy: {'simple', 'homomorphic'} (plaintext or homomorphic variant)
        :return:
        """

        # (1.1) Read labels from client 1
        self.logger.info("read labels c1")
        with self.benchmark(READ_LABELS_C1):
            self.read_labels_client1()

        # (1.2) Read labels from client 2
        self.logger.info("read labels c2")
        with self.benchmark(READ_LABELS_C2):
            self.read_labels_client2()

        if strategy == S_SIMPLE:
            self.logger.info("running plaintext variant")
            self.run_simple()
        elif strategy == S_HOMOMORPHIC:
            self.logger.info("running homomorphic variant")
            self.run_homomorphic()
        else:
            raise ValueError("Strategy not supported.")

    def run_simple(self):
        """
        Plaintext variant.
        Method to perform both matching directions, which instantiates the instance variables ``self.similarity_values``
        and ``self.cell_level_matches`` by calling ``self.process_one_direction(matching_direction)``, for both matching
        directions. The string ``matching_direction`` serves as the file_name identifier for the files written by the
        similarity server, as well as the dictionary keys for both instance variables ``self.similarity_values``
        and ``self.cell_level_matches``.
        :return:
        """
        # cell_level_matches of matching direction client1 -> client2
        self.logger.info("aggregating c1 -> c2 direction")
        matching_direction_c1_c2 = '{}_{}'.format(self.client1_name, self.client2_name)
        self.process_one_direction(matching_direction_c1_c2)

        # cell_level_matches of matching direction client2 -> client1
        self.logger.info("aggregating c2 -> c1 direction")
        matching_direction_c2_c1 = '{}_{}'.format(self.client2_name, self.client1_name)
        self.process_one_direction(matching_direction_c2_c1)

    def process_one_direction(self, matching_direction: str):
        """
        Plaintext variant.
        Performs cell level computations for one direction, i.e., matches each cell to a cluster. It instantiates the
        instance variables ``self.similarity_values`` and ``self.cell_level_matches`` for the passed ``matching_direction``.
            (1.) Read the similarity values with ``matching_direction`` as the file_name, e.g.,
             matching_direction=aachen_dresden -> read file 'aachen_dresden.npz' containing similarity values for the
             three methods cosine, pearsonr and spearmanr. The .npz format allows for keyed access of contained
             numpy arrays. The similarity arrays of one method can be accessed by COSINE, PEARSONR and SPEARMANR.
            (2.) Compute the matched cluster for each cell.
            (3.) Instantiate the instance variables ``self.similarity_values`` and ``self.cell_level_matches``.
        :param matching_direction: String identifier for the matching direction, e.g., 'aachen_dresden'.
        :return:
        """
        is_c1_c2 = matching_direction == '{}_{}'.format(self.client1_name, self.client2_name)

        self.logger.info("loading similarity matrices")
        with self.benchmark(READ_SIMILARITIES_C1_C2 if is_c1_c2 else READ_SIMILARITIES_C2_C1):
            similarity_matrices = np.load(
                Path(self.similarity_server_write_directory, '{}{}'.format(matching_direction, '.npz')))

        self.logger.info("matching cells")
        start_time = time.time()
        with self.benchmark(COMPUTE_CELL_LEVEL_MATCHES_C1_C2 if is_c1_c2 else COMPUTE_CELL_LEVEL_MATCHES_C2_C1):
            pred = self.match_cells(
                similarity_matrices[COSINE],
                similarity_matrices[PEARSONR],
                similarity_matrices[SPEARMANR]
            )

            # Instantiate similarity and cell_level_matches instance variables. Such that we can access them from top level.
            self.similarity_values[matching_direction] = {
                COSINE: similarity_matrices[COSINE],
                PEARSONR: similarity_matrices[PEARSONR],
                SPEARMANR: similarity_matrices[SPEARMANR],
            }

            self.cell_level_matches[matching_direction] = pred

    def run_homomorphic(self):
        """
        Runs the aggregation server in homomorphic version.
            (1.) Server reads meta_data that specifies for one client:
                - Number of cells in query data
                - Feature vector length of reference data (Centroid length)
            (2.) Perform cell level matching for both directions
        :return:
        """

        # (1.) Read metadata of both clients
        self.logger.info("reading metadata")
        with self.benchmark(READ_META_C1):
            with open(self.client1_write_directory / 'meta_aggregation_server.json', 'r') as f:
                meta_information = json.load(f)
                self.client1_query_cell_count = meta_information[M_QUERY_CELL_COUNT]
                self.client1_centroid_length = meta_information[M_CENTROID_LENGTH]

        with self.benchmark(READ_META_C2):
            with open(self.client2_write_directory / 'meta_aggregation_server.json', 'r') as f:
                meta_information = json.load(f)
                self.client2_query_cell_count = meta_information[M_QUERY_CELL_COUNT]
                self.client2_centroid_length = meta_information[M_CENTROID_LENGTH]

        # Matching direction specification client1 -> client2
        specification_c1_c2 = {
            # access correct file and name the output file
            'matching_direction': '{}_{}'.format(self.client1_name, self.client2_name),
            # Centroid length of the reference client
            'centroid_length': self.client2_centroid_length,
            # Total query cells of query client (To cut off zero values due to packing)
            'query_cell_count': self.client1_query_cell_count
        }

        # Matching direction specification client2 -> client1
        specification_c2_c1 = {
            'matching_direction': '{}_{}'.format(self.client2_name, self.client1_name),
            'centroid_length': self.client1_centroid_length,
            'query_cell_count': self.client2_query_cell_count
        }

        # (2) Process both matching rounds
        # (2.1) Perform cell level matching first matching direction.
        self.logger.info("aggregating c1 -> c2 direction")
        self.process_one_direction_homomorphic(**specification_c1_c2)
        # (2.2) Perform cell level matching second matching direction.
        self.logger.info("aggregating c2 -> c1 direction")
        self.process_one_direction_homomorphic(**specification_c2_c1)

    def process_one_direction_homomorphic(
            self,
            matching_direction: str,
            centroid_length: int,
            query_cell_count: int):
        """
        Instantiates the instance variables ``self.similarity_values`` and ``self.cell_level_matches`` for given
        ``matching_direction``.
        :param matching_direction:
        :param centroid_length:
        :param query_cell_count:
        :return:
        """
        is_c1_c2 = matching_direction == f'{self.client1_name}_{self.client2_name}'
        similarity_values = dict()

        # (1.) Read similarity values.
        # read_similarities_homomorphic(...) reads (and decrypts) and reorders the values to correct order.
        self.logger.info("reading similarity values")
        for method in [COSINE, PEARSONR, SPEARMANR]:
            similarity_values[method] = self.read_similarities_homomorphic(
                matching_direction=matching_direction,
                method=method,
                cell_count=query_cell_count,
                centroid_length=centroid_length)

        # (2.) Compute the cell-level matches based on the similarity values and threshold.
        self.logger.info("matching cells")
        with self.benchmark(COMPUTE_CELL_LEVEL_MATCHES_C1_C2 if is_c1_c2 else COMPUTE_CELL_LEVEL_MATCHES_C2_C1):
            cell_level_matches = self.match_cells(
                similarity_values[COSINE], similarity_values[PEARSONR], similarity_values[SPEARMANR]
            )
            # (3.) Instantiate instance variable with similarity values
            self.similarity_values[matching_direction] = similarity_values
            # (4.) Instantiate the cell_level matches instance variable.
            self.cell_level_matches[matching_direction] = cell_level_matches

    def read_similarities_homomorphic(self, matching_direction, method, cell_count, centroid_length) -> NDArray[np.float64]:
        """
        Read similarities of one file, e.g, cosine similarity values. Reorders the similarity values, accordingly.
        The filename is constructed with matching_direction and method, e.g.,
        matching_direction=aachen_dresden AND method=cosine -> read file 'aachen_dresden_cosine.by'.
        Furthermore, matching direction servers as a benchmark key.

        :param matching_direction: String identifier.
        :param method: Identifier for the file to read. One of COSINE, PEARSONR or SPEARMANR.
        :param cell_count: Number of cells in the query dataset of matching_direction (query-client_reference-client).
        :param centroid_length: Length of centroid.
        :return: Similarity values unencrypted.
        """
        is_c1_c2 = matching_direction == '{}_{}'.format(self.client1_name, self.client2_name)

        # (1.) read all similarity values of 1 file as bytes and deserialize to pyfhel PyCtxt.
        with self.benchmark(READ_SIMILARITIES_C1_C2 if is_c1_c2 else READ_SIMILARITIES_C2_C1):
            def deserializer(x): return PyCtxt(pyfhel=self.pyfhel_he_object, bytestring=x)

            # Example file: sent_data/similarity_server/aachen_dresden_cosine.by
            #               similarity_server_write_dir      name of the file
            file_path = Path(
                self.similarity_server_write_directory,
                '{}_{}{}'.format(matching_direction, method, '.by')
            )

            similarity_storage = self.read_bytes(file_path, deserializer)

        # (2.) Decrypt.
        with self.benchmark(DECRYPT_SIMILARITIES_C1_C2 if is_c1_c2 else DECRYPT_SIMILARITIES_C2_C1):
            decrypted_similarity_storage = list()
            for ctxt in similarity_storage:
                decrypted_similarity_storage.append(self.pyfhel_he_object.decrypt(ctxt))
            similarity_storage = decrypted_similarity_storage

        # (3.) Reorder and cut off zeros
        with self.benchmark(COMPUTE_CELL_LEVEL_MATCHES_C1_C2 if is_c1_c2 else COMPUTE_CELL_LEVEL_MATCHES_C2_C1):
            # The overall similarity matrix storing per column the similarity values for one centroid
            similarity_matrix = list()
            # An auxiliary list, storing the similarity values for the current centroid.
            similarity_values_centroid = list()
            # Recalculate the correct index order, as in the homomorphic version, the indices follow a specific
            # pattern due to the packing strategy
            correct_order = [
                i + centroid_length * j for i in range(centroid_length) for j in
                range(self.ciphertext_slots // centroid_length)]

            # Keep track of processed ciphertexts. This counter is used to associate the similarity scores to the
            # correct centroid.
            counter = 0
            for similarity_array in similarity_storage:
                # Order the array
                unordered_similarity_values = similarity_array
                reordered_similarity_values = np.ndarray(shape=self.ciphertext_slots, dtype=np.float64, )
                for i, correct_i in enumerate(correct_order):
                    reordered_similarity_values[i] = unordered_similarity_values[correct_i]
                counter += self.ciphertext_slots
                similarity_values_centroid.append(reordered_similarity_values)
                # If processed cell count (counter) is greater or equal to actual cell count, append the similarity
                # values
                if counter >= cell_count:
                    similarity_values_centroid = np.hstack(similarity_values_centroid)  # Transform to one array
                    reshaped_sims = np.array(similarity_values_centroid).reshape(-1, 1)  # Transform to column
                    similarity_matrix.append(reshaped_sims)
                    similarity_values_centroid = list()
                    counter = 0
            # Finally stack columns to a np.array and cut of zeros due to ciphertext packing.
            cut_matrix = np.hstack(similarity_matrix)[:cell_count]
        return cut_matrix

    def match_cells(
            self,
            cosine: NDArray[np.float64],
            pearsonr: NDArray[np.float64],
            spearmanr: NDArray[np.float64]) -> NDArray[np.int64]:
        """
        Compute clusters labels with the given similarity values ``cosine``, ``pearsonr`` and
        ``spearmanr`` for each cell.
        :param cosine: Array of similarity values of form `cell` x `similarity_values`, where each column corresponds to
                         one centroid containing cosine similarity scores.
        :param pearsonr: Pearsonr similarity array of form ``a_cosine``.
        :param spearmanr: Spearmanr similarity array of form ``a_cosine``.
        :return: Matched cluster labels.
        """
        # Calculate matched clusters with regard to each similarity value combination.
        # cp: cosine_similarity and pearsonr
        # cs: cosine_similarity and spearmanr
        # ps: pearsonr and spearmanr
        cp = self.match_cells_(cosine, pearsonr)
        cs = self.match_cells_(cosine, spearmanr)
        ps = self.match_cells_(pearsonr, spearmanr)

        # Collapse all three row vectors containing the assigned clusters to one row vector with the final
        # assignment. If at least 1 value (the assigned class/cluster) is above -1 at cell i in the arrays cp, cs and
        # ps chose this index as the assigned class/cluster for cell i. That is, e.g., for cp, cosine and pearsonr agree
        # on the maximum value and at least one of these maximum values is above 0.7.
        #
        # If two values in the partial result arrays cp, cs and ps are above -1 for a cell i, then it is guaranteed that
        # they are the same, because at least two similarity values have to agree on the same cluster.
        #
        #                           index:      0  1  2  3  4
        # Example assignments:          cp =   [2, 3,-1,-1, 5]
        #                               cs =   [2,-1, 2,-1, 5]
        #                               ps =   [-1,3, 2, 3, 5]
        #                               --->   [2, 3, 2, 3, 5]
        #
        return np.maximum.reduce([cp, cs, ps])

    def match_cells_(self, a1: NDArray[np.float64], a2: NDArray[np.float64]) -> NDArray[np.int64]:
        """
        Expects two similarity arrays of form

            cell x similarity_values

        where each column corresponds to one centroid. Caller has to ensure that a1 and a2 are similarity values of
        different similarity measures, e.g., a1=cosine_similarity_array and a2=pearsonr_similarity_array.

        Columns (centroids) are ordered as they were sent by the clients.

        Generates a row vector with the assigned cluster per cell on basis of the similarity values across all centroids.
        That is,

             0   1   2   3   4   5   6   7   8  9     # Implicit cell id.
            [2,  3, -1,  5, -1,  5,  6, -1, 10, 8]    # Assigned cluster per cell. Corresponds to the colum index of the
                                                        centroid.

        where -1 indicates unassigned. Computes cluster based on 2 similarity values matrices.

        :param a1: similarity matrix of shape (n_cells, n_similarities), where n_similarity = number of centroids
        :param a2: similarity matrix of shape (n_cells, n_similarities), where n_similarity = number of centroids
        """
        # Per cell get column index with maximum value, i.e., the cluster with greatest similarity score, and store in a
        # 1D row array.
        argmax_a1 = np.argmax(a1, axis=1)
        argmax_a2 = np.argmax(a2, axis=1)

        # Per cell get the maximum similarity score and store in a 1D row array.
        # Used to ensure, that threshold requirement is met.
        max_values_a1 = np.amax(a1, axis=1)
        max_values_a2 = np.amax(a2, axis=1)

        # Boolean mask: Per cell, the assigned clusters, i.e., the column indices, have to agree.
        agree_on_max_value = argmax_a1 == argmax_a2

        # 2 Boolean masks: Indicating, whether the maximum similarity value of a cell is greater than the threshold.
        a1_threshold_mask = max_values_a1 > self.scmap_threshold
        a2_threshold_mask = max_values_a2 > self.scmap_threshold

        # At least one of the maximum similarity values has to be greater than the threshold.
        at_least_one_greater_threshold = np.any(np.vstack((a1_threshold_mask, a2_threshold_mask)), axis=0)

        # Combine the agreement on most similar cluster and the threshold requirement.
        # Boolean mask: Per cell the clusters have to agree AND at least one of the two similarity is greater than threshold.
        bool_mask_clusters = np.all(np.vstack((agree_on_max_value, at_least_one_greater_threshold)), axis=0)

        # With help of boolean_mask_clusters filter the matched clusters.
        return np.where(bool_mask_clusters, argmax_a1, -1)

    def read_labels_client1(self):
        """
        Reads anonymized labels from client1.
        :return:
        """
        with open(self.client1_write_directory / 'anonymized_labels.by', 'br') as f:
            self.labels_client1 = np.frombuffer(f.read(), dtype=int)

    def read_labels_client2(self):
        """
        Reads anonymized labels from client2.
        :return:
        """
        with open(self.client2_write_directory / 'anonymized_labels.by', 'br') as f:
            self.labels_client2 = np.frombuffer(f.read(), dtype=int)

    def generate_he_keys_and_context(self, he_parameters: dict):
        """
        :param he_parameters: context_params for ckks scheme
        :return:
        """
        with self.benchmark(GENERATE_CKKS_KEY):
            self.pyfhel_he_object = Pyfhel(
                context_params={
                    'scheme': 'ckks',
                    'n': he_parameters['n'],
                    'scale': he_parameters['scale'],
                    'qi_sizes': he_parameters['qi_sizes']})
            self.pyfhel_he_object.keyGen()
            self.pyfhel_he_object.relinKeyGen()
            self.pyfhel_he_object.rotateKeyGen()
            self.ciphertext_slots = self.pyfhel_he_object.get_nSlots()

    def write_public_key(self):
        """
        Write client key as bytes. Only context and public key necessary. For clients.
        :return:
        """
        with self.benchmark(WRITE_PUBLIC_KEY):
            self.write_bytes(
                file=self.write_directory / 'public_key.by',
                byte_strings=[self.pyfhel_he_object.to_bytes_context(),
                              self.pyfhel_he_object.to_bytes_public_key()])
        self.add_benchmark_result(BYTES_PUBLIC_KEY, self.get_size_of_files(
            self.write_directory, ['public_key.by']
        ))

    def write_public_and_eval_keys(self):
        """
        Write public and evaluation keys in a file. For similarity server.
        :return:
        """
        with self.benchmark(WRITE_PUBLIC_AND_EVAL_KEYS):
            self.write_bytes(
                file=self.write_directory / 'public_and_eval_keys.by',
                byte_strings=[self.pyfhel_he_object.to_bytes_context(),
                              self.pyfhel_he_object.to_bytes_public_key(),
                              self.pyfhel_he_object.to_bytes_relin_key(),
                              self.pyfhel_he_object.to_bytes_rotate_key()])
        self.add_benchmark_result(BYTES_PUBLIC_AND_EVAL_KEYS, self.get_size_of_files(
            self.write_directory, ['public_and_eval_keys.by']
        ))

    def compute_and_write_matching_results(self, result_type):
        # After decrypting and predicting the cell level matches, compute cluster level matches.

        # Compute the percentage of each original cluster, how it is distributed over the clusters of the other clients
        # dataset.
        # Original: Anonymized original labels
        # Predicted: Predicted anonymized labels of labels from collaborating client.
        if result_type not in [R_ONE_WAY_MATCH, R_1_TO_1_ONE_WAY_MATCH, R_TWO_WAY_MATCH, R_1_TO_1_TOW_WAY_MATCH]:
            raise ValueError('Invalid result_type. Select one of the following {}'.format(
                [R_ONE_WAY_MATCH, R_1_TO_1_ONE_WAY_MATCH, R_TWO_WAY_MATCH, R_1_TO_1_TOW_WAY_MATCH]))

        matching_direction_c1_c2 = '{}_{}'.format(self.client1_name, self.client2_name)
        matching_direction_c2_c1 = '{}_{}'.format(self.client2_name, self.client1_name)

        # (2.1) Compute cluster_level matches based on type
        if result_type == R_ONE_WAY_MATCH or result_type == R_TWO_WAY_MATCH:
            with self.benchmark(COMPUTE_ONE_WAY_MATCHES_C1_C2):
                client1_matches = self.get_one_way_matches(
                    AggregationServer.compute_percentages_cluster_level(
                        original_query_labels=self.labels_client1,
                        predicted_query_labels=self.cell_level_matches[matching_direction_c1_c2]
                    )
                )
            with self.benchmark(COMPUTE_ONE_WAY_MATCHES_C2_C1):
                client2_matches = self.get_one_way_matches(
                    AggregationServer.compute_percentages_cluster_level(
                        original_query_labels=self.labels_client2,
                        predicted_query_labels=self.cell_level_matches[matching_direction_c2_c1]
                    )
                )

            # Aggregation phase for simple two-way matches
            if result_type == R_TWO_WAY_MATCH:
                with self.benchmark(COMPUTE_TWO_WAY_MATCHES):
                    client1_matches = self.get_two_way_matches(client1_matches, client2_matches)
                    client2_matches = {tuple(reversed(t)) for t in client1_matches}

        elif result_type == R_1_TO_1_ONE_WAY_MATCH or result_type == R_1_TO_1_TOW_WAY_MATCH:
            with self.benchmark(COMPUTE_1_TO_1_ONE_WAY_MATCHES_C1_C2):
                client1_matches = self.get_one_to_one_matches(
                    self.get_one_way_matches(
                        AggregationServer.compute_percentages_cluster_level(
                            original_query_labels=self.labels_client1,
                            predicted_query_labels=self.cell_level_matches[matching_direction_c1_c2])))
            with self.benchmark(COMPUTE_1_TO_1_ONE_WAY_MATCHES_C2_C1):
                client2_matches = self.get_one_to_one_matches(
                    self.get_one_way_matches(
                        AggregationServer.compute_percentages_cluster_level(
                            original_query_labels=self.labels_client2,
                            predicted_query_labels=self.cell_level_matches[matching_direction_c2_c1])))

            # Aggregation phase for 1 to 1 two-way matches
            if result_type == R_1_TO_1_TOW_WAY_MATCH:
                with self.benchmark(COMPUTE_1_TO_1_TWO_WAY_MATCHES):
                    client1_matches = self.get_two_way_matches(client1_matches, client2_matches)
                    client2_matches = {tuple(reversed(t)) for t in client1_matches}

        # Check if any cluster level matches left, otherwise set -1 as indicator for no cluster level match
        if not client1_matches:
            client1_matches = {-1}
        if not client2_matches:
            client2_matches = {-1}

        # (5.1) Write results for client1
        start_time = time.time()
        with open(self.write_directory / '{}_{}.by'.format(self.client1_name, result_type), 'wb') as f:
            f.write(np.array(list(client1_matches)).tobytes())
        # Save benchmarks depending on result type
        if result_type == R_ONE_WAY_MATCH:
            self.add_benchmark_result(WRITE_ONE_WAY_C1, time.time() - start_time)
            self.add_benchmark_result(BYTES_ONE_WAY_C1, self.get_size_of_files(
                self.write_directory, ['{}_{}.by'.format(self.client1_name, result_type)]
            ))
        elif result_type == R_1_TO_1_ONE_WAY_MATCH:
            self.add_benchmark_result(WRITE_1_TO_1_ONE_WAY_C1, time.time() - start_time)
            self.add_benchmark_result(BYTES_1_TO_1_ONE_WAY_C1, self.get_size_of_files(
                self.write_directory, ['{}_{}.by'.format(self.client1_name, result_type)]
            ))
        elif result_type == R_TWO_WAY_MATCH:
            self.add_benchmark_result(WRITE_TWO_WAY_C1, time.time() - start_time)
            self.add_benchmark_result(BYTES_TWO_WAY_C1, self.get_size_of_files(
                self.write_directory, ['{}_{}.by'.format(self.client1_name, result_type)]
            ))
        elif result_type == R_1_TO_1_TOW_WAY_MATCH:
            self.add_benchmark_result(WRITE_1_TO_1_TWO_WAY_C1, time.time() - start_time)
            self.add_benchmark_result(BYTES_1_TO_1_TWO_WAY_C1, self.get_size_of_files(
                self.write_directory, ['{}_{}.by'.format(self.client1_name, result_type)]
            ))

        # (5.2) Write results for client2
        start_time = time.time()
        with open(self.write_directory / '{}_{}.by'.format(self.client2_name, result_type), 'wb') as f:
            f.write(np.array(list(client2_matches)).tobytes())
        if result_type == R_ONE_WAY_MATCH:
            self.add_benchmark_result(WRITE_ONE_WAY_C2, time.time() - start_time)
            self.add_benchmark_result(BYTES_ONE_WAY_C2, self.get_size_of_files(
                self.write_directory, ['{}_{}.by'.format(self.client1_name, result_type)]
            ))
        elif result_type == R_1_TO_1_ONE_WAY_MATCH:
            self.add_benchmark_result(WRITE_1_TO_1_ONE_WAY_C2, time.time() - start_time)
            self.add_benchmark_result(BYTES_1_TO_1_ONE_WAY_C2, self.get_size_of_files(
                self.write_directory, ['{}_{}.by'.format(self.client1_name, result_type)]
            ))
        elif result_type == R_TWO_WAY_MATCH:
            self.add_benchmark_result(WRITE_TWO_WAY_C2, time.time() - start_time)
            self.add_benchmark_result(BYTES_TWO_WAY_C2, self.get_size_of_files(
                self.write_directory, ['{}_{}.by'.format(self.client1_name, result_type)]
            ))
        elif result_type == R_1_TO_1_TOW_WAY_MATCH:
            self.add_benchmark_result(WRITE_1_TO_1_TWO_WAY_C2, time.time() - start_time)
            self.add_benchmark_result(BYTES_1_TO_1_TWO_WAY_C2, self.get_size_of_files(
                self.write_directory, ['{}_{}.by'.format(self.client1_name, result_type)]
            ))

    @staticmethod
    def compute_percentages_cluster_level(
            original_query_labels: NDArray[np.int64],
            predicted_query_labels: NDArray[np.int64]) -> dict[np.int64, dict[np.int64, float]]:
        """
        For each original class, e.g, cell type cluster, it computes the percentages
        of how many of the member cells of that query cluster are mapped to each of
        the clusters in the reference dataset.
        :param original_query_labels:   Cell types of each of the cells before matching
        :param predicted_query_labels:  Cell types of each of the cells after matching.
                                        Predicted labels are cluster types (encoded labels) from the
                                        reference dataset.
        :return: A dictionary that stores for each cell type a dictionary of percentages,
        i.e., {query_cell_type: {reference_cell_type: percentage}}.
        """
        # Instantiate the dictionary
        class_sizes = Counter(original_query_labels)
        percentage_dict = {
            org: {pred: 0 for pred in predicted_query_labels} for org in set(original_query_labels)
        }

        # Count occurrences
        for org, pred in zip(original_query_labels, predicted_query_labels):
            percentage_dict[org][pred] += 1

        for org_class in set(original_query_labels):
            for pred_class in set(predicted_query_labels):
                percentage_dict[org_class][pred_class] /= class_sizes[org_class]
        return percentage_dict

    def get_one_way_matches(
            self,
            percentage_dictionary: dict[np.int64, dict[np.int64, float]]) -> set[(np.int64, np.int64)]:
        """
        :return: A set of tuples, where each tuple represents the derived match, e.g, (1,4) is a match from
        cluster 1 to cluster 4.
        """
        return {(org, pred)
                for org in percentage_dictionary.keys()
                for pred in percentage_dictionary[org].keys()
                if percentage_dictionary[org][pred] > self.cluster_level_threshold and pred != -1}

    @staticmethod
    def get_two_way_matches(one_way_matches1: set[(np.int64, np.int64)], one_way_matches2: set[(np.int64, np.int64)]) \
            -> set[(np.int64, np.int64)]:
        """
        Identify clusters that match in both directions.

        :return: Returns set of tuples, where each tuple is a two-way match.
        """
        # Matching tuples of both arguments are mirrored. That is, if there is a two-way match 5 <-> 8,
        # then ``one_way_matches1`` contains the tuple (5,8) and ``one_way_matches2`` contains the tuple (8,5).
        return {
            (cluster_type_1, cluster_type_2)
            for cluster_type_1, cluster_type_2 in one_way_matches1
            if (cluster_type_2, cluster_type_1) in one_way_matches2}

    @staticmethod
    def get_one_to_one_matches(one_way_matches: set[tuple[np.int64, np.int64]]) -> set[tuple[np.int64, np.int64]]:
        """
        Filters out the many-to-one and one-to-many matches.
        :param one_way_matches:
        :return:
        """
        if not one_way_matches:
            return set()
        original_clusters, predicted_clusters = zip(*one_way_matches)

        return {
            (cluster_type_1, cluster_type_2)
            for cluster_type_1, cluster_type_2 in one_way_matches
            # filter many-to-one
            if sum(1 for cluster in original_clusters if cluster == cluster_type_1) == 1
            # filter one-to-many
            if sum(1 for cluster in predicted_clusters if cluster == cluster_type_2) == 1
        }
