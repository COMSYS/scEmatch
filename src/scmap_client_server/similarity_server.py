import json
import logging
import signal
from functools import partial
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
from threading import Thread

import math
from pathlib import Path
import numpy as np
from Pyfhel import PyCtxt, Pyfhel
import time

from src.scmap_client_server import server
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from numpy.typing import NDArray
from typing import Optional
from src.scmap_client_server.constants import *
from src.scmap_client_server.host import Benchmark

# Benchmarks
READ_REF_C1 = 'read-ref-c1'
READ_QUERY_C1 = 'read-query-c1'
READ_CENTROIDS = 'read-centroids'
COMPUTE_SIMILARITIES_C1_C2 = 'compute-similarities-c1-to-c2'
WRITE_SIMILARITIES_C1_C2 = 'write-similarities-c1-to-c2'
BYTES_RESULT_SIMILARITIES_C1_C2 = 'bytes-result-similarities-c1-to-c2'

READ_REF_C2 = 'read-ref-c2'
READ_QUERY_C2 = 'read-query-c2'
COMPUTE_SIMILARITIES_C2_C1 = 'compute-similarities-c2-to-c1'
WRITE_SIMILARITIES_C2_C1 = 'write-similarities-c2-to-c1'
BYTES_RESULT_SIMILARITIES_C2_C1 = 'bytes-result-similarities-c2-to-c1'

READ_PUBLIC_AND_EVAL_KEYS = 'read-public-and-eval-keys'
READ_META_C1 = 'read-meta-c1'
READ_META_C2 = 'read-meta-c2'

worker = None


class SimilarityServerWorker:

    @staticmethod
    def worker_init(write_dir, benchmark_queue: mp.Queue):
        global worker
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        worker = SimilarityServerWorker(write_dir, benchmark_queue)

    def __init__(self, aggregation_server_write_directory: Path, benchmark_queue: mp.Queue):
        self.aggregation_server_write_directory = aggregation_server_write_directory
        self.benchmark_queue = benchmark_queue
        self.pyfhel_he_object: Pyfhel = Pyfhel()
        self.logger = logging.getLogger("sim-worker")
        self.deserializer = lambda x: PyCtxt(pyfhel=self.pyfhel_he_object, bytestring=x)
        self.centroids = {}

        self.load_keys()
        self.ciphertext_slots = self.pyfhel_he_object.get_nSlots()

    def add_benchmark_result(self, key, value):
        self.benchmark_queue.put_nowait(("sim-worker", key, value))

    def benchmark(self, key):
        return Benchmark(self.benchmark_queue, "sim-worker", key)

    def load_keys(self):
        with self.benchmark(READ_PUBLIC_AND_EVAL_KEYS):
            with open(self.aggregation_server_write_directory / 'public_and_eval_keys.by', 'rb') as f:
                for m in [
                    self.pyfhel_he_object.from_bytes_context,
                    self.pyfhel_he_object.from_bytes_public_key,
                    self.pyfhel_he_object.from_bytes_relin_key,
                    self.pyfhel_he_object.from_bytes_rotate_key
                ]:
                    length_byte_string = f.read(4)
                    length = int.from_bytes(length_byte_string, byteorder='big')
                    m(f.read(length))

    def load_centroids(self, method, reference_directory):
        if method not in self.centroids:
            filename = {
                COSINE: COSINE_REFERENCE_DATA_NAME,
                PEARSONR: PEARSONR_REFERENCE_DATA_NAME,
                SPEARMANR: SPEARMANR_REFERENCE_DATA_NAME
            }.get(method)
            with self.benchmark(READ_CENTROIDS):
                self.centroids[f"{reference_directory}_{method}"] = SimilarityServer.read_bytes(
                    reference_directory / f'{filename}.by', transformer=self.deserializer)

    def calculate_similarity_batch(self, method, reference_directory, cumulative_add_operations, centroid_length,
                                   batch):
        computation_time_in_loop = 0
        shift_number = 0
        self.load_centroids(method, reference_directory)
        centroid_key = f"{reference_directory}_{method}"
        similarity_ctxts = [
            self.pyfhel_he_object.encryptFrac(np.array([0] * self.ciphertext_slots, dtype=np.float64))
            for _ in range(len(self.centroids[centroid_key]))
        ]
        n_unit = self.ciphertext_slots // centroid_length  # For unit vector construction
        for ciphertext in batch:
            ciphertext = self.deserializer(ciphertext)
            start_compute_time = time.time()
            for i, centroid in enumerate(self.centroids[centroid_key]):
                # Similarity ctxt containing redundant similarity values
                redundant_similarity_ctxt = self.compute_similarity_measure_encrypted(
                    ciphertext, centroid, steps=cumulative_add_operations)

                # Construct a (n-times concatenation) of the first unit vector (n = n_unit)
                unit_vector = np.hstack([
                    [np.array([1 if i % centroid_length == 0 else 0 for i in range(centroid_length)],
                              dtype=np.float64)]
                    for _ in range(n_unit)
                ]).flatten()

                # Encode the unit vector to pyfhel plain text
                unit_vector = self.pyfhel_he_object.encode(unit_vector)
                redundant_similarity_ctxt, unit_vector = self.pyfhel_he_object.align_mod_n_scale(
                    redundant_similarity_ctxt, unit_vector)

                # Zero out redundant values (Necessary as otherwise sums of vectors lead to false results)
                self.pyfhel_he_object.multiply_plain(redundant_similarity_ctxt, unit_vector)
                self.pyfhel_he_object.rescale_to_next(redundant_similarity_ctxt)

                # Rotate zeroed out ciphertext to the right (rotate behaves opposite to numpys roll)
                self.pyfhel_he_object.rotate(redundant_similarity_ctxt, -shift_number)

                # Align mod and scale + rounding of scaling factors
                redundant_similarity_ctxt, similarity_ctxts[i] = self.pyfhel_he_object.align_mod_n_scale(
                    redundant_similarity_ctxt, similarity_ctxts[i])

                # Add to the similarity_ctxt packing all similarity values
                self.pyfhel_he_object.add(similarity_ctxts[i], redundant_similarity_ctxt)

            # Increment shift number
            shift_number += 1
            computation_time_in_loop += time.time() - start_compute_time

        # If the matrix similarity_ctxts is filled up with similarity values (fully packed), append
        # matrix to the result_similarity_ctxts.
        start = time.time()
        result_similarity_ctxts = list(list() for _ in range(len(self.centroids[centroid_key])))
        if shift_number == centroid_length:
            for i in range(len(self.centroids[centroid_key])):
                result_similarity_ctxts[i].append(similarity_ctxts[i])
                # Reset initial similarity ciphertext
                similarity_ctxts[i] = self.pyfhel_he_object.encryptFrac(
                    np.array([0] * self.ciphertext_slots, dtype=np.float64))
        else:
            for i, similarity_ctxt in enumerate(similarity_ctxts):
                result_similarity_ctxts[i].append(similarity_ctxt)
        computation_time_in_loop += time.time() - start

        for i in range(len(result_similarity_ctxts)):
            for j in range(len(result_similarity_ctxts[i])):
                result_similarity_ctxts[i][j] = result_similarity_ctxts[i][j].to_bytes()

        return computation_time_in_loop, result_similarity_ctxts

    def compute_similarity_measure_encrypted(self, cell_ctxt, centroid_ctxt, steps):
        """
        Encrypted version of compute_similarity_measure.
        :param cell_ctxt: Contains n cell vectors concatenated (cell_ctxt is 1D); n >= 1.
        :param centroid_ctxt: Contains n cell vectors concatenated (cell_ctxt is 1D); n >= 1.
        :param steps: Specifies the number of rotate operations to perform.
        """
        similarity_ctxt = self.pyfhel_he_object.multiply(cell_ctxt, centroid_ctxt,
                                                         in_new_ctxt=True)  # in place in cell_ctxt
        self.pyfhel_he_object.relinearize(similarity_ctxt)
        self.pyfhel_he_object.rescale_to_next(similarity_ctxt)

        # Cumulative sum
        similarity_ctxt_copy = similarity_ctxt.copy()
        for i in range(steps):
            self.pyfhel_he_object.rotate(similarity_ctxt_copy, 2 ** i)
            self.pyfhel_he_object.add(similarity_ctxt, similarity_ctxt_copy)
            similarity_ctxt_copy = similarity_ctxt.copy()
        return similarity_ctxt


def calculate_similarity_batch(method, reference_directory, cumulative_add_operations, centroid_length, batch):
    global worker
    return worker.calculate_similarity_batch(method, reference_directory, cumulative_add_operations, centroid_length,
                                             batch)


class SimilarityServer(server.Server):
    """
    The similarity server computes the similarities between each pair of reference and query data for both clients.

            Client1                 Client2
            ------------------------------------ Matching client1's aligned query data to client2's reference data.
            query_cosine    ->      ref_cosine
            query_pearsonr  ->      ref_pearsonr
            query_spearmanr ->      ref_spearmanr

            ------------------------------------ Matching client2's aligned query data to client1's reference data.
            ref_cosine      ->      query_cosine
            ref_pearsonr    ->      query_pearsonr
            ref_spearmanr   ->      query_spearmanr
    """

    def __init__(
            self,
            server_name: str,
            write_directory: Path,
            client1_name: str,
            client1_write_directory: Path,
            client2_name: str,
            client2_write_directory: Path,
            aggregation_server_write_directory: Path,
            benchmark_collector,
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

        # Information of aggregation server
        self.aggregation_server_write_directory = aggregation_server_write_directory

        # Meta information of both clients
        self.client1_centroid_length = None
        self.client2_centroid_length = None

        # CKKS information
        self.ciphertext_slots = None

        # Creates the write directory to write its data in. Other hosts access files from this directory.
        self.create_directory()

        self.logger = logging.getLogger(server_name)

    # --- ENTRY POINT --- #
    def run(self, strategy: str):
        """
        Entry point level run method. Select strategy and encrypted from top level, i.e., script running the scenario.
        :param strategy: {'simple', 'homomorphic'}
        :return:
        """

        if strategy == S_SIMPLE:
            self.logger.info("running plaintext variant")
            self.run_simple()
        elif strategy == S_HOMOMORPHIC:
            self.logger.info("running homomorphic variant")
            self.run_homomorphic()
        else:
            raise ValueError("Strategy not supported.")

    def run_simple(self):

        # Matching direction specification client1 -> client2
        specification_c1_c2 = {
            'reference_directory': self.client2_write_directory,  # Read reference data from client2's write directory
            'query_directory': self.client1_write_directory,  # Read query data from client1's write directory
            'matching_direction': '{}_{}'.format(self.client1_name, self.client2_name),
        }

        # Matching direction specification client2 -> client1
        specification_c2_c1 = {
            'reference_directory': self.client1_write_directory,
            'query_directory': self.client2_write_directory,
            'matching_direction': '{}_{}'.format(self.client2_name, self.client1_name),
        }

        # Matching direction client1 -> client2.
        self.logger.info("computing c1 -> c2 similarity scores")
        self.match_one_direction_simple(**specification_c1_c2)

        # Matching direction client2 -> client1.
        self.logger.info("computing c2 -> c1 similarity scores")
        self.match_one_direction_simple(**specification_c2_c1)

    def match_one_direction_simple(
            self,
            reference_directory: Path,
            query_directory: Path,
            matching_direction: str,
    ):
        """
        :param reference_directory: Directory to read the reference data from (centroids)
        :param query_directory: Directory to read the query data from (gene x cell)
        :param matching_direction: String identifier for several purposes (Reading and naming files + benchmarking)
        :return:
        """

        # Read centroids of current reference client
        self.logger.info("reading centroids")
        with self.benchmark(READ_REF_C2 if matching_direction == '{}_{}'.format(self.client1_name,
                                                                                self.client2_name) else READ_REF_C1):
            centroids = self.read_array(
                Path(reference_directory, '{}.npy'.format(REFERENCE_DATA_SUFFIX))
            )

        # Read query cells of current query client
        self.logger.info("reading query data")
        with self.benchmark(READ_QUERY_C2 if matching_direction == '{}_{}'.format(self.client1_name,
                                                                                  self.client2_name) else READ_QUERY_C1):
            cell_x_gene = self.read_array(
                Path(query_directory, '{}.npy'.format(QUERY_DATA_SUFFIX))
            )

        # Compute similarities
        self.logger.info("computing similarities")
        with self.benchmark(COMPUTE_SIMILARITIES_C1_C2 if matching_direction == '{}_{}'.format(self.client1_name,
                                                                                               self.client2_name) else COMPUTE_SIMILARITIES_C2_C1):
            similarity_matrices = SimilarityServer.compute_similarities_simple(cell_x_gene, centroids)

        self.logger.info("storing results")
        with self.benchmark(WRITE_SIMILARITIES_C1_C2 if matching_direction == '{}_{}'.format(self.client1_name,
                                                                                             self.client2_name) else WRITE_SIMILARITIES_C2_C1):
            # Write file
            file_name = '{}.npz'.format(matching_direction)
            np.savez(
                Path(self.write_directory, file_name),
                cosine=similarity_matrices[COSINE],
                pearsonr=similarity_matrices[PEARSONR],
                spearmanr=similarity_matrices[SPEARMANR]
            )
        if matching_direction == '{}_{}'.format(self.client1_name, self.client2_name):
            # Read size of the written file (Specify file(s) as the directory collects all written files)
            sent_bytes = self.get_size_of_files(self.write_directory, [file_name])  # Only one file is written
            self.add_benchmark_result(BYTES_RESULT_SIMILARITIES_C1_C2, sent_bytes)
        else:
            # Read size of the written file (Specify file(s) as the directory collects all written files)
            sent_bytes = self.get_size_of_files(self.write_directory, [file_name])  # Only one file is written
            self.add_benchmark_result(BYTES_RESULT_SIMILARITIES_C2_C1, sent_bytes)

    @staticmethod
    def compute_similarities_simple(query_data: np.array, reference_data: np.array) -> {str: NDArray[np.float64]}:
        """
        :param query_data:
        :param reference_data:
        :return: A dictionary with keys 'cosine', 'pearsonr', 'spearmanr' and items the corresponding similarity
                 2D arrays. For one array, the i-th column of the array corresponds to the i-th centroid and the j-th
                 row corresponds to the j-th cell, i.e., in the cell (i,j) the similarity value for the i-th cell j-th
                 centroid is given.
        """
        similarity_data = dict()
        for method in [COSINE, PEARSONR, SPEARMANR]:
            if method == COSINE:
                similarity_matrix = 1 - cdist(query_data, reference_data, metric='cosine')
            elif method == PEARSONR:
                similarity_matrix = 1 - cdist(query_data, reference_data, metric='correlation')
            elif method == SPEARMANR:
                ranked_query = rankdata(query_data, axis=1, method='average')
                ranked_reference = rankdata(reference_data, axis=1, method='average')
                similarity_matrix = 1 - cdist(ranked_query, ranked_reference, metric='correlation')
            else:
                raise ValueError(method)
            similarity_data[method] = similarity_matrix
        return similarity_data

    def run_homomorphic(self):
        """
        Before running the homomorphic version, metadata has to be read.
        :return:
        """

        # (1) Read metadata of both clients
        self.logger.info("reading metadata")
        with self.benchmark(READ_META_C1):
            with open(self.client1_write_directory / 'meta_similarity_server.json', 'r') as f:
                meta_information = json.load(f)
                self.client1_centroid_length = meta_information[M_CENTROID_LENGTH]

        with self.benchmark(READ_META_C2):
            with open(self.client2_write_directory / 'meta_similarity_server.json', 'r') as f:
                meta_information = json.load(f)
                self.client2_centroid_length = meta_information[M_CENTROID_LENGTH]

        # (2.) Perform actual computations
        pool = mp.Pool(int(mp.cpu_count()), initializer=SimilarityServerWorker.worker_init,
                       initargs=(self.aggregation_server_write_directory, self.benchmark_collector.queue))

        # Matching direction client1 -> client2.
        self.logger.info("computing c1 -> c2 similarity scores")
        t_c1c2 = Thread(target=self.match_one_direction_homomorphic, args=[
            self.client2_write_directory,
            self.client1_write_directory,
            '{}_{}'.format(self.client1_name, self.client2_name),  # name of result file
            self.client2_centroid_length,
            pool
        ])
        t_c1c2.start()
        self.logger.info("computing c2 -> c1 similarity scores")
        t_c2c1 = Thread(target=self.match_one_direction_homomorphic, args=[
            self.client1_write_directory,
            self.client2_write_directory,
            '{}_{}'.format(self.client2_name, self.client1_name),  # name of result file
            self.client1_centroid_length,
            pool
        ])
        t_c2c1.start()
        t_c1c2.join()
        t_c2c1.join()

        pool.terminate()

    def match_one_direction_homomorphic(
            self,
            reference_directory: Path,
            query_directory: Path,
            matching_direction: str,
            centroid_length: int,
            pool: Optional[mp.Pool] = None):
        """
        Reads data, computes similarities and writes computed similarities.
        :param reference_directory: Directory to read the centroid ciphertexts from.
        :param query_directory: Directory to read the query cell ciphertexts from.
        :param matching_direction: String identifier for several purposes (Reading and naming files + benchmarking)
        :param centroid_length: Length of a centroid (as multiple cells and centroids are packed into one ciphertext)
        :param pool:
        :return:
        """
        is_c1_c2 = matching_direction == f'{self.client1_name}_{self.client2_name}'

        params = [(query_directory, reference_directory, centroid_length, method, matching_direction, pool) for method
                  in [COSINE, PEARSONR, SPEARMANR]]
        with ThreadPool(len(params)) as p:
            similarities = dict(p.starmap(self.calculate_similarities, params))

        self.logger.info("writing results")
        with self.benchmark(WRITE_SIMILARITIES_C1_C2 if is_c1_c2 else WRITE_SIMILARITIES_C2_C1):
            # Write similarities per similarity list (One list per scaling method -> one file per scaling method)
            for file_name, similarities_list in similarities.items():
                file_path = Path(self.write_directory, file_name)
                with open(file_path, 'wb') as f:
                    for bytes_to_write in similarities_list:
                        f.write(self.prepend_length(bytes_to_write))

        # Benchmarks for current matching direction
        if is_c1_c2:
            # Read size of the written file (Specify file(s) as the directory collects all written files)
            sent_bytes = self.get_size_of_files(self.write_directory, list(similarities.keys()))
            self.add_benchmark_result(BYTES_RESULT_SIMILARITIES_C1_C2, sent_bytes)
        else:
            # Read size of the written file (Specify file(s) as the directory collects all written files)
            sent_bytes = self.get_size_of_files(self.write_directory, list(similarities.keys()))
            self.add_benchmark_result(BYTES_RESULT_SIMILARITIES_C2_C1, sent_bytes)

    def calculate_similarities(
            self,
            query_directory: Path,
            reference_directory: Path,
            centroid_length: int,
            method: str,
            matching_direction,
            pool: Optional[mp.Pool] = None
    ) -> [PyCtxt]:
        """
        Calculates similarities between centroids and query cells of one similarity method. Returns a list
        of ctxts containing the similarities. In particular, each ctxt is of length ciphertext_slots and
        stores the similarity values per cell for one centroid in the following order:

        Example: centroid_length := n=512, self.ciphertext_slots = m=2048
                    [0, 0+n, 0+n*2, 0+n*3, 1, 1+n, 1+n*2, 1+n*3, ..., 511, 511+n, 511+n*2, 511+n*3]
            i.e.,   [0, 512, 1024,  1536,  1, 513, 1025,  1537      , 511, 1023,  1535,    2047   ]

        The aggregation server, with the corresponding read similarities method, restores the correct order.
        :param query_directory: The directory containing the query cells for the current query client.
        :param reference_directory:  The directory containing the centroids for the current reference client.
        :param centroid_length: The adjusted centroid length, e.g., 128, 256, 512, 1024, ...
        :param method: String identifier to select the correct query data file (Using the naming convention.)
        :param matching_direction: String identifier for several purposes (Reading and naming files + benchmarking)
        :return: List of ciphertexts containing similarity values in the cell order given above. Ciphertexts themselves
                 are ordered by centroids. If 20 ciphertexts for 5 centroids are returned, the first 4 similarity
                 ciphertexts correspond to the first centroid.
        """
        # Select the correct query file name
        if method == COSINE:
            file_name = COSINE_QUERY_DATA_NAME
        elif method == PEARSONR:
            file_name = PEARSONR_QUERY_DATA_NAME
        elif method == SPEARMANR:
            file_name = SPEARMANR_QUERY_DATA_NAME
        else:
            raise ValueError("Invalid similarity method.")

        # Construct the file path
        file_path = Path(query_directory, '{}.by'.format(file_name))

        # Other necessary information
        cumulative_add_operations = math.floor(math.log2(centroid_length))

        start_time_for_complete_loop = time.time()  # The complete time, including read and deserialize

        self.logger.info("calculating similarities")
        iterator = SimilarityServer.read_bytes_generator_parallel(file_path, centroid_length)

        fn = partial(calculate_similarity_batch, method, reference_directory, cumulative_add_operations,
                     centroid_length)
        result_list = pool.map(fn, iterator)

        runtimes, result_similarity_ctxts = tuple(list(zip(*result_list)))

        centroid_count = len(result_similarity_ctxts[0])
        results = [[] for _ in range(centroid_count)]
        for ctxts in result_similarity_ctxts:
            for i, ctxt in enumerate(ctxts):
                results[i] += ctxt
        result_similarity_ctxts = results

        computation_time_in_loop = max(runtimes)

        # Benchmark time for computing similarity values
        # Benchmark time for reading the query ciphertexts
        if matching_direction == '{}_{}'.format(self.client1_name, self.client2_name):
            self.add_benchmark_result(COMPUTE_SIMILARITIES_C1_C2, computation_time_in_loop)
            self.add_benchmark_result(READ_QUERY_C1,
                                      time.time() - start_time_for_complete_loop - computation_time_in_loop)
        else:
            self.add_benchmark_result(COMPUTE_SIMILARITIES_C2_C1, computation_time_in_loop)
            self.add_benchmark_result(READ_QUERY_C2,
                                      time.time() - start_time_for_complete_loop - computation_time_in_loop)

        # Final data processing
        with self.benchmark(COMPUTE_SIMILARITIES_C1_C2 if matching_direction == '{}_{}'.format(self.client1_name,
                                                                                               self.client2_name) else COMPUTE_SIMILARITIES_C2_C1):
            # flatten the similarity ctxts
            result_similarity_ctxts = [ctxt for l in result_similarity_ctxts for ctxt in l]
        return f'{matching_direction}_{method}.by', result_similarity_ctxts

    @staticmethod
    def compute_similarity_measures_simple(query_client, ref_client, data, query_type, ref_type):
        """
        Computes similarity scores for plaintext variant
        """
        results = list()
        for centroid in data[ref_client][ref_type]:
            results.append(data[query_client][query_type].multiply(centroid).sum(axis=1))
        return np.hstack(results)
