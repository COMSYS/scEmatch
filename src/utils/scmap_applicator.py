from omegaconf import DictConfig

from ..scmap_client_server import Client, SimilarityServer, AggregationServer
from ..scmap_client_server.benchmarks import BenchmarkCollector
from ..scmap_client_server.constants import *


class Applicator:

    def __init__(self, benchmark_collector: BenchmarkCollector):

        self.benchmark_collector = benchmark_collector

    def apply_scmap(
            self,
            cfg: DictConfig,
            client1_ref_data_file, client1_query_data_file,
            client2_ref_data_file, client2_query_data_file,
            cluster_type_key_1, cluster_type_key_2,
            mode, ckks_params, sent_data_dir):
        """
        Apply scmap. Expects file paths to aligned datasets.
        :param cfg:
        :param client1_ref_data_file:
        :param client1_query_data_file:
        :param client2_ref_data_file:
        :param client2_query_data_file:
        :param mode: Mode to run the prototype in. {'simple', 'homomorphic'}
        :param ckks_params:
        :param cluster_type_key_1:
        :param cluster_type_key_2:
        :param sent_data_dir:
        :return:
        """

        # (0.) Instantiation -------------------------------------------------------------------------------------- ####
        # Instantiate first client
        client1 = Client(
            client_name=cfg.client1.name,
            write_directory=sent_data_dir / cfg.client1.name,
            aggregation_server_write_directory=sent_data_dir / cfg.aggregation_server.name,
            adata_reference_path=client1_ref_data_file,
            adata_query_path=client1_query_data_file,
            cluster_type_key=cluster_type_key_1,
            benchmark_collector=self.benchmark_collector,
            seed=cfg.run,
            num_cells=cfg.dataset.client1_cell_counts
        )

        # Instantiate second client
        client2 = Client(
            client_name=cfg.client2.name,
            write_directory=sent_data_dir / cfg.client2.name,
            aggregation_server_write_directory=sent_data_dir / cfg.aggregation_server.name,
            adata_reference_path=client2_ref_data_file,
            adata_query_path=client2_query_data_file,
            cluster_type_key=cluster_type_key_2,
            benchmark_collector=self.benchmark_collector,
            seed=cfg.run,
            num_cells=cfg.dataset.client2_cell_counts
        )

        similarity_server = SimilarityServer(
            server_name=cfg.similarity_server.name,
            write_directory=sent_data_dir / cfg.similarity_server.name,
            client1_name=client1.name,
            client1_write_directory=client1.write_directory,
            client2_name=client2.name,
            client2_write_directory=client2.write_directory,
            aggregation_server_write_directory=sent_data_dir / cfg.aggregation_server.name,
            benchmark_collector=self.benchmark_collector
        )

        aggregation_server = AggregationServer(
            server_name=cfg.aggregation_server.name,
            write_directory=sent_data_dir / cfg.aggregation_server.name,
            client1_name=client1.name,
            client1_write_directory=client1.write_directory,
            client2_name=client2.name,
            client2_write_directory=client2.write_directory,
            similarity_server_write_directory=similarity_server.write_directory,
            scmap_threshold=cfg.scmap.threshold,
            cluster_level_threshold=cfg.scmap.cluster_level_threshold,
            benchmark_collector=self.benchmark_collector
        )

        # (1.) Key distribution ----------------------------------------------------------------------------------- ####
        if mode == S_HOMOMORPHIC:
            n_exp = ckks_params['n_exp']
            scale_exp = ckks_params['scale_exp']
            qi_sizes = ckks_params['qi_sizes']

            he_params = {
                'n': 2 ** n_exp,
                'scale': 2 ** scale_exp,
                'qi_sizes': qi_sizes
            }

            # Generate and distribute keys
            with aggregation_server.benchmark(COMPLETE_RUNTIME):
                aggregation_server.generate_he_keys_and_context(he_params)
                aggregation_server.write_public_key()
                aggregation_server.write_public_and_eval_keys()

        # (2.) Run hosts in selected mode ------------------------------------------------------------------------- ####
        client1_process = client1.start_as_process(mode)
        client2_process = client2.start_as_process(mode)
        client1_process.join()
        client2_process.join()

        with similarity_server.benchmark(COMPLETE_RUNTIME):
            similarity_server.run(mode)

        with aggregation_server.benchmark(COMPLETE_RUNTIME):
            aggregation_server.run(mode)

        # (3.) Result computation and sending to clients ---------------------------------------------------------- ####
        for result_type in [R_ONE_WAY_MATCH, R_1_TO_1_ONE_WAY_MATCH, R_TWO_WAY_MATCH, R_1_TO_1_TOW_WAY_MATCH]:
            # Individual benchmark for every result type.
            with aggregation_server.benchmark(COMPLETE_RUNTIME):
                aggregation_server.compute_and_write_matching_results(result_type)

            with client1.benchmark(COMPLETE_RUNTIME):
                client1.read_matching_results(result_type)

            with client2.benchmark(COMPLETE_RUNTIME):
                client2.read_matching_results(result_type)

        self.benchmark_collector.post_hook(client1, only_matching_results=True)
        self.benchmark_collector.post_hook(client2, only_matching_results=True)
        self.benchmark_collector.post_hook(aggregation_server)
        self.benchmark_collector.post_hook(similarity_server)

        if 'extra_thresholds' in cfg.scmap:
            for scmap_threshold in cfg.scmap.extra_thresholds:
                aggregation_server.scmap_threshold = scmap_threshold

                matching_direction_1 = f'{client1.name}_{client2.name}'
                matching_direction_2 = f'{client2.name}_{client1.name}'

                for matching_direction in [matching_direction_1, matching_direction_2]:
                    similarity_values = aggregation_server.similarity_values[matching_direction]

                    cell_level_matches = aggregation_server.match_cells(
                        similarity_values[COSINE],
                        similarity_values[PEARSONR],
                        similarity_values[SPEARMANR]
                    )
                    aggregation_server.cell_level_matches[matching_direction] = cell_level_matches

                self.benchmark_collector.post_hook_cell_matches_all_thresholds(aggregation_server, scmap_threshold)
