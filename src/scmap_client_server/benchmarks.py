import copy
import gzip
import json
import logging
import queue
import shutil
import signal
from pathlib import Path

import numpy as np
import pandas as pd
from joblib._multiprocessing_helpers import mp
from numpy.typing import NDArray
from omegaconf import DictConfig

from . import Client, AggregationServer
from .constants import *


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class BenchmarkCollector:
    def __init__(self, cfg: DictConfig, artifact_path: Path):
        self.artifact_path = artifact_path
        artifact_path.mkdir(parents=True, exist_ok=True)
        self.queue = mp.Queue()
        self.cfg = cfg
        self.consumer = None
        self.benchmark_data = {}
        self.mapping_data = {}
        self.logger = logging.getLogger("benchmark")

    def run_exists(self):
        """check if artifacts for config are already present. Returns true if this is the case"""
        return (self.artifact_path / "results.csv").exists()

    def post_hook(self, entity, only_matching_results=False):
        """collect relevant data from individual entities, may run in various processes"""
        self.logger.info("finalizing %s", entity.name)

        if isinstance(entity, Client):
            if not self.cfg.benchmark.omit_saving_cluster_level_matches and only_matching_results:
                self.queue.put_nowait(("post_hook", "matching_results", entity.name, entity.matching_results))
                return
            if not self.cfg.benchmark.omit_saving_cell_level_matches:
                self.queue.put_nowait(
                    ("post_hook", "original_indices", entity.name, entity.adata_query.obs['encoded_label'].to_numpy()))
                self.queue.put_nowait(("post_hook", "original_cell_labels", entity.name,
                                       entity.adata_query.obs[entity.cluster_type_key].to_numpy()))
            self.queue.put_nowait(("post_hook", "cell_mapper", entity.name, entity.cell_mapper))
        elif isinstance(entity, AggregationServer):
            if not self.cfg.benchmark.omit_saving_cell_level_matches:
                cell_matches = copy.deepcopy(entity.cell_level_matches)
                self.queue.put_nowait(("post_hook", "cell_level_matches", entity.name, cell_matches))
            if not self.cfg.benchmark.omit_saving_similarity_scores:
                self.queue.put_nowait(("post_hook", "similarity_values", entity.name, entity.similarity_values))

    def post_hook_cell_matches_all_thresholds(self, entity, threshold):
        if not self.cfg.benchmark.omit_saving_cluster_level_matches:
            cell_matches = copy.deepcopy(entity.cell_level_matches)
            self.queue.put_nowait(("post_hook", f"cell_level_matches_{threshold}", entity.name, cell_matches))

    def run(self):
        # reset signal handler to default (necessary for slurm)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        try:
            while True:
                values = self.queue.get()
                if values is None:
                    break
                if len(values) == 3:
                    entity, key, value = values
                    self.logger.debug("%s: %s=%.03f", entity, key, value)
                    self.benchmark_data.setdefault(entity, {}).setdefault(key, []).append(value)
                elif len(values) == 4:
                    _, kind, entity_name, value = values
                    self.logger.debug("%s: %s %s", entity_name, kind, value)
                    self.mapping_data.setdefault(entity_name, {})[kind] = value
        except queue.Empty:
            pass
        self.logger.info("writing benchmarks")
        self.write_benchmarks()
        self.write_mappings()
        self.write_cell_level_matches_for_all_thresholds()
        self.copy_selected_features()
        self.logger.info("exiting")

    def start_consumer(self):
        self.consumer = mp.Process(target=self.run, daemon=True)
        self.consumer.start()

    def finalize(self):
        self.queue.put_nowait(None)
        self.queue.close()
        self.consumer.join(1000)

    def get_cfg_params(self):
        return {
            "run": self.cfg.run,
            "n_exp": self.cfg.he_params.n_exp,
            "qi_sizes": list(self.cfg.he_params.qi_sizes),
            "scale_exp": self.cfg.he_params.scale_exp,
            "client1_name": self.cfg.client1.name,
            "client2_name": self.cfg.client2.name,
            "threshold": self.cfg.scmap.threshold,
            "cluster_level_threshold": self.cfg.scmap.cluster_level_threshold,
            "feature_selection": self.cfg.dataset.feature_selection
        }

    def write_benchmarks(self):
        """
        Write benchmarks to file in artefact path.
        :param kwargs: Additional scenario parameter for pandas, e.g., run: 1, cell_count_c1: 1000, ...
        :return:
        """
        benchmarks = self.get_cfg_params()
        benchmarks.update(self.benchmark_data)

        with gzip.open(self.artifact_path / "raw.json.gz", "wb") as f:
            f.write(json.dumps(benchmarks).encode("utf-8"))
        filename = self.artifact_path / "results.csv"

        rows = []
        for entity, values in benchmarks.items():
            if isinstance(values, dict):
                for k, list_of_values in values.items():
                    rows.append((entity, k, "sum", sum(list_of_values)))
                    rows.append((entity, k, "max", max(list_of_values)))
                    rows.append((entity, k, "last", list_of_values[-1]))
            else:
                rows.append(("meta", entity, "last", values))

        df = pd.DataFrame(rows, columns=["entity", "param", "agg", "value"])

        df.to_csv(filename, index=False)
        self.logger.info("wrote %s", filename)

    def write_mappings(self):

        matching_direction_c1_c2 = '{}_{}'.format(self.cfg.client1.name, self.cfg.client2.name)
        matching_direction_c2_c1 = '{}_{}'.format(self.cfg.client2.name, self.cfg.client1.name)

        client1 = self.mapping_data[self.cfg.client1.name]
        client2 = self.mapping_data[self.cfg.client2.name]
        aggregation_server = self.mapping_data["agg_server"]

        if not self.cfg.benchmark.omit_saving_similarity_scores:
            similarity_values = aggregation_server["similarity_values"]
            data = {
                "c1_c2": similarity_values[matching_direction_c1_c2],
                "c2_c1": similarity_values[matching_direction_c2_c1]
            }

            filename = self.artifact_path / 'similarities.json.gz'
            with gzip.open(filename, "wb") as f:
                f.write(json.dumps(data, cls=NumpyEncoder).encode("utf-8"))

            self.logger.info("wrote %s", filename)

        if not self.cfg.benchmark.omit_saving_cell_level_matches:
            matches_c1_c2 = {
                'original_indices': client1['original_indices'],
                'matched_indices': aggregation_server["cell_level_matches"][matching_direction_c1_c2],
                'original_labels': client1['original_cell_labels'],
                'matched_labels':
                    [client2["cell_mapper"].get_class_by_index(matched_centroid_index)
                     for matched_centroid_index
                     in aggregation_server["cell_level_matches"][matching_direction_c1_c2]],
            }
            matches_c2_c1 = {
                'original_indices': client2['original_indices'],
                'matched_indices': aggregation_server["cell_level_matches"][matching_direction_c2_c1],
                'original_labels': client2['original_cell_labels'],
                'matched_labels':
                    [client1["cell_mapper"].get_class_by_index(matched_centroid_index)
                     for matched_centroid_index
                     in aggregation_server["cell_level_matches"][matching_direction_c2_c1]],
            }
            data = {
                "c1_c2": matches_c1_c2,
                "c2_c1": matches_c2_c1
            }

            filename = self.artifact_path / 'cell_level_matches.json.gz'
            with gzip.open(filename, "wb") as f:
                f.write(json.dumps(data, cls=NumpyEncoder).encode("utf-8"))
            self.logger.info("wrote %s", filename)

        if not self.cfg.benchmark.omit_saving_cluster_level_matches:
            matches_c1_c2 = get_cluster_level_row_dict(client1, client2)
            matches_c2_c1 = get_cluster_level_row_dict(client2, client1)

            data = {
                "c1_c2": matches_c1_c2,
                "c2_c1": matches_c2_c1
            }

            filename = self.artifact_path / 'cluster_level_matches.json.gz'
            with gzip.open(filename, "wb") as f:
                f.write(json.dumps(data, cls=NumpyEncoder).encode("utf-8"))
            self.logger.info("wrote %s", filename)

        # Also write cell and centroid mappers per client
        cell_mapper_c1 = client1['cell_mapper']
        cell_mapper_c2 = client2['cell_mapper']

        data = {
            self.cfg.client1.name: {
                'cell_mapper': {
                    'label_to_index': cell_mapper_c1.class_to_index,
                    'index_to_label': cell_mapper_c1.index_to_class
                }
            },
            self.cfg.client2.name: {
                'cell_mapper': {
                    'label_to_index': cell_mapper_c2.class_to_index,
                    'index_to_label': cell_mapper_c2.index_to_class
                }
            }
        }

        filename = self.artifact_path / 'client_mappers.json.gz'
        with gzip.open(filename, "wb") as f:
            f.write(json.dumps(data).encode("utf-8"))
        self.logger.info("wrote %s", filename)

    def write_cell_level_matches_for_all_thresholds(self):

        if not 'extra_thresholds' in self.cfg.scmap:
            return

        extra_thresholds = self.cfg.scmap.extra_thresholds

        extra_cell_matches_path = self.artifact_path / 'extra_cell_level_matches'
        extra_cell_matches_path.mkdir(exist_ok=True, parents=True)

        for extra_threshold in extra_thresholds:
            matching_direction_c1_c2 = '{}_{}'.format(self.cfg.client1.name, self.cfg.client2.name)
            matching_direction_c2_c1 = '{}_{}'.format(self.cfg.client2.name, self.cfg.client1.name)

            client1 = self.mapping_data[self.cfg.client1.name]
            client2 = self.mapping_data[self.cfg.client2.name]
            aggregation_server = self.mapping_data["agg_server"]

            if not self.cfg.benchmark.omit_saving_cell_level_matches:
                matches_c1_c2 = {
                    'original_indices': client1['original_indices'],
                    'matched_indices': aggregation_server[f"cell_level_matches_{extra_threshold}"][matching_direction_c1_c2],
                    'original_labels': client1['original_cell_labels'],
                    'matched_labels':
                        [client2["cell_mapper"].get_class_by_index(matched_centroid_index)
                         for matched_centroid_index
                         in aggregation_server[f"cell_level_matches_{extra_threshold}"][matching_direction_c1_c2]],
                }
                matches_c2_c1 = {
                    'original_indices': client2['original_indices'],
                    'matched_indices': aggregation_server[f"cell_level_matches_{extra_threshold}"][matching_direction_c2_c1],
                    'original_labels': client2['original_cell_labels'],
                    'matched_labels':
                        [client1["cell_mapper"].get_class_by_index(matched_centroid_index)
                         for matched_centroid_index
                         in aggregation_server[f"cell_level_matches_{extra_threshold}"][matching_direction_c2_c1]],
                }
                data = {
                    "c1_c2": matches_c1_c2,
                    "c2_c1": matches_c2_c1
                }

                filename = extra_cell_matches_path / f'cell_level_matches_{extra_threshold}.json.gz'
                with gzip.open(filename, "wb") as f:
                    f.write(json.dumps(data, cls=NumpyEncoder).encode("utf-8"))
                self.logger.info("wrote %s", filename)

    def copy_selected_features(self):
        if 'selected_features' in self.cfg.dataset:
            try:
                shutil.copytree(self.cfg.dataset.selected_features, self.cfg.artifact_path / 'selected_features', dirs_exist_ok=True)
            except FileNotFoundError:
                self.logger.warning("could not copy selected features")
        if 'blinded_features' in self.cfg.dataset:
            try:
                shutil.copytree(self.cfg.dataset.blinded_features, self.cfg.artifact_path / 'blinded_features', dirs_exist_ok=True)
            except FileNotFoundError:
                self.logger.warning("could not copy blinded features")


def get_cluster_level_row_dict(query_client, ref_client):
    """
    Cells of query_client are matches against centroids of ref_client.
    Mappers used to convert encoded labels back to names.
    :param query_client:
    :param ref_client:
    :return:
    """
    mapper1 = query_client["cell_mapper"].get_class_by_index
    mapper2 = ref_client["cell_mapper"].get_class_by_index
    cluster_level_matches = {
        R_ONE_WAY_MATCH: [query_client["matching_results"][R_ONE_WAY_MATCH]],
        R_1_TO_1_ONE_WAY_MATCH: [query_client["matching_results"][R_1_TO_1_ONE_WAY_MATCH]],
        R_TWO_WAY_MATCH: [query_client["matching_results"][R_TWO_WAY_MATCH]],
        R_1_TO_1_TOW_WAY_MATCH: [query_client["matching_results"][R_1_TO_1_TOW_WAY_MATCH]],
        R_ONE_WAY_MATCH + '_label': [apply_functions_to_columns(
            query_client["matching_results"][R_ONE_WAY_MATCH], mapper1, mapper2
        )],
        R_1_TO_1_ONE_WAY_MATCH + '_label': [apply_functions_to_columns(
            query_client["matching_results"][R_1_TO_1_ONE_WAY_MATCH], mapper1, mapper2
        )],
        R_TWO_WAY_MATCH + '_label': [apply_functions_to_columns(
            query_client["matching_results"][R_TWO_WAY_MATCH], mapper1, mapper2
        )],
        R_1_TO_1_TOW_WAY_MATCH + '_label': [apply_functions_to_columns(
            query_client["matching_results"][R_1_TO_1_TOW_WAY_MATCH], mapper1, mapper2
        )],
    }
    return cluster_level_matches


def apply_functions_to_columns(array: NDArray, f1, f2):
    if array.size == 1:
        return np.array(['No match'])
    if array.shape[1] != 2:
        raise ValueError('Array with size grater than 1 must have 2 columns')
    col1 = np.array([f1(i) for i in array[:, 0]])
    col2 = np.array([f2(i) for i in array[:, 1]])
    return np.column_stack((col1, col2))
