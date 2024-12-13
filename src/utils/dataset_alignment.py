import math
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import yaml
import nsforest as ns

from numpy.typing import NDArray

from src.utils import dropout_feature_selection
from src.utils.helper_functions import get_stratified_sample


def feature_select_and_align_datasets(
        adata1: sc.AnnData, adata2: sc.AnnData,
        adata1_name: str, adata2_name: str,
        cluster_type_key1: str, cluster_type_key2: str,
        output_path: Path, n_features: int = 500,
        feature_selection: str = 'dropout',
        cell_count_feature_selection_ns_forest: int = 10000,
        random_noise_features_rate: float = 0.0,
        random_state: int = 0,
        normalize: bool = True, target_sum=1e6):
    """
    Creates and aligns two datasets to their selected feature sets according to scmap's alignment process.
    Datasets are written to `output_path`
    :param adata1: AnnData object of first dataset
    :param adata2: AnnData object of second dataset
    :param adata1_name: Name of first dataset for config
    :param adata2_name: Name of second dataset for config
    :param cluster_type_key1: Access key for cell type labels in adata1
    :param cluster_type_key2: Access key for cell type labels in adata2
    :param output_path: Path to save the aligned datasets
    :param n_features: Number of features to select (only for 'dropout')
    :param feature_selection: 'ns_forest' or 'dropout'
    :param cell_count_feature_selection_ns_forest: 'ns_forest' library not optimized for many cells. Reduce for selection
    :param random_noise_features_rate: Blinding feature rate. 1.0 -> double the selected features
    :param random_state: Random state for nsforest feature selection
    :param normalize: if True log normalize the data
    :param target_sum: target sum of normalization
    :return:
    """

    output_path.mkdir(exist_ok=True, parents=True)

    if normalize:
        sc.pp.normalize_total(adata1, target_sum=target_sum)
        sc.pp.log1p(adata1, base=2)
        sc.pp.normalize_total(adata2, target_sum=target_sum)
        sc.pp.log1p(adata2, base=2)

    # Feature selection on datasets
    if feature_selection == 'dropout':
        selector = dropout_feature_selection.DropoutFeatureSelection()
        ref_features_1 = selector.feature_selection(adata1, n_top_genes=n_features)
        ref_features_2 = selector.feature_selection(adata2, n_top_genes=n_features)
        output_path_features_1 = Path(output_path / 'selected_features' / adata1_name)
        output_path_features_2 = Path(output_path / 'selected_features' / adata2_name)
        output_path_features_1.mkdir(exist_ok=True, parents=True)
        output_path_features_2.mkdir(exist_ok=True, parents=True)
        pd.DataFrame(ref_features_1).to_csv(output_path_features_1 / 'dropout.csv')
        pd.DataFrame(ref_features_2).to_csv(output_path_features_2 / 'dropout.csv')

    elif feature_selection == 'ns_forest':

        if cell_count_feature_selection_ns_forest == -1:
            max_cells = adata1.shape[0]
        else:
            max_cells = cell_count_feature_selection_ns_forest

        if max_cells < adata1.shape[0]:
            adata_for_selection_1 = get_stratified_sample(adata1, max_cells, cluster_type_key1, seed=random_state)
            res = ns.NSForest(
                adata_for_selection_1, cluster_header=cluster_type_key1, n_trees=1000, n_genes_eval=6,
                output_folder=str((output_path / 'selected_features' / adata1_name).absolute()) + '/')
            ref_features_1 = np.array(
                list(set([gene for gene_list in res.binary_genes.tolist() for gene in gene_list])))
        else:
            res = ns.NSForest(
                adata1, cluster_header=cluster_type_key1, n_trees=1000, n_genes_eval=6,
                output_folder=str((output_path / 'selected_features' / adata1_name).absolute()) + '/')
            ref_features_1 = np.array(
                list(set([gene for gene_list in res.binary_genes.tolist() for gene in gene_list])))

        if cell_count_feature_selection_ns_forest == -1:
            max_cells = adata2.shape[0]
        else:
            max_cells = cell_count_feature_selection_ns_forest

        if max_cells < adata2.shape[0]:
            adata_for_selection_2 = get_stratified_sample(adata2, max_cells, cluster_type_key2, seed=random_state)
            res = ns.NSForest(
                adata_for_selection_2, cluster_header=cluster_type_key2, n_trees=1000, n_genes_eval=6,
                output_folder=str((output_path / 'selected_features' / adata2_name).absolute()) + '/')
            ref_features_2 = np.array(
                list(set([gene for gene_list in res.binary_genes.tolist() for gene in gene_list])))
        else:
            res = ns.NSForest(
                adata2, cluster_header=cluster_type_key2, n_trees=1000, n_genes_eval=6,
                output_folder=str((output_path / 'selected_features' / adata2_name).absolute()) + '/')
            ref_features_2 = np.array(
                list(set([gene for gene_list in res.binary_genes.tolist() for gene in gene_list])))

    else:
        raise ValueError('feature_selection must be "dropout" or "nsforest"')

    if random_noise_features_rate > 0:
        output_path_features = Path(output_path / 'blinded_features' / adata1_name)
        ref_features_1 = append_random_features(
            selected_features=ref_features_1, all_features=adata1.var.index.to_numpy(),
            random_noise_features_rate=random_noise_features_rate, random_state=random_state)
        output_path_features.mkdir(exist_ok=True, parents=True)
        pd.DataFrame(ref_features_1).to_csv(output_path_features / 'blinded.csv')

        output_path_features = Path(output_path / 'blinded_features' / adata2_name)
        ref_features_2 = append_random_features(
            selected_features=ref_features_2, all_features=adata2.var.index.to_numpy(),
            random_noise_features_rate=random_noise_features_rate, random_state=random_state)
        output_path_features.mkdir(exist_ok=True, parents=True)
        pd.DataFrame(ref_features_2).to_csv(output_path_features / 'blinded.csv')

    adata1_query, adata2_ref = align_datasets_(adata1, adata2, ref_features_2)
    adata2_query, adata1_ref = align_datasets_(adata2, adata1, ref_features_1)

    # Create yaml file with dataset information
    dataset_information = {
        'dataset_1': {
            'name': adata1_name,
            'cluster_type_key': cluster_type_key1,
            'query_version': {
                **get_dimension(adata1_query, cluster_type_key1),
                'filename': 'd1_aligned.h5ad'
            },
            'ref_version': {
                **get_dimension(adata1_ref, cluster_type_key1),
                'filename': 'd1.h5ad',
            }
        },
        'dataset_2': {
            'name': adata2_name,
            'cluster_type_key': cluster_type_key2,
            'query_version': {
                **get_dimension(adata2_query, cluster_type_key2),
                'filename': 'd2_aligned.h5ad',
            },
            'ref_version': {
                **get_dimension(adata2_ref, cluster_type_key2),
                'filename': 'd2.h5ad',
            },
            'filename': 'd2.h5ad'
        },
        'normalized': normalize,
        'log_normalize': target_sum,
        'feature_selection': feature_selection,
        'random_noise_features_rate': random_noise_features_rate,
        'random_state': random_state
    }

    assert np.all(adata1_query.var.index == adata2_ref.var.index)
    assert np.all(adata2_query.var.index == adata1_ref.var.index)

    adata1_query.write_h5ad(filename=output_path / 'd1_aligned.h5ad')
    adata1_ref.write_h5ad(filename=output_path / 'd1.h5ad')
    adata2_query.write_h5ad(filename=output_path / 'd2_aligned.h5ad')
    adata2_ref.write_h5ad(filename=output_path / 'd2.h5ad')
    yaml.dump(dataset_information, open(output_path / 'config.yaml', 'w'))


def append_random_features(
        selected_features: NDArray, all_features: NDArray,
        random_noise_features_rate: float, random_state: int):
    remaining_features = np.setdiff1d(all_features, selected_features)
    additional_features = math.floor(len(selected_features) * random_noise_features_rate)
    random_features = np.array(pd.Series(remaining_features).sample(additional_features, random_state=random_state))
    blinded_features = np.append(selected_features, random_features)
    assert len(np.unique(blinded_features)) == len(blinded_features)
    return blinded_features


def align_datasets_(adata_query: sc.AnnData, adata_ref: sc.AnnData, reference_features):
    """
    Aligns two datasets to their selected feature sets according to scmap's official alignment process.
    :param adata_query: Create query dataset version from adata_query
    :param adata_ref: Create reference dataset version from adata_ref
    :param reference_features: Reference feature set of adata_ref
    :return:
    """
    # identify common features
    common_features = list(set(reference_features).intersection(adata_query.var.index))
    # Select common features on query dataset with selected reference features, i.e., create query dataset version
    adata_query_version = adata_query[:, common_features].copy()
    # Filter zero features from query dataset version
    sc.pp.filter_genes(adata_query_version, min_cells=1)
    # Filter zero query features also from the reference dataset version of adata2
    adata_ref_version = adata_ref[:, adata_query_version.var.index].copy()
    # Filter zero cells
    sc.pp.filter_cells(adata_query_version, min_genes=1)
    sc.pp.filter_cells(adata_ref_version, min_genes=1)
    return adata_query_version, adata_ref_version


def get_dimension(adata: sc.AnnData, cluster_type_key):
    return {
        'cell_count': adata.shape[0],
        'feature_count': adata.shape[1],
        'cluster_count': adata.obs[cluster_type_key].unique().size,
    }
