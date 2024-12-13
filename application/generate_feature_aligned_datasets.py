from src.utils.dataset_alignment import feature_select_and_align_datasets

if __name__ == '__main__':
    import scanpy as sc
    from pathlib import Path

    seed = 0
    random_noise_features_rate = 0.0  # If random features should be added
    feature_selection = 'ns_forest'  # or 'dropout'
    output_path = f'feature_aligned_datasets/example/{feature_selection}'

    litvinukova = sc.read_h5ad('../datasets/litvinukova.h5ad')
    litvinukova_cell_type_key = 'cell_type'  # Key to access the cell-type labels
    filter_list = ['doublets', 'NotAssigned']
    litvinukova = litvinukova[~litvinukova.obs[litvinukova_cell_type_key].isin(filter_list)]

    chaffin = sc.read_h5ad('../datasets/chaffin.h5ad')
    chaffin_cell_type_key = 'cell_type_leiden0.6'  # Key to access the cell-type labels

    # Generate the aligned datasets and store them in 'feature_aligned_datasets/example/ns_forest'
    feature_select_and_align_datasets(
        adata1=litvinukova.copy(), adata2=chaffin.copy(),
        adata1_name='Litvinukova', adata2_name='Chaffin',
        cluster_type_key1=litvinukova_cell_type_key,
        cluster_type_key2=chaffin_cell_type_key,
        output_path=Path(output_path),
        feature_selection=feature_selection,
        random_state=seed,
        random_noise_features_rate=random_noise_features_rate,
        cell_count_feature_selection_ns_forest=10000
    )
