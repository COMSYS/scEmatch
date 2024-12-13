import numpy as np
from sklearn.model_selection import train_test_split


def get_stratified_sample(adata, count, cluster_type_key, seed):
    index = np.arange(0, adata.shape[0])
    _, sample_index, _, _ = train_test_split(
        index,
        adata.obs[cluster_type_key],
        test_size=count,
        shuffle=True,
        stratify=adata.obs[cluster_type_key],
        random_state=seed
    )
    adata = adata[sample_index, :].copy()
    return adata
