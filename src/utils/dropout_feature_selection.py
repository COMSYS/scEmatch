import numpy as np
from scipy.sparse import csc_matrix
from sklearn.linear_model import LinearRegression


class DropoutFeatureSelection:
    """Feature selection as proposed by scmap"""

    def __init__(self):

        self.feature_scores = None
        self.variable_gene_index = None

    def feature_selection(self, adata, n_top_genes=500):
        X = adata.X

        if not isinstance(X, csc_matrix):
            X = csc_matrix(X)

        cell_count = X.shape[0]
        feature_count = X.shape[1]
        dropout_rates = []
        for i in range(0, feature_count):
            dropout_rate = np.sum(X.getcol(i).toarray().flatten() == 0) / cell_count
            dropout_rates.append(dropout_rate)
        dropout_rates = np.array(dropout_rates)
        dropout_rates *= 100

        filter_genes_mask = (dropout_rates != 0) & (dropout_rates != 100)

        filtered_means = np.array(X[:, filter_genes_mask].mean(axis=0)).flatten()
        filtered_dropout_rates = dropout_rates[filter_genes_mask]
        filtered_gene_ids = np.array(range(0, X.shape[1]))[filter_genes_mask]

        dropout_rates_log = np.log2(filtered_dropout_rates)
        reg = LinearRegression()
        reg.fit(filtered_means.reshape(-1, 1), y=dropout_rates_log)

        predicted_dropouts = reg.predict(filtered_means.reshape(-1, 1))
        residuals = dropout_rates_log - predicted_dropouts

        variable_genes = sorted(
            list(zip(residuals, filtered_gene_ids, filtered_means, dropout_rates_log, predicted_dropouts)),
            reverse=True,
            key=lambda x: x[0])

        residuals, filtered_gene_ids, filtered_means, dropout_rates_log, predicted_dropouts = zip(*variable_genes)

        filtered_gene_ids = np.array(filtered_gene_ids)
        gene_identifiers = np.array(adata.var.index)[filtered_gene_ids]
        self.variable_gene_index = filtered_gene_ids

        scores = np.full(feature_count, np.nan)
        scores[filtered_gene_ids] = residuals
        self.feature_scores = scores

        return gene_identifiers[:n_top_genes]
