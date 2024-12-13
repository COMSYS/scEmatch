# Similarity score names
COSINE = 'cosine'
PEARSONR = 'pearsonr'
SPEARMANR = 'spearmanr'

# Homomorphic encrpytion keys for pyfhel object
HE_CONTEXT = 'context'
HE_PUBLIC_KEY = 'public_key'
HE_RELIN_KEY = 'relin_key'
HE_ROT_KEY = 'rotate_key'

# Strategies
S_SIMPLE = 'simple'  # Plaintext variant
S_HOMOMORPHIC = 'homomorphic'

# Naming convention for files that are unscaled
REFERENCE_DATA_SUFFIX = 'reference_data'
QUERY_DATA_SUFFIX = 'query_data'

# Naming convention of files sent by each client for scaled data.
COSINE_QUERY_DATA_NAME = 'cosine_query_data'
COSINE_REFERENCE_DATA_NAME = 'cosine_reference_data'
PEARSONR_QUERY_DATA_NAME = 'pearsonr_query_data'
PEARSONR_REFERENCE_DATA_NAME = 'pearsonr_reference_data'
SPEARMANR_QUERY_DATA_NAME = 'spearmanr_query_data'
SPEARMANR_REFERENCE_DATA_NAME = 'spearmanr_reference_data'

# Meta information keys
M_CENTROID_LENGTH = 'centroid_length'
M_QUERY_CELL_COUNT = 'query_cell_count'

# Result types (cluster-level matches)
R_ONE_WAY_MATCH = 'one_way_match'
R_1_TO_1_ONE_WAY_MATCH = '1_to_1_one_way_match'
R_TWO_WAY_MATCH = 'two_way_match'
R_1_TO_1_TOW_WAY_MATCH = '1_to_1_two_way_match'

COMPLETE_RUNTIME = "runtime_total"
