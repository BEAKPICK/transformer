# parameters for dataset
src_lang = 'en'
tgt_lang = 'de'

# no force for max_len
src_max_len = -1
tgt_max_len = -1

# parameters for model
num_units = 512
hidden_size = 2048
dropout = 0.1
epsilon = 0.1
num_layers = 6
num_heads = 8
scaled = True

# parameters for testing
beam_size = 4
lp_alpha = 0.6
lp_k = 5
bleu = 4