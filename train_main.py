'''
    training script
'''

import sys
import time
from attention_model import train_from_scratch as train_att


if __name__ == '__main__':
    # See function train for all possible parameter and there definition.
    state = {
        'lrate': 0.01,
        'dim_word': 256,        # word embbedding dim
        'cond_dim': 512,
        'fc_dim': 500,
        'obj_fc_dim': 500,
        'encoder_dim': 512,
        'ctx_dim': 4096,        # frame feature dim
        'obj_ctx_dim': 1024,    # region feature dim
        'n_words': 6500,        # fixed vocab size
        'encoder': 'gru_bi',

        'n_layers_init': 0,
        'n_words_out': 1,
        'clip_c': 0,
        'decay_c': 1e-4,

        'dispFreq': 10,
        'validFreq': 200,
        'use_dropout': True,
        'from_dir': './snapshots/',
        'patience': 50,
        'max_epochs': 500,
        'batch_size': 64,
        'valid_batch_size': 64,

        # attention opts
        'alpha_entropy_r': 0.,
        # 'alpha_c': 0.70602,
        'alpha_c': 0.20602,
        'selector': True,
        'reload_': False
        }
    if len(sys.argv) > 1:
        k, v = sys.argv[1].split('=')
        state[k] = int(v)
    log_file_name = '%s_%d' % ('ha_main', int(time.time()))
    state['save_file_prefix'] = log_file_name
    train_att(state, has_obj=True)
