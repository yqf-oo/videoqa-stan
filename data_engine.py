import os
import cPickle as pkl
import numpy as np


def load_data_no_feats():
    dataset = []
    data_dir = './data'
    for ds in ['train', 'val', 'test']:
        data = pkl.load(open(os.path.join(data_dir, '%s_gif_qa.pkl' % ds)))
        dataset.append(data)
    return dataset


def load_data():
    dataset = []
    gif_feats = []
    data_dir = './data'
    for ds in ['train', 'val', 'test']:
        data = pkl.load(open(os.path.join(data_dir, '%s_gif_qa.pkl' % ds)))
        gif_feat = np.load(os.path.join(data_dir, 'tgif_mp_%s.npy' % ds))
        dataset.append(data)
        gif_feats.append(gif_feat)
    return (dataset, gif_feats)


def prepare_data(seqs, labels, maxlen=None):
    lens = [len(s) for s in seqs]
    if maxlen:
        for x, y in zip(seqs, labels):
            assert len(x) == len(y)
            new_seqs, new_labels = [], []
            if len(x) < maxlen:
                new_seqs.append(x)
                new_labels.append(y)
            seqs, labels = new_seqs, new_labels

    x_maxlen = max(lens)
    n_samples = len(seqs)
    x = np.zeros((x_maxlen, n_samples), dtype='int64')
    mask = np.zeros((x_maxlen, n_samples), dtype=np.float32)

    for i, s in enumerate(seqs):
        x[:lens[i], i] = s
        mask[:lens[i], i] = 1.
    return (x, mask, labels)
