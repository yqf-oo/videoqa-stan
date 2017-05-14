import time
import cPickle, os, sys
from collections import OrderedDict

import copy
import time
import logging

import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams

# Set the random number generators' seeds for consistency


def get_two_rngs(seed=None):
    if seed is None:
        seed = 1234
    else:
        seed = seed
    rng_numpy = numpy.random.RandomState(seed)
    rng_theano = MRG_RandomStreams(seed)
    return rng_numpy, rng_theano

rng_numpy, rng_theano = get_two_rngs()


def concatenate(tensor_list, axis=0):
    """
    Alternative implementation of `theano.tensor.concatenate`.
    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.
    Backpropagating through `theano.tensor.concatenate` yields slowdowns
    because the inverse operation (splitting) needs to be done on the CPU.
    This implementation does not have that problem.
    :usage:
        >>> x, y = theano.tensor.matrices('x', 'y')
        >>> c = concatenate([x, y], axis=1)
    :parameters:
        - tensor_list : list
            list of Theano tensor expressions that should be concatenated.
        - axis : int
            the tensors will be joined along this axis.
    :returns:
        - out : tensor
            the concatenated tensor expression.
    """
    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = tensor.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = tensor.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]

    return out

def itemlist(tparams):
    """
    get the list of parameters: Note that tparams must be OrderedDict
    """
    return [vv for kk, vv in tparams.iteritems()]


def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=0.5, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 0.5)
    return proj


def get_ctx_mask(ctx, ctx_dim):
    if ctx.ndim == 3:
        rval = (ctx[:, :, :ctx_dim].sum(axis=-1) != 0).astype('int32').astype('float32')
    elif ctx.ndim == 2:
        rval = (ctx[:, :ctx_dim].sum(axis=-1) != 0).astype('int32').astype('float32')
    elif ctx.ndim == 4:
        rval = (ctx.sum(-1).sum(-1) != 0).astype('int32').astype('float32')
    else:
        # import pdb; pdb.set_trace()
        raise NotImplementedError()

    return rval


def pad_frames(frames, limit, jpegs=False):
    last_frame = frames[-1]
    if jpegs:
        frames_padded = frames + [last_frame]*(limit-len(frames))
    else:
        padding = numpy.asarray([last_frame * 0.]*(limit-len(frames)))
        frames_padded = numpy.concatenate([frames, padding], axis=0)
    return frames_padded


def load_gif_feats(red, gif_ids, key_dict, limit=20, ctx_dim=4096):
    # fetch feats from remote redis server
    assert (key_dict is not None)
    feat_list = []
    # s = time.time()
    for i, gid in enumerate(gif_ids):
        keys = key_dict[gid]
        feat_strs = red.mget(keys)
        feats = [numpy.fromstring(f, dtype='float32') for f in feat_strs]
        feat_list.append(numpy.asarray(feats, dtype='float32'))
    # print 'Time used feteching feats: %s' % (time.time() - s)

    # (n_samples, n_frames, ctx_dim)
    feats = [pad_frames(frames, limit) if len(frames) < limit else frames[:limit]
             for frames in feat_list]
    feats = numpy.asarray(feats).astype('float32')
    feat_mask = get_ctx_mask(feats, ctx_dim)
    return feats, feat_mask


def load_obj_feats(red, gif_ids, key_dict,
                   olimit=3, limit=20, obj_ctx_dim=1024):
    """
    fetch feats from remote redis server
    """
    assert (key_dict is not None)
    feat_list = []

    def chunks(l, n):
        for i in xrange(0, len(l), n):
            yield l[i:i+n]
    # s = time.time()
    # import pdb
    # pdb.set_trace()
    for i, gid in enumerate(gif_ids):
        keys = [str(k).zfill(10) for obj_list in key_dict[gid]
                for k in obj_list]
        if len(keys) <= 0:
            # import pdb
            # pdb.set_trace()
            feats = numpy.zeros((limit, olimit, obj_ctx_dim))
        else:
            keys += [keys[-1]] * (olimit - len(keys) % olimit)
            feat_strs = red.mget(keys)
            feat_strs = list(chunks(feat_strs, olimit))
            feats = [[numpy.fromstring(f, dtype='float32') for f in f_list]
                     for f_list in feat_strs]
        feat_list.append(numpy.asarray(feats, dtype='float32'))
        # for obj_list in key_dict[gid]:
        #     if len(obj_list) < olimit:
        #         continue
        #     keys = [str(k).zfill(10) for k in obj_list[:olimit]]
        #     feat_strs = red.mget(keys)
        #     feats = [numpy.fromstring(f, dtype='float32') for f in feat_strs]
        #     frame_objs.append(numpy.asarray(feats, dtype='float32'))
    # print 'Time used feteching feats: %s' % (time.time() - s)

    # (n_samples, n_frames, n_objs, obj_ctx_dim)
    feats = [pad_frames(frames, limit) if len(frames) < limit else frames[:limit]
             for frames in feat_list]
    feats = numpy.asarray(feats).astype('float32')
    feat_mask = get_ctx_mask(feats, obj_ctx_dim)
    return feats, feat_mask


def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]
    return params


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def tanh(x):
    return tensor.tanh(x)


def rectifier(x):
    return tensor.maximum(0., x)


def linear(x):
    return x


def grad_nan_report(grads, tparams):
    numpy.set_printoptions(precision=3)
    D = OrderedDict()
    i = 0
    NaN_keys = []
    magnitude = []
    assert len(grads) == len(tparams)
    for k, v in tparams.iteritems():
        grad = grads[i]
        magnitude.append(numpy.abs(grad).mean())
        if numpy.isnan(grad.sum()):
            NaN_keys.append(k)
        #assert v.get_value().shape == grad.shape
        D[k] = grad
        i += 1
    #norm = [numpy.sqrt(numpy.sum(grad**2)) for grad in grads]
    #print '\tgrad mean(abs(x))', numpy.array(magnitude)
    return D, NaN_keys


def load_pkl(path):
    """
    Load a pickled file.

    :param path: Path to the pickled file.

    :return: The unpickled Python object.
    """
    f = open(path, 'rb')
    try:
        rval = cPickle.load(f)
    finally:
        f.close()
    return rval

def dump_pkl(obj, path):
    """
    Save a Python object into a pickle file.
    """
    f = open(path, 'wb')
    try:
        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
    finally:
        f.close()


def create_dir_if_not_exist(directory):
    if not os.path.exists(directory):
        print 'creating directory %s'%directory
        os.makedirs(directory)
    else:
        print "%s already exists!"%directory


def flatten_list_of_list(l):
    """
    l is a list of list
    """
    return [item for sublist in l for item in sublist]


def load_txt_file(path):
    f = open(path,'r')
    lines = f.readlines()
    f.close()
    return lines


def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.items():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params


def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(config.floatX)


def norm_weight(nin, nout=None, scale=0.01, ortho=True):
    """
    Random weights drawn from a Gaussian
    """
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * numpy.random.randn(nin, nout)
    return W.astype('float32')


def sgd(lr, tparams, args, cost):
    """ Stochastic Gradient Descent

    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.

    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.items()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function(args, cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update


def adagrad(lr, tparams, grads, args, cost, epsilon=1e-6):
    """Adagrad updates
    Parameters
    ----------
    lr : float or symbolic scalar
        The learning rate controlling the size of update steps
    epsilon : float or symbolic scalar
        Small value added for numerical stability
    """

    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.items()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    accu_grads = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
                  for k, p in tparams.items()]
    accu_up = [(accu, accu + g ** 2) for accu, g in zip(accu_grads, grads)]

    f_grad_shared = theano.function(args, cost, updates=gsup + accu_up,
                                    name='adagrad_f_grad_shared')

    updir = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
             for k, p in tparams.items()]
    updir_new = [(ud, lr * g / tensor.sqrt(accu + epsilon))
                 for ud, g, accu in zip(updir, gshared, accu_grads)]
    param_up = [(p, p - udn[1]) for p, udn in zip(tparams.values(), updir_new)]

    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               name='adagrad_f_update')

    return f_grad_shared, f_update


def adadelta(lr, tparams, grads, args, cost):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(args, cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, args, cost):
    """
    A variant of  SGD that scales the step size by running average of the
    recent step norms.

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [Hint2014]_.

    .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
       lecture 6a,
       http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(args, cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * numpy_floatX(0.),
                           name='%s_updir' % k)
             for k, p in tparams.items()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update


def adam(lr, tparams, grads, inp, cost):
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(inp, cost, updates=gsup)

    lr0 = 0.0002
    b1 = 0.1
    b2 = 0.001
    e = 1e-8

    updates = []

    i = theano.shared(numpy.float32(0.))
    i_t = i + 1.
    fix1 = 1. - b1**(i_t)
    fix2 = 1. - b2**(i_t)
    lr_t = lr0 * (tensor.sqrt(fix2) / fix1)

    for p, g in zip(tparams.values(), gshared):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * tensor.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (tensor.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))

    f_update = theano.function([lr], [], updates=updates, on_unused_input='ignore')

    return f_grad_shared, f_update
