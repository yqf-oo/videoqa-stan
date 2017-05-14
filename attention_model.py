from __future__ import print_function

import cPickle as pickle
import os
import sys
import time
import logging
import warnings
from collections import OrderedDict

import redis
import numpy
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from base_model import BaseModel
import utils
from utils import *
from data_engine import prepare_data
from data_engine import load_data_no_feats as load_data


def _p(pp, name):
    return '%s_%s' % (pp, name)


def validate_options(options):
    if options['dim_word'] > options['cond_dim']:
        warnings.warn('dim_word should only be as large as cond_dim.')
    return options


class RegionAttention(BaseModel):
    def __init__(self, host='localhost'):
        super(RegionAttention, self).__init__()
        self.frame_db = redis.StrictRedis(host=host, port=6379, db=0)
        self.obj_db = redis.StrictRedis(host=host, port=6379, db=2)

    def init_params(self, options):
        # all parameters
        params = OrderedDict()
        # embedding
        if options['use_w2v']:
            params['Wemb'] = options['w2v_embs']
        else:
            params['Wemb'] = norm_weight(options['n_words'], options['dim_word'])

        params = self.get_layer('gru')[0](options, params, prefix='q_enc_fwd',
                                      nin=options['dim_word'], dim=options['dim_word'])
        params = self.get_layer('gru')[0](options, params, prefix='q_enc_rev',
                                      nin=options['dim_word'], dim=options['dim_word'])
        dim_word = 2 * options['dim_word']


        if options['encoder'] == 'lstm_bi':
            print ('bi-directional lstm encoder on ctx')
            params = self.get_layer('lstm')[0](options, params, prefix='encoder',
                                          nin=options['ctx_dim'], dim=options['encoder_dim'])
            params = self.get_layer('lstm')[0](options, params, prefix='encoder_rev',
                                          nin=options['ctx_dim'], dim=options['encoder_dim'])
            ctx_dim = options['encoder_dim'] * 2 + options['ctx_dim']
            # ctx_dim = options['encoder_dim'] * 2

        elif options['encoder'] == 'gru_bi':
            print ('bi-directional gru encoder on ctx')
            params = self.get_layer('gru')[0](options, params, prefix='encoder',
                                              nin=options['ctx_dim'], dim=options['encoder_dim'])
            params = self.get_layer('gru')[0](options, params, prefix='encoder_rev',
                                              nin=options['ctx_dim'], dim=options['encoder_dim'])
            ctx_dim = options['encoder_dim'] * 2 + options['ctx_dim']
            # ctx_dim = options['encoder_dim'] * 2

        else:
            print ('no gru on ctx')
            ctx_dim = options['ctx_dim']

        params = self.get_layer('ff')[0](options, params,
                                         prefix='ff_obj_proj_q',
                                         nin=dim_word,
                                         nout=options['obj_fc_dim'])
        params = self.get_layer('ff')[0](options, params,
                                         prefix='ff_obj_proj_o',
                                         nin=options['obj_ctx_dim'],
                                         nout=options['obj_fc_dim'])
        params = self.param_init_obj_enc_layer(options, params,
                                               nin=options['obj_fc_dim'],
                                               nout=options['obj_fc_dim'])

        # init_state, init_cell
        for lidx in xrange(options['n_layers_init']):
            params = self.get_layer('ff')[0](
                options, params, prefix='ff_init_%d'%lidx, nin=ctx_dim, nout=ctx_dim)
        params = self.get_layer('ff')[0](
            options, params, prefix='ff_state', nin=ctx_dim, nout=options['cond_dim'])
        params = self.get_layer('ff')[0](
            options, params, prefix='ff_memory', nin=ctx_dim, nout=options['cond_dim'])

        params = self.get_layer('gru_cond')[0](options, params,
                                               prefix='gru_temp_att',
                                               nin=dim_word,
                                               dim=options['cond_dim'],
                                               dimctx=ctx_dim)

        params = self.get_layer('agru')[0](options, params,
                                                 prefix='agru_fwd',
                                                 nin=ctx_dim,
                                                 dim=options['cond_dim'],)
        params = self.get_layer('agru')[0](options, params,
                                                 prefix='agru_rev',
                                                 nin=ctx_dim,
                                                 dim=options['cond_dim'])


        
        # readout
        params = self.get_layer('ff')[0](
            options, params, prefix='ff_ctx2emb_gru',
            nin=2*options['cond_dim'], nout=options['cond_dim'])
        params = self.get_layer('ff')[0](
            options, params, prefix='ff_logit_gru',
            nin=options['cond_dim'], nout=options['fc_dim'])
        if options['n_words_out'] > 1:
            for lidx in xrange(1, options['n_words_out']):
                params = self.get_layer('ff')[0](
                    options, params, prefix='ff_logit_h%d'%lidx,
                    nin=options['fc_dim'], nout=options['fc_dim'])
                params = self.get_layer('ff')[0](
                    options, params, prefix='ff_logit_o%d'%lidx,
                    nin=options['fc_dim'], nout=optinos['ydim'])
        params = self.get_layer('ff')[0](
            options, params, prefix='ff_logit',
            nin=options['fc_dim'], nout=options['ydim'])
        return params

    def param_init_obj_enc_layer(self, options, params,
                                 prefix='obj_enc_layer',
                                 nin=None, nout=None):
        params = self.get_layer('ff')[0](options, params, prefix='ff_obj_att_hid',
                                         nin=nin, nout=nout)

        params[_p(prefix, 'Wa')] = norm_weight(nin, nout, scale=0.01)
        params[_p(prefix, 'Wb')] = norm_weight(nin, nout, scale=0.01)
        params[_p(prefix, 'b')] = numpy.zeros((nout,)).astype('float32')

        params = self.get_layer('ff')[0](options, params, prefix='ff_obj_att_p',
                                         nin=nin, nout=1)
        return params

    def obj_encode_layer(self, tparams, options, query, obj_ctx,
                         mask_obj_ctx=None, prefix='obj_enc_layer'):
        # (o, n, f, d)
        obj_ctx_ = obj_ctx.dimshuffle(2, 0, 1, 3)
        # (n, 1, d)
        query_ = query.dimshuffle(0, 'x', 1)
        # (o, n, f, d)
        q_up_ = self.get_layer('ff')[1](tparams, (obj_ctx_ * query_), options,
                                        prefix='ff_obj_att_hid', activ='rectifier')
        hc = tanh(tensor.dot(q_up_, tparams[_p(prefix, 'Wa')]) +
                  tensor.dot(query_, tparams[_p(prefix, 'Wb')]) + tparams[_p(prefix, 'b')])
        # (o, n, f)
        obj_att = self.get_layer('ff')[1](tparams, hc, options, prefix='ff_obj_att_p', activ='linear')
        shp = obj_att.shape
        obj_att = obj_att.reshape([shp[0], shp[1], shp[2]]).dimshuffle(1, 2, 0)
        # (n, f, o)
        obj_att_probs = tensor.nnet.softmax(obj_att.reshape([shp[2] * shp[1], shp[0]]))
        obj_att_probs = obj_att_probs.reshape([shp[1], shp[2], shp[0]])
        # (n, f, d)
        q_up_ = q_up_.dimshuffle(1, 2, 0, 3) * mask_obj_ctx[:, :, None, None]
        q_att_ = (obj_att_probs[:, :, :, None] * q_up_).sum(axis=-2)

        return q_att_

    def build_model(self, tparams, options):
        trng = RandomStreams(1234)
        use_noise = theano.shared(numpy.float32(0.))
        # description string: #words x #samples
        x = tensor.matrix('x', dtype='int64')
        # x.tag.test_value = self.x_tv
        mask = tensor.matrix('mask', dtype='float32')
        # mask.tag.test_value = self.mask_tv
        # context: #samples x #annotations x dim
        ctx = tensor.tensor3('ctx', dtype='float32')
        # ctx.tag.test_value = self.ctx_tv
        mask_ctx = tensor.matrix('mask_ctx', dtype='float32')
        # obj context: #samples x #annotations x #regions x dim
        obj_ctx = tensor.tensor4('obj_ctx', dtype='float32')
        mask_obj_ctx = tensor.matrix('mask_obj_ctx', dtype='float32')
        # (words, n_samples)
        y = tensor.matrix('y', dtype='int64')
        n_timesteps = x.shape[0]
        n_samples = x.shape[1]

        # index into the word embedding matrix, shift it forward in time

        emb = tparams['Wemb'][x.flatten()].reshape(
            [n_timesteps, n_samples, options['dim_word']])
        counts = mask_ctx.sum(-1).dimshuffle(0, 'x')
        emb_fwd = self.get_layer('gru')[1](tparams, emb,
                                           options, mask=mask,
                                           prefix='q_enc_fwd')[0]
        emb_rev = self.get_layer('gru')[1](tparams, emb[::-1],
                                           options, mask=mask[::-1],
                                           prefix='q_enc_rev')[0]
        emb0 = concatenate((emb_fwd, emb_rev[::-1]), axis=2)
        # emb0 = emb
        emb_mean = emb0.sum(0) / (mask.sum(0).dimshuffle(0, 'x'))
        emb_mean = self.get_layer('ff')[1](tparams, emb_mean, options,
                                       prefix='ff_obj_proj_q', activ='rectifier')

        def _inner_loop(init_emb):
            '''
            init_emb: #n_samples x #fc_dim
            '''
            obj_ctx0 = self.get_layer('ff')[1](tparams, obj_ctx, options,
                                               prefix='ff_obj_proj_o', activ='rectifier')
            # obj_ctx_mean = obj_ctx0.sum(2) / (mask_obj_ctx.sum(-1).dimshuffle(0, 1, 'x'))
            # emb_att = obj_ctx_mean * emb_mean.dimshuffle(0, 'x', 1)
            obj_att = self.obj_encode_layer(tparams, options, init_emb,
                                            obj_ctx0, mask_obj_ctx)

            ctx_ = concatenate((ctx, obj_att), axis=2)

            if options['encoder'] == 'lstm_bi':
                # encoder
                ctx_fwd = self.get_layer('lstm')[1](tparams, ctx_.dimshuffle(1, 0, 2),
                                                    options, mask=mask_ctx.dimshuffle(1, 0),
                                                    prefix='encoder')[0]
                ctx_rev = self.get_layer('lstm')[1](tparams, ctx_.dimshuffle(1, 0, 2)[::-1],
                                                    options, mask=mask_ctx.dimshuffle(1, 0)[::-1],
                                                    prefix='encoder_rev')[0]
                ctx0 = concatenate((ctx_fwd, ctx_rev[::-1]), axis=2)
                ctx0 = ctx0.dimshuffle(1, 0, 2)
                ctx0 = concatenate((ctx_, ctx0), axis=2)
                ctx_mean = ctx0.sum(1)/counts

            elif options['encoder'] == 'gru_bi':
                ctx_fwd = self.get_layer('gru')[1](tparams, ctx_.dimshuffle(1, 0, 2),
                                                   options, mask=mask_ctx.dimshuffle(1, 0),
                                                   prefix='encoder')[0]
                ctx_rev = self.get_layer('gru')[1](tparams, ctx_.dimshuffle(1, 0, 2)[::-1],
                                                   options, mask=mask_ctx.dimshuffle(1, 0)[::-1],
                                                   prefix='encoder_rev')[0]
                ctx0 = concatenate((ctx_fwd, ctx_rev[::-1]), axis=2)
                ctx0 = ctx0.dimshuffle(1, 0, 2)
                ctx0 = concatenate((ctx_, ctx0), axis=2)
                ctx_mean = ctx0.sum(1)/counts

            else:
                ctx0 = ctx_
                ctx_mean = ctx0.sum(1)/counts
            # initial state/cell
            for lidx in xrange(options['n_layers_init']):
                ctx_mean = self.get_layer('ff')[1](
                    tparams, ctx_mean, options, prefix='ff_init_%d'%lidx, activ='rectifier')
                if options['use_dropout']:
                    ctx_mean = dropout_layer(ctx_mean, use_noise, trng)

            init_state = self.get_layer('ff')[1](
                tparams, ctx_mean, options, prefix='ff_state', activ='tanh')
            init_memory = self.get_layer('ff')[1](
                tparams, ctx_mean, options, prefix='ff_memory', activ='tanh')

            # temporal attention
            rval = self.get_layer('gru_cond')[1](tparams, emb0, options,
                                                 prefix='gru_temp_att',
                                                 mask=mask, context=ctx0,
                                                 one_step=False,
                                                 init_state=init_state,
                                                 init_memory=init_memory,
                                                 trng=trng,
                                                 use_noise=use_noise)
            emb_h = rval[0]
            alphas = rval[1]
            emb_h = (emb_h * mask[:, :, None]).sum(axis=0)
            emb_h = emb_h / mask.sum(axis=0)[:, None]

            # bidirectional attention gru
            proj_fwd = self.get_layer('agru')[1](tparams, ctx0.dimshuffle(1, 0, 2),
                                                 alphas[-1].dimshuffle(1, 0), options,
                                                 prefix='agru_fwd',
                                                 mask=mask_ctx.dimshuffle(1, 0),
                                                 trng=trng,
                                                 use_noise=use_noise)

            proj_rev = self.get_layer('agru')[1](tparams, ctx0.dimshuffle(1, 0, 2)[::-1],
                                                 alphas[-1].dimshuffle(1, 0)[::-1], options,
                                                 prefix='agru_rev',
                                                 mask=mask_ctx.dimshuffle(1, 0)[::-1],
                                                 trng=trng,
                                                 use_noise=use_noise)

            proj_h = concatenate((proj_fwd[0], proj_rev[0][::-1]), axis=2)
            # proj_h = proj_h.dimshuffle(1, 0, 2).sum(1) / counts
            indices = tensor.cast(mask_ctx.sum(-1), 'int64')
            proj_h = proj_h.dimshuffle(1, 0, 2)[tensor.arange(mask_ctx.shape[0]), indices - 1]
            proj_h = self.get_layer('ff')[1](
                tparams, proj_h, options, prefix='ff_ctx2emb_gru', activ='linear')
            proj_h = emb_h + proj_h
            if options['use_dropout']:
                proj_h = dropout_layer(proj_h, use_noise, trng)

            logit = self.get_layer('ff')[1](
                tparams, proj_h, options, prefix='ff_logit_gru', activ='linear')

            # logit = logit.dimshuffle(1, 0, 2).sum(1) / counts
            # logit = (logit * mask[:, :, None]).sum(axis=0)
            # logit = logit / mask.sum(axis=0)[:, None]
            return logit, alphas

        logit_list = [emb_mean]
        logit, alphas = _inner_loop(emb_mean)
        logit_list.append(logit_list[-1] + logit)
        for _ in xrange(1, options['n_loops']):
            logit, alphas = _inner_loop(logit_list[-1])
            logit_list.append(logit_list[-1] + logit)
            # logit += logit_u

        # MLP
        logit = tanh(logit_list[-1])
        if options['use_dropout']:
            logit = dropout_layer(logit, use_noise, trng)

        probs_list = []
        preds_list = []

        logit_o = self.get_layer('ff')[1](
            tparams, logit, options, prefix='ff_logit', activ='linear')
        probs = tensor.nnet.softmax(logit_o)
        probs_list.append(probs)
        preds_list.append(probs.argmax(axis=1))

        if options['n_words_out'] > 1:
            for lidx in xrange(1, options['n_words_out']):
                logit = self.get_layer('ff')[1](
                    tparams, logit, options, prefix='ff_logit_h%d'%lidx, activ='rectifier')
                # if options['use_dropout']:
                #     logit = dropout_layer(logit, use_noise, trng)
            
                logit_o = self.get_layer('ff')[1](
                    tparams, logit, options, prefix='ff_logit_o%d'%lidx, activ='linear')
                probs = tensor.nnet.softmax(logit_o)
                probs_list.append(probs)
                preds_list.append(probs.argmax(axis=1))

        # (m, ydim), only deal with one word output
        f_pred_prob = theano.function([x, mask, ctx, mask_ctx, obj_ctx, mask_obj_ctx],
                                      probs, name='f_pred_prob')
        f_pred = theano.function([x, mask, ctx, mask_ctx, obj_ctx, mask_obj_ctx],
                                 probs.argmax(axis=1), name='f_pred')
        f_debug = [None]

	      # cost
        y_flat = y.flatten()
        cost = -tensor.log(probs[tensor.arange(n_samples), y_flat] + 1e-8)
        cost = cost.mean()
        # cost = cost.reshape([y.shape[0], y.shape[1]]).sum(0).mean()

        # cost = -tensor.log(probs[tensor.arange(n_samples), y] + 1e-8).mean()

        extra = [probs, alphas]
        return (trng, use_noise, x, mask, ctx, mask_ctx, obj_ctx, mask_obj_ctx, y,
                alphas, cost, extra, f_pred_prob, f_pred, f_debug)


    def load_obj_dict(self, data_dir):
        self.obj_dict = {}
        for k in ['train', 'val', 'test']:
            pkl_file = os.path.join(data_dir, 'gif_frame_obj_%s.pkl' % k)
            self.obj_dict[k] = pickle.load(open(pkl_file))

    def prepare_all(self, prepare_data, data, index, split='train'):
        x, mask, y = prepare_data([data[1][t] for t in index],
                                  numpy.array(data[2])[index],
                                  maxlen=None)
        gif_x = [data[0][t] for t in index]
        ctx, mask_ctx = load_gif_feats(self.frame_db, gif_x,
                key_dict=self.gif_dict[split], ctx_dim=4096, limit=20)
        obj_ctx, mask_ctx_obj = load_obj_feats(self.obj_db, gif_x,
                                                key_dict=self.obj_dict[split])
        ans_types = numpy.array([data[3][t] for t in index], dtype='int64')

        return ([x, mask, ctx, mask_ctx, obj_ctx, mask_ctx_obj], y[None,:], ans_types)

    def train(self,
              random_seed=1234,
              dim_word=256, # word vector dimensionality
              use_w2v=False,
              w2v_embs=None,
              ctx_dim=-1, # context vector dimensionality, auto set
              obj_ctx_dim=-1,
              cond_dim=1024, # the number of LSTM units
              fc_dim=1000,
              obj_fc_dim=512,
              n_words_out=1,
              n_layers_init=1,
              n_loops=1,
              encoder='none',
              encoder_dim=1000,
              patience=30,
              max_epochs=5000,
              dispFreq=100,
              decay_c=0.,
              alpha_c=0.,
              alpha_entropy_r=0.,
              lrate=0.01,
              selector=False,
              n_words=6500,
              maxlen=50, # maximum length of the description
              optimizer=adadelta,
              clip_c=2.,
              batch_size = 64,
              valid_batch_size = 64,
              save_model_dir='./snapshots/',
              save_file_prefix='lstm',
              validFreq=10,
              saveFreq=1500, # save the parameters after every saveFreq updates
              sampleFreq=1000, # generate some samples after every sampleFreq updates
              video_feature='vggnet',
              saveto='model_best_so_far.npz',
              use_dropout=False,
              reload_=False,
              from_dir=None,
              verbose=True,
              debug=True
              ):
        self.rng_numpy, self.rng_theano = utils.get_two_rngs()

        model_options = locals().copy()
        if 'self' in model_options:
            del model_options['self']

        print ('Loading data')

        # answer word dict
        ans_word_dict = pickle.load(open('./data/ansdict.pkl'))
        self.ans_words = dict((i, w) for w, i in ans_word_dict.items())
        # gif id -> list of frames
        gif_dict = pickle.load(open('./data/tgif_key_dict.pkl'))
        self.gif_dict = gif_dict
        # gif id -> list of frames -> list of objects(regions)
        self.load_obj_dict('./data')
        saveto = os.path.join(save_model_dir, save_file_prefix + '_' + saveto)

        train, valid, test = load_data()
        ydim = numpy.max(train[2]) + 1
        model_options['ydim'] = ydim
        model_options['ctx_dim'] += model_options['obj_fc_dim']
        model_options = validate_options(model_options)
        model_options_file = save_file_prefix + '_model_options.pkl'
        with open('%s/%s'%(save_model_dir, model_options_file), 'wb') as f:
            pickle.dump(model_options, f)

        kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
        kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)

        print ('init params')
        t0 = time.time()
        params = self.init_params(model_options)

        # reloading
        if reload_:
            model_saved = from_dir + 'model_best_so_far.npz'
            # model_saved = from_dir + saveto
            assert os.path.isfile(model_saved)
            print ("Reloading model params...")
            params = load_params(model_saved, params)

        tparams = init_tparams(params)

        (trng, use_noise, x, mask, ctx, mask_ctx, obj_ctx, mask_obj_ctx, y,
         alphas, cost, extra, f_pred_prob, f_pred, f_debug) = self.build_model(tparams, model_options)

        if decay_c > 0.:
            decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
            weight_decay = 0.
            for kk, vv in tparams.iteritems():
                weight_decay += (vv ** 2).sum()
            weight_decay *= decay_c
            cost += weight_decay

        if alpha_c > 0.:
            alpha_c = theano.shared(numpy.float32(alpha_c), name='alpha_c')
            alpha_reg = alpha_c * ((1.-alphas.sum(0))**2).sum(0).mean()
            cost += alpha_reg

        if alpha_entropy_r > 0:
            alpha_entropy_r = theano.shared(numpy.float32(alpha_entropy_r),
                                            name='alpha_entropy_r')
            alpha_reg_2 = alpha_entropy_r * (-tensor.sum(alphas *
                        tensor.log(alphas+1e-8),axis=-1)).sum(0).mean()
            cost += alpha_reg_2
        else:
            alpha_reg_2 = tensor.zeros_like(cost)

        print ('building f_alpha')
        f_alpha = theano.function([x, mask, ctx, mask_ctx, obj_ctx, mask_obj_ctx, y],
                              [alphas, alpha_reg_2],
                              name='f_alpha',
                              on_unused_input='ignore')

        f_cost = theano.function([x, mask, ctx, mask_ctx, obj_ctx, mask_obj_ctx, y], cost, name='f_cost')

        print ('compute grad')
        grads = tensor.grad(cost, wrt=itemlist(tparams))
        if clip_c > 0.:
            g2 = 0.
            for g in grads:
                g2 += (g**2).sum()
            new_grads = []
            for g in grads:
                new_grads.append(tensor.switch(g2 > (clip_c**2),
                                               g / tensor.sqrt(g2) * clip_c,
                                               g))
            grads = new_grads

        f_grad = theano.function([x, mask, ctx, mask_ctx, obj_ctx, mask_obj_ctx, y], grads, name='f_grad')

        lr = tensor.scalar(name='lr')
        f_grad_shared, f_update = optimizer(lr, tparams, grads,
                                            [x, mask, ctx, mask_ctx, obj_ctx, mask_obj_ctx, y], cost)

        print('Optimization')

        print("%d train examples" % len(train[0]))
        print("%d valid examples" % len(valid[0]))
        print("%d test examples" % len(test[0]))

        history_errs = []
        alphas_ratio = []
        best_alpha_ratio = None
        best_p = None
        bad_count = 0
        pred_error = self.pred_error

        if validFreq == -1:
            validFreq = len(train[0]) // batch_size
        if saveFreq == -1:
            saveFreq = len(train[0]) // batch_size

        uidx = 0  # the number of update done
        estop = False  # early stop
        start_time = time.time()
        try:
            for eidx in range(max_epochs):
                n_samples = 0

                # Get new shuffled index for the training set.
                kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

                for _, train_index in kf:
                    uidx += 1
                    n_samples += len(train_index)
                    use_noise.set_value(1.)

                    # Select the random examples for this minibatch
                    (x_args, y, _) = \
                        self.prepare_all(prepare_data, train, train_index, split='train')

                    all_args = x_args + [y]
                    # print (f_debug[0](x_args[0], x_args[4], x_args[5]))
                    cost = f_grad_shared(*all_args)
                    f_update(lrate)

                    if numpy.isnan(cost) or numpy.isinf(cost):
                        print('bad cost detected: ', cost)
                        return 1., 1., 1.

                    if numpy.mod(uidx, dispFreq) == 0:
                        alphas, reg = f_alpha(*all_args)
                        print ('Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'Alpha ratio %.3f, reg %.3f'%(
                            alphas.min(-1).mean() / (alphas.max(-1).mean()), reg))

                    if saveto and numpy.mod(uidx, saveFreq) == 0:
                        print('Saving...')

                        alpha_saveto = os.path.join(save_model_dir, save_file_prefix + '_alpha_ratio.txt')
                        numpy.savetxt(alpha_saveto, best_alpha_ratio)

                        if best_p is not None:
                            params = best_p
                        else:
                            params = unzip(tparams)
                        numpy.savez(saveto, history_errs=history_errs, **params)
                        pickle.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)
                        print('Done')

                    if numpy.mod(uidx, validFreq) == 0:
                        use_noise.set_value(0.)

                        alphas, _ = f_alpha(*all_args)
                        ratio = alphas.min(-1).mean() / (alphas.max(-1)).mean()
                        alphas_ratio.append(ratio)

                        train_err = pred_error(f_pred, prepare_data, train, kf,
                                               'train', f_debug)
                        valid_err = pred_error(f_pred, prepare_data, valid,
                                               kf_valid, 'val', f_debug)
                        test_err = pred_error(f_pred, prepare_data, test, kf_test,
                                              'test', f_debug)

                        history_errs.append([valid_err, test_err])

                        if (best_p is None or
                            valid_err <= numpy.array(history_errs)[:,
                                                                   0].min()):

                            best_p = unzip(tparams)
                            best_alpha_ratio = alphas_ratio
                            bad_counter = 0

                        print( ('Train ', train_err, 'Valid ', valid_err,
                               'Test ', test_err) )
                        logging.info('Epoch: %d, Update: %d, Train %s, Valid %s, Test %s' % (eidx, uidx, train_err, valid_err, test_err))

                        if (len(history_errs) > patience and
                            valid_err >= numpy.array(history_errs)[:-patience,
                                                                   0].min()):
                            bad_counter += 1
                            if bad_counter > patience:
                                print('Early Stop!')
                                estop = True
                                break

                print('Seen %d samples' % n_samples)

                if estop:
                    break

        except KeyboardInterrupt:
            print("Training interupted")

        end_time = time.time()
        if best_p is not None:
            zipp(best_p, tparams)
        else:
            best_p = unzip(tparams)

        use_noise.set_value(0.)
        kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
        train_err = pred_error(f_pred, prepare_data, train, kf_train_sorted, 'train', f_debug)
        valid_err = pred_error(f_pred, prepare_data, valid, kf_valid, 'val', f_debug)
        test_err = pred_error(f_pred, prepare_data, test, kf_test, 'test', f_debug)

        print( 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err )
        if saveto:
            numpy.savez(saveto, train_err=train_err,
                        valid_err=valid_err, test_err=test_err,
                        history_errs=history_errs, **best_p)
        print('The code run for %d epochs, with %f sec/epochs' % (
            (eidx + 1), (end_time - start_time) / (1. * (eidx + 1))))
        print( ('Training took %.1fs' %
                (end_time - start_time)), file=sys.stderr)
        return train_err, valid_err, test_err


def train_from_scratch(state, has_obj=False, host='localhost'):
    logging.basicConfig(filename='logs/%s.log' % state['save_file_prefix'],
                        level=logging.INFO)
    if has_obj:
        model = RegionAttention(host=host)
    else:
        model = BaseAttention()
    model.train(**state)

