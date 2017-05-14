'''
Basic Model Components
'''
from __future__ import print_function
import theano
import theano.tensor as tensor
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle as pickle
import redis
import numpy
import os, sys
import time
import logging
import warnings
from collections import OrderedDict

import utils
from utils import *
from data_engine import prepare_data
from data_engine import load_data_no_feats as load_data
from metrics import compute_wups


# make prefix-appended name
def _p(pp, name):
    return '%s_%s' % (pp, name)


def validate_options(options):
    if options['dim_word'] > options['uni_dim']:
        warnings.warn('dim_word should only be as large as dim.')
    return options


class BaseModel(object):
    def __init__(self, channel=None):
        # layers: 'name': ('parameter initializer', 'feedforward')
        self.layers = {
            'ff': ('self.param_init_fflayer', 'self.fflayer'),
            'lstm': ('self.param_init_lstm', 'self.lstm_layer'),
            'lstm_cond': ('self.param_init_lstm_cond', 'self.lstm_cond_layer'),
	        'gru': ('self.param_init_gru', 'self.gru_layer'),
            'agru': ('self.param_init_agru', 'self.agru_layer'),
	        'gru_cond': ('self.param_init_gru_cond', 'self.gru_cond_layer'),
            }
        self.channel = channel
        self.ans_words = []
        self.gif_dict = {}

    def get_layer(self, name):
        """
        Part of the reason the init is very slow is because,
        the layer's constructor is called even when it isn't needed
        """
        fns = self.layers[name]
        return (eval(fns[0]), eval(fns[1]))

    def load_params(self, path, params):
        # load params from disk
        pp = numpy.load(path)
        for kk, vv in params.iteritems():
            if kk not in pp:
                raise Warning('%s is not in the archive'%kk)
            params[kk] = pp[kk]

        return params

    def init_tparams(self, params, force_cpu=False):
        # initialize Theano shared variables according to the initial parameters
        tparams = OrderedDict()
        for kk, pp in params.iteritems():
            if force_cpu:
                tparams[kk] = theano.tensor._shared(params[kk], name=kk)
            else:
                tparams[kk] = theano.shared(params[kk], name=kk)
        return tparams

    def param_init_fflayer(self, options, params, prefix='ff', nin=None, nout=None):
        if nin == None:
            nin = options['dim_proj']
        if nout == None:
            nout = options['dim_proj']
        params[_p(prefix,'W')] = norm_weight(nin, nout, scale=0.01)
        params[_p(prefix,'b')] = numpy.zeros((nout,)).astype('float32')
        return params

    def fflayer(self, tparams, state_below, options,
                prefix='rconv', activ='lambda x: tensor.tanh(x)', **kwargs):
        return eval(activ)(tensor.dot(state_below, tparams[_p(prefix,'W')])+tparams[
            _p(prefix,'b')])

    # LSTM layer
    def param_init_lstm(self, options, params, prefix=None, nin=None, dim=None):
        assert prefix is not None
        if nin == None:
            nin = options['dim_proj']
        if dim == None:
            dim = options['dim_proj']
        # Stack the weight matricies for faster dot prods
        W = numpy.concatenate([norm_weight(nin,dim),
                               norm_weight(nin,dim),
                               norm_weight(nin,dim),
                               norm_weight(nin,dim)], axis=1)
        params[_p(prefix,'W')] = W
        U = numpy.concatenate([ortho_weight(dim),
                               ortho_weight(dim),
                               ortho_weight(dim),
                               ortho_weight(dim)], axis=1)
        params[_p(prefix,'U')] = U
        params[_p(prefix,'b')] = numpy.zeros((4 * dim,)).astype('float32')

        return params


    # GRU layer
    def param_init_gru(self, options, params, prefix='gru', nin=None, dim=None):
        assert prefix is not None
        if nin == None:
            nin = options['dim_proj']
        if dim == None:
            dim = options['dim_proj']
        # Stack the weight matricies for faster dot prods
        W = numpy.concatenate([norm_weight(nin,dim),
                               norm_weight(nin,dim)], axis=1)
        params[_p(prefix,'W')] = W
        U = numpy.concatenate([ortho_weight(dim),
                               ortho_weight(dim)], axis=1)
        params[_p(prefix,'U')] = U
        params[_p(prefix,'b')] = numpy.zeros((2 * dim,)).astype('float32')
    
        # Params for hidden state
        params[_p(prefix, 'W_h')] = norm_weight(nin, dim)
        params[_p(prefix, 'U_h')] = ortho_weight(dim)
        params[_p(prefix, 'b_h')] = numpy.zeros(dim).astype('float32')
    
        return params
    
    # This function implements the gru fprop
    def gru_layer(self, tparams, state_below, options, prefix='gru', mask=None,
                   forget=False, use_noise=None, trng=None, **kwargs):
        nsteps = state_below.shape[0]
        dim = tparams[_p(prefix,'U')].shape[0]
    
        if state_below.ndim == 3:
            n_samples = state_below.shape[1]
            init_state = tensor.alloc(0., n_samples, dim)
            init_memory = tensor.alloc(0., n_samples, dim)
        else:
            n_samples = 1
            init_state = tensor.alloc(0., dim)
            init_memory = tensor.alloc(0., dim)
    
        if mask == None:
            mask = tensor.alloc(1., state_below.shape[0], 1)
    
        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n*dim:(n+1)*dim]
            elif _x.ndim == 2:
                return _x[:, n*dim:(n+1)*dim]
            return _x[n*dim:(n+1)*dim]
    
    	Uh = tparams[_p(prefix, 'U_h')]
    	bh = tparams[_p(prefix, 'b_h')]
        def _step(m_, x_, x_h_, h_, U, b, Uh, bh):
            preact = tensor.dot(h_, U)
            preact += x_
            preact += b
    
            z = tensor.nnet.sigmoid(_slice(preact, 0, dim))
            r = tensor.nnet.sigmoid(_slice(preact, 1, dim))
    	    h = tensor.tanh(x_h_ + tensor.dot(r *  h_, Uh) + bh)
    	    h = (1 - z) * h_ + z * h
            if m_.ndim == 0:
                # when using this for minibatchsize=1
                h = m_ * h
            else:
    	        h = h * m_[:, None]
            return h, z, r, preact
    
        state_below_h = tensor.dot(
            state_below, tparams[_p(prefix, 'W_h')]) + tparams[_p(prefix, 'b_h')]
        state_below = tensor.dot(
            state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]
        U = tparams[_p(prefix, 'U')]
        b = tparams[_p(prefix, 'b')]
        rval, updates = theano.scan(
            _step,
            sequences=[mask, state_below, state_below_h],
            non_sequences=[U,b,Uh,bh],
            outputs_info = [init_state, None, None, None],
            name=_p(prefix, '_layers'),
            n_steps=nsteps,
            strict=True,
            profile=False)
        return rval

        # aGRU layer
    def param_init_agru(self, options, params, prefix='agru', nin=None, dim=None):
        assert prefix is not None
        if nin == None:
            nin = options['dim_proj']
        if dim == None:
            dim = options['dim_proj']
        # Stack the weight matricies for faster dot prods
        W = numpy.concatenate([norm_weight(nin,dim),
                               norm_weight(nin,dim)], axis=1)
        params[_p(prefix,'W')] = W
        U = numpy.concatenate([ortho_weight(dim),
                               ortho_weight(dim)], axis=1)
        params[_p(prefix,'U')] = U
        params[_p(prefix,'b')] = numpy.zeros((2 * dim,)).astype('float32')
    
        # Params for hidden state
        params[_p(prefix, 'W_h')] = norm_weight(nin, dim)
        params[_p(prefix, 'U_h')] = ortho_weight(dim)
        params[_p(prefix, 'b_h')] = numpy.zeros(dim).astype('float32')
    
        return params
    # This function implements the agru fprop
    def agru_layer(self, tparams, state_below, alpha, options, prefix='agru', mask=None,
                   forget=False, use_noise=None, trng=None, **kwargs):
        nsteps = state_below.shape[0]
        dim = tparams[_p(prefix,'U')].shape[0]
    
        if state_below.ndim == 3:
            n_samples = state_below.shape[1]
            init_state = tensor.alloc(0., n_samples, dim)
            init_memory = tensor.alloc(0., n_samples, dim)
        else:
            n_samples = 1
            init_state = tensor.alloc(0., dim)
            init_memory = tensor.alloc(0., dim)
    
        if mask == None:
            mask = tensor.alloc(1., state_below.shape[0], 1)
    
        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n*dim:(n+1)*dim]
            elif _x.ndim == 2:
                return _x[:, n*dim:(n+1)*dim]
            return _x[n*dim:(n+1)*dim]
    
        Uh = tparams[_p(prefix, 'U_h')]
        bh = tparams[_p(prefix, 'b_h')]
        def _step(m_, x_, x_h_, a_, h_, U, b, Uh, bh):
            preact = tensor.dot(h_, U)
            preact += x_
            preact += b
    
            r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
            h = tensor.tanh(x_h_ + tensor.dot(r *  h_, Uh) + bh)
            h = (1 - a_[:, None]) * h_ + a_[:, None] * h
            if m_.ndim == 0:
                # when using this for minibatchsize=1
                h = m_ * h
            else:
                h = h * m_[:, None]
            return h, r, preact
        
        state_below = state_below * alpha[:,:,None]
        state_below_h = tensor.dot(
            state_below, tparams[_p(prefix, 'W_h')]) + tparams[_p(prefix, 'b_h')]
        state_below = tensor.dot(
            state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]
        U = tparams[_p(prefix, 'U')]
        b = tparams[_p(prefix, 'b')]
        rval, updates = theano.scan(
            _step,
            sequences=[mask, state_below, state_below_h, alpha],
            non_sequences=[U,b,Uh,bh],
            outputs_info = [init_state, None, None],
            name=_p(prefix, '_layers'),
            n_steps=nsteps,
            strict=True,
            profile=False)
        return rval

    # Conditional GRU layer with Attention
    def param_init_gru_cond(self, options, params,
                             prefix='gru_cond', nin=None, dim=None, dimctx=None):
        if nin == None:
            nin = options['dim']
        if dim == None:
            dim = options['dim']
        if dimctx == None:
            dimctx = options['dim']
        # input to GRU
        W = numpy.concatenate([norm_weight(nin,dim),
                               norm_weight(nin,dim)], axis=1)
        params[_p(prefix,'W')] = W
    
        # GRU to GRU
        U = numpy.concatenate([ortho_weight(dim),
                               ortho_weight(dim)], axis=1)
        params[_p(prefix,'U')] = U
    
        # bias to GRU
        params[_p(prefix,'b')] = numpy.zeros((2 * dim,)).astype('float32')
    
        # extra params
        params[_p(prefix, 'W_h')] = norm_weight(nin, dim)
        params[_p(prefix, 'U_h')] = ortho_weight(dim)
        params[_p(prefix, 'b_h')] = numpy.zeros(dim).astype('float32')
    
        # context to GRU
        Wc = norm_weight(dimctx,dim*2)
        params[_p(prefix,'Wc')] = Wc
    
        # attention: context -> hidden
        Wc_att = norm_weight(dimctx, ortho=False)
        params[_p(prefix,'Wc_att')] = Wc_att
    
        # attention: GRU -> hidden
        Wd_att = norm_weight(dim,dimctx)
        params[_p(prefix,'Wd_att')] = Wd_att
    
        # attention: hidden bias
        b_att = numpy.zeros((dimctx,)).astype('float32')
        params[_p(prefix,'b_att')] = b_att
    
        # attention:
        U_att = norm_weight(dimctx,1)
        params[_p(prefix,'U_att')] = U_att
        c_att = numpy.zeros((1,)).astype('float32')
        params[_p(prefix, 'c_tt')] = c_att
    
        if options['selector']:
            # attention: selector
            W_sel = norm_weight(dim, 1)
            params[_p(prefix, 'W_sel')] = W_sel
            b_sel = numpy.float32(0.)
            params[_p(prefix, 'b_sel')] = b_sel
    
        return params
    
    def gru_cond_layer(self, tparams, state_below, options, prefix='gru_cond',
                       mask=None, context=None, one_step=False,
                       init_memory=None, init_state=None,
                       trng=None, use_noise=None,mode=None,
                       **kwargs):
        # state_below (t, m, dim_word), or (m, dim_word) in sampling
        # mask (t, m)
        # context (m, f, dim_ctx), or (f, dim_word) in sampling
        # init_memory, init_state (m, dim)
        assert context, 'Context must be provided'
    
        if one_step:
            assert init_memory, 'previous memory must be provided'
            assert init_state, 'previous state must be provided'
    
        nsteps = state_below.shape[0]
        if state_below.ndim == 3:
            n_samples = state_below.shape[1]
        else:
            n_samples = 1
    
        # mask
        if mask == None:
            mask = tensor.alloc(1., state_below.shape[0], 1)
    
        dim = tparams[_p(prefix, 'U')].shape[0]
    
        # initial/previous state
        if init_state == None:
            init_state = tensor.alloc(0., n_samples, dim)
        # initial/previous memory
        if init_memory == None:
            init_memory = tensor.alloc(0., n_samples, dim)
    
        # projected context
        pctx_ = tensor.dot(context, tparams[_p(prefix,'Wc_att')]) + tparams[
                _p(prefix, 'b_att')]
        if one_step:
            # tensor.dot will remove broadcasting dim
            pctx_ = T.addbroadcast(pctx_,0)
        # projected x
        state_below_h = tensor.dot(
            state_below, tparams[_p(prefix, 'W_h')]) + tparams[_p(prefix, 'b_h')]
        state_below = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[
            _p(prefix, 'b')]
    
        Wd_att = tparams[_p(prefix,'Wd_att')]
        U_att = tparams[_p(prefix,'U_att')]
        c_att = tparams[_p(prefix, 'c_tt')]
        if options['selector']:
            W_sel = tparams[_p(prefix, 'W_sel')]
            b_sel = tparams[_p(prefix,'b_sel')]
        else:
            W_sel = T.alloc(0., 1)
            b_sel = T.alloc(0., 1)
        U = tparams[_p(prefix, 'U')]
        Wc = tparams[_p(prefix, 'Wc')]
        Wh = tparams[_p(prefix, 'U_h')]
        bh = tparams[_p(prefix, 'b_h')]
        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n*dim:(n+1)*dim]
            return _x[:, n*dim:(n+1)*dim]
    
        def _step(m_, x_, x_h_, # sequences
                  h_, a_, ct_, # outputs_info
                  pctx_, ctx_, Wd_att, U_att, c_att, W_sel, b_sel, U, Wc, Wh, bh,# non_sequences
                  dp_=None, dp_att_=None):
            # attention
            pstate_ = tensor.dot(h_, Wd_att)
            pctx_ = pctx_ + pstate_[:,None,:]
            pctx_list = []
            pctx_list.append(pctx_)
            pctx_ = tanh(pctx_)
            
            alpha = tensor.dot(pctx_, U_att)+c_att
            alpha_pre = alpha
            alpha_shp = alpha.shape
            alpha = tensor.nnet.softmax(alpha.reshape([alpha_shp[0],alpha_shp[1]])) # softmax
            ctx_ = (context * alpha[:,:,None]).sum(1) # (m,ctx_dim)
            if options['selector']:
                sel_ = tensor.nnet.sigmoid(tensor.dot(h_, W_sel) + b_sel)
                sel_ = sel_.reshape([sel_.shape[0]])
                ctx_ = sel_[:,None] * ctx_
            preact = tensor.dot(h_, U)
            preact += x_
            preact += tensor.dot(ctx_, Wc)
    
            z = _slice(preact, 0, dim)
            r = _slice(preact, 1, dim)
            if options['use_dropout']:
                z = z * _slice(dp_, 0, dim)
                r = r * _slice(dp_, 1, dim)
            z = tensor.nnet.sigmoid(z)
            r = tensor.nnet.sigmoid(r)
            h = tensor.tanh(x_h_ + tensor.dot(r * h_, Wh) + bh)
            h = (1 - z) * h_ + z * h
            h = m_[:, None] * h
            rval = [h, alpha, ctx_, pstate_, pctx_, z, r, preact, alpha_pre]+pctx_list
            return rval
        if options['use_dropout']:
            _step0 = lambda m_, x_, x_h_, dp_, h_, \
                a_, ct_, \
                pctx_, context, Wd_att, U_att, c_att, W_sel, b_sel, U, Wc, Wh, bh: _step(
                m_, x_, x_h_, h_,
                a_, ct_, pctx_, context, Wd_att, U_att, c_att, W_sel, b_sel, U, Wc, Wh, bh, dp_)
            dp_shape = state_below.shape
            if one_step:
                dp_mask = tensor.switch(use_noise,
                                        trng.binomial((dp_shape[0], 3*dim),
                                                      p=0.5, n=1, dtype=state_below.dtype),
                                        tensor.alloc(0.5, dp_shape[0], 2 * dim))
            else:
                dp_mask = tensor.switch(use_noise,
                                        trng.binomial((dp_shape[0], dp_shape[1], 2*dim),
                                                      p=0.5, n=1, dtype=state_below.dtype),
                                        tensor.alloc(0.5, dp_shape[0], dp_shape[1], 2*dim))
        else:
            _step0 = lambda m_, x_, x_h_, h_, \
                a_, ct_, pctx_, context, Wd_att, U_att, c_att, W_sel, b_sel, U, Wc, Wh, bh: _step(
                m_, x_, x_h_, h_, a_, ct_, pctx_, context,
                    Wd_att, U_att, c_att, W_sel, b_sel, U, Wc, Wh, bh)
    
        if one_step:
            if options['use_dropout']:
                rval = _step0(
                    mask, state_below, state_below_h, dp_mask, init_state, None, None,
                    pctx_, context, Wd_att, U_att, c_att, W_sel, b_sel, U, Wc, Wh, bh)
            else:
                rval = _step0(mask, state_below, state_below_h, init_state, None, None,
                              pctx_, context, Wd_att, U_att, c_att, W_sel, b_sel, U, Wc, Wh, bh)
        else:
            seqs = [mask, state_below, state_below_h]
            if options['use_dropout']:
                seqs += [dp_mask]
            rval, updates = theano.scan(
                _step0,
                sequences=seqs,
                outputs_info = [init_state,
                                tensor.alloc(0., n_samples, pctx_.shape[1]),
                                tensor.alloc(0., n_samples, context.shape[2]),
                                None, None, None, None, None, None, None],
                                non_sequences=[pctx_, context,
                                               Wd_att, U_att, c_att, W_sel, b_sel, U, Wc, Wh, bh],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps, profile=False, mode=mode, strict=True)
    
        return rval

    # This function implements the lstm fprop
    def lstm_layer(self, tparams, state_below, options, prefix='lstm', mask=None,
                   forget=False, use_noise=None, trng=None, **kwargs):
        nsteps = state_below.shape[0]
        dim = tparams[_p(prefix,'U')].shape[0]

        if state_below.ndim == 3:
            n_samples = state_below.shape[1]
            init_state = tensor.alloc(0., n_samples, dim)
            init_memory = tensor.alloc(0., n_samples, dim)
        else:
            n_samples = 1
            init_state = tensor.alloc(0., dim)
            init_memory = tensor.alloc(0., dim)

        if mask == None:
            mask = tensor.alloc(1., state_below.shape[0], 1)

        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n*dim:(n+1)*dim]
            elif _x.ndim == 2:
                return _x[:, n*dim:(n+1)*dim]
            return _x[n*dim:(n+1)*dim]

        def _step(m_, x_, h_, c_, U, b):
            preact = tensor.dot(h_, U)
            preact += x_
            preact += b

            i = tensor.nnet.sigmoid(_slice(preact, 0, dim))
            f = tensor.nnet.sigmoid(_slice(preact, 1, dim))
            o = tensor.nnet.sigmoid(_slice(preact, 2, dim))
            c = tensor.tanh(_slice(preact, 3, dim))

            if forget:
                f = T.zeros_like(f)
            c = f * c_ + i * c
            h = o * tensor.tanh(c)
            if m_.ndim == 0:
                # when using this for minibatchsize=1
                h = m_ * h + (1. - m_) * h_
                c = m_ * c + (1. - m_) * c_
            else:
                h = m_[:,None] * h + (1. - m_)[:,None] * h_
                c = m_[:,None] * c + (1. - m_)[:,None] * c_
            return h, c, i, f, o, preact

        state_below = tensor.dot(
            state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]
        U = tparams[_p(prefix, 'U')]
        b = tparams[_p(prefix, 'b')]
        rval, updates = theano.scan(
            _step,
            sequences=[mask, state_below],
            non_sequences=[U,b],
            outputs_info = [init_state, init_memory, None, None, None, None],
            name=_p(prefix, '_layers'),
            n_steps=nsteps,
            strict=True,
            profile=False)
        return rval

    # Conditional LSTM layer with Attention
    def param_init_lstm_cond(self, options, params,
                             prefix='lstm_cond', nin=None, dim=None, dimctx=None):
        if nin == None:
            nin = options['dim']
        if dim == None:
            dim = options['dim']
        if dimctx == None:
            dimctx = options['dim']
        # input to LSTM
        W = numpy.concatenate([norm_weight(nin,dim),
                               norm_weight(nin,dim),
                               norm_weight(nin,dim),
                               norm_weight(nin,dim)], axis=1)
        params[_p(prefix,'W')] = W

        # LSTM to LSTM
        U = numpy.concatenate([ortho_weight(dim),
                               ortho_weight(dim),
                               ortho_weight(dim),
                               ortho_weight(dim)], axis=1)
        params[_p(prefix,'U')] = U

        # bias to LSTM
        params[_p(prefix,'b')] = numpy.zeros((4 * dim,)).astype('float32')

        # context to LSTM
        Wc = norm_weight(dimctx,dim*4)
        params[_p(prefix,'Wc')] = Wc

        # attention: context -> hidden
        Wc_att = norm_weight(dimctx, ortho=False)
        params[_p(prefix,'Wc_att')] = Wc_att

        # attention: LSTM -> hidden
        Wd_att = norm_weight(dim,dimctx)
        params[_p(prefix,'Wd_att')] = Wd_att

        # attention: hidden bias
        b_att = numpy.zeros((dimctx,)).astype('float32')
        params[_p(prefix,'b_att')] = b_att

        # attention:
        U_att = norm_weight(dimctx,1)
        params[_p(prefix,'U_att')] = U_att
        c_att = numpy.zeros((1,)).astype('float32')
        params[_p(prefix, 'c_tt')] = c_att

        if options['selector']:
            # attention: selector
            W_sel = norm_weight(dim, 1)
            params[_p(prefix, 'W_sel')] = W_sel
            b_sel = numpy.float32(0.)
            params[_p(prefix, 'b_sel')] = b_sel

        return params

    def lstm_cond_layer(self, tparams, state_below, options, prefix='lstm',
                        mask=None, context=None, one_step=False,
                        init_memory=None, init_state=None,
                        trng=None, use_noise=None,mode=None,
                        **kwargs):
        # state_below (t, m, dim_word), or (m, dim_word) in sampling
        # mask (t, m)
        # context (m, f, dim_ctx), or (f, dim_word) in sampling
        # init_memory, init_state (m, dim)
        assert context, 'Context must be provided'

        if one_step:
            assert init_memory, 'previous memory must be provided'
            assert init_state, 'previous state must be provided'

        nsteps = state_below.shape[0]
        if state_below.ndim == 3:
            n_samples = state_below.shape[1]
        else:
            n_samples = 1

        # mask
        if mask == None:
            mask = tensor.alloc(1., state_below.shape[0], 1)

        dim = tparams[_p(prefix, 'U')].shape[0]

        # initial/previous state
        if init_state == None:
            init_state = tensor.alloc(0., n_samples, dim)
        # initial/previous memory
        if init_memory == None:
            init_memory = tensor.alloc(0., n_samples, dim)

        # projected context
        pctx_ = tensor.dot(context, tparams[_p(prefix,'Wc_att')]) + tparams[
                _p(prefix, 'b_att')]
        if one_step:
            # tensor.dot will remove broadcasting dim
            pctx_ = T.addbroadcast(pctx_,0)
        # projected x
        state_below = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[
            _p(prefix, 'b')]

        Wd_att = tparams[_p(prefix,'Wd_att')]
        U_att = tparams[_p(prefix,'U_att')]
        c_att = tparams[_p(prefix, 'c_tt')]
        if options['selector']:
            W_sel = tparams[_p(prefix, 'W_sel')]
            b_sel = tparams[_p(prefix,'b_sel')]
        else:
            W_sel = T.alloc(0., 1)
            b_sel = T.alloc(0., 1)
        U = tparams[_p(prefix, 'U')]
        Wc = tparams[_p(prefix, 'Wc')]
        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n*dim:(n+1)*dim]
            return _x[:, n*dim:(n+1)*dim]

        def _step(m_, x_, # sequences
                  h_, c_, a_, ct_, # outputs_info
                  pctx_, ctx_, Wd_att, U_att, c_att, W_sel, b_sel, U, Wc, # non_sequences
                  dp_=None, dp_att_=None):
            # attention
            pstate_ = tensor.dot(h_, Wd_att)
            pctx_ = pctx_ + pstate_[:,None,:]
            pctx_list = []
            pctx_list.append(pctx_)
            pctx_ = tanh(pctx_)
            
            alpha = tensor.dot(pctx_, U_att)+c_att
            alpha_pre = alpha
            alpha_shp = alpha.shape
            alpha = tensor.nnet.softmax(alpha.reshape([alpha_shp[0],alpha_shp[1]])) # softmax
            ctx_ = (context * alpha[:,:,None]).sum(1) # (m,ctx_dim)
            if options['selector']:
                sel_ = tensor.nnet.sigmoid(tensor.dot(h_, W_sel) + b_sel)
                sel_ = sel_.reshape([sel_.shape[0]])
                ctx_ = sel_[:,None] * ctx_
            preact = tensor.dot(h_, U)
            preact += x_
            preact += tensor.dot(ctx_, Wc)

            i = _slice(preact, 0, dim)
            f = _slice(preact, 1, dim)
            o = _slice(preact, 2, dim)
            if options['use_dropout']:
                i = i * _slice(dp_, 0, dim)
                f = f * _slice(dp_, 1, dim)
                o = o * _slice(dp_, 2, dim)
            i = tensor.nnet.sigmoid(i)
            f = tensor.nnet.sigmoid(f)
            o = tensor.nnet.sigmoid(o)
            c = tensor.tanh(_slice(preact, 3, dim))

            c = f * c_ + i * c
            c = m_[:,None] * c + (1. - m_)[:,None] * c_

            h = o * tensor.tanh(c)
            h = m_[:,None] * h + (1. - m_)[:,None] * h_
            rval = [h, c, alpha, ctx_, pstate_, pctx_, i, f, o, preact, alpha_pre]+pctx_list
            return rval
        if options['use_dropout']:
            _step0 = lambda m_, x_, dp_, h_, c_, \
                a_, ct_, \
                pctx_, context, Wd_att, U_att, c_att, W_sel, b_sel, U, Wc: _step(
                m_, x_, h_, c_,
                a_, ct_,
                pctx_, context, Wd_att, U_att, c_att, W_sel, b_sel, U, Wc,  dp_)
            dp_shape = state_below.shape
            if one_step:
                dp_mask = tensor.switch(use_noise,
                                        trng.binomial((dp_shape[0], 3*dim),
                                                      p=0.5, n=1, dtype=state_below.dtype),
                                        tensor.alloc(0.5, dp_shape[0], 3 * dim))
            else:
                dp_mask = tensor.switch(use_noise,
                                        trng.binomial((dp_shape[0], dp_shape[1], 3*dim),
                                                      p=0.5, n=1, dtype=state_below.dtype),
                                        tensor.alloc(0.5, dp_shape[0], dp_shape[1], 3*dim))
        else:
            _step0 = lambda m_, x_, h_, c_, \
                a_, ct_, pctx_, context, Wd_att, U_att, c_att, W_sel, b_sel, U, Wc: _step(
                m_, x_, h_, c_, a_, ct_, pctx_, context,
                    Wd_att, U_att, c_att, W_sel, b_sel, U, Wc)

        if one_step:
            if options['use_dropout']:
                rval = _step0(
                    mask, state_below, dp_mask, init_state, init_memory, None, None,
                    pctx_, context, Wd_att, U_att, c_att, W_sel, b_sel, U, Wc)
            else:
                rval = _step0(mask, state_below, init_state, init_memory, None, None,
                              pctx_, context, Wd_att, U_att, c_att, W_sel, b_sel, U, Wc)
        else:
            seqs = [mask, state_below]
            if options['use_dropout']:
                seqs += [dp_mask]
            rval, updates = theano.scan(
                _step0,
                sequences=seqs,
                outputs_info = [init_state,
                                init_memory,
                                tensor.alloc(0., n_samples, pctx_.shape[1]),
                                tensor.alloc(0., n_samples, context.shape[2]),
                                None, None, None, None, None, None, None, None],
                                non_sequences=[pctx_, context,
                                               Wd_att, U_att, c_att, W_sel, b_sel, U, Wc],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps, profile=False, mode=mode, strict=True)

        return rval

    def init_params(self, options):
        # all parameters
        params = OrderedDict()
        # embedding
        params['Wemb'] = norm_weight(options['n_words'], options['dim_word'])

        if options['encoder'] == 'lstm_bi':
            print ('bi-directional lstm encoder on ctx')
            params = self.get_layer('lstm')[0](options, params, prefix='encoder',
                                          nin=options['ctx_dim'], dim=options['encoder_dim'])
            params = self.get_layer('lstm')[0](options, params, prefix='encoder_rev',
                                          nin=options['ctx_dim'], dim=options['encoder_dim'])
            # ctx_dim = options['encoder_dim'] * 2 + options['ctx_dim']
            ctx_dim = options['encoder_dim'] * 2

        elif options['encoder'] == 'lstm_uni':
            print ('uni-directional lstm encoder on ctx')
            params = self.get_layer('lstm')[0](options, params, prefix='encoder',
                                          nin=options['ctx_dim'], dim=options['uni_dim'])
            ctx_dim = options['uni_dim']

        else:
            print ('no lstm on ctx')
            params = self.get_layer('ff')[0](
                options, params, prefix='ff_ctx_mean',
                nin=options['ctx_dim'], nout=options['uni_dim'])
            ctx_dim = options['uni_dim']

        # init_state, init_cell
        for lidx in xrange(options['n_layers_init']):
            params = self.get_layer('ff')[0](
                options, params, prefix='ff_init_%d'%lidx, nin=ctx_dim, nout=ctx_dim)

        # query embedding
        params = self.get_layer('lstm')[0](options, params, prefix='query_encoder',
                                      nin=options['dim_word'], dim=options['uni_dim'])

        # readout
        params = self.get_layer('ff')[0](
            options, params, prefix='ff_logit_lstm',
            nin=options['uni_dim'], nout=options['fc_dim'])
        if options['n_layers_out'] > 1:
            for lidx in xrange(1, options['n_layers_out']):
                params = self.get_layer('ff')[0](
                    options, params, prefix='ff_logit_h%d'%lidx,
                    nin=options['fc_dim'], nout=options['fc_dim'])
        params = self.get_layer('ff')[0](
            options, params, prefix='ff_logit',
            nin=options['fc_dim'], nout=options['ydim'])
        return params

    def prepare_all(self, prepare_data, data, index, split='train'):
        x, mask, y = prepare_data([data[1][t] for t in index],
                                  numpy.array(data[2])[index],
                                  maxlen=None)
        gif_x = [data[0][t] for t in index]
        ctx, mask_ctx = load_gif_feats(self.redis, gif_x,
                key_dict=self.gif_dict[split], ctx_dim=4096, limit=20)
        ans_types = numpy.array([data[3][t] for t in index], dtype='int64')

        return ([x, mask, ctx, mask_ctx], y, ans_types)

    def build_pred_fn(self, tparams, model_options):
        (trng, use_noise, x, mask, ctx, mask_ctx,
         y, cost, f_pred_prob, f_pred, f_debug) = self.build_model(tparams, model_options)
        return (f_pred_prob, f_pred)

    def pred_answer(self, f_pred, prepare_data, data, iterator,
                    split, output_prefix='haha', verbose=False):
        """
        Just compute the error
        f_pred: Theano fct computing the prediction
        prepare_data: usual prepare_data for that dataset.
        """
        s0 = time.time()
        ans_words = self.ans_words
        assert (len(ans_words) > 0)
        gif_x = []
        gt_words = []
        pred_words = []

        for _, valid_index in iterator:
            (x_args, y, ans_types) = \
                self.prepare_all(prepare_data, data, valid_index, split=split)

            preds = f_pred(*x_args)
            targets = numpy.array(data[2])[valid_index]

            gif_x += [data[0][t] for t in valid_index]
            pred_words += [ans_words[p] for p in preds]
            gt_words += [ans_words[p] for p in targets]

        output_dir = './outputs/'
        output_file = os.path.join(output_dir, '%s_%s_pred_answers.pkl' %
                                               (output_prefix, split))
        pickle.dump((gif_x, pred_words, gt_words), open(output_file, 'w+'))

        print ('Time used when generating answers: %s' % (time.time() - s0))

    def pred_error(self, f_pred, prepare_data, data, iterator, split, f_debug, verbose=False):
        """
        Just compute the error
        f_pred: Theano fct computing the prediction
        prepare_data: usual prepare_data for that dataset.
        """
        s0 = time.time()
        ans_words = self.ans_words
        assert (len(ans_words) > 0)
        valid_err = 0
        gt_words = []
        pred_words = []
        gt_words_by_type = [[] for _ in range(4)]
        pred_words_by_type = [[] for _ in range(4)]
        total_by_type = [0 for _ in range(4)]
        corr_by_type = [0 for _ in range(4)]

        for _, valid_index in iterator:
            (x_args, y, ans_types) = \
                self.prepare_all(prepare_data, data, valid_index, split=split)

            preds = f_pred(*x_args)
            targets = numpy.array(data[2])[valid_index]
            cmp_res = (preds == targets)
            valid_err += cmp_res.sum()

            for i in range(4):
                total_by_type[i] += (ans_types == i).sum()
                corr_by_type[i] += cmp_res[ans_types == i].sum()

            for i, (p, t) in enumerate(zip(preds, targets)):
                ans_type = ans_types[i]
                p_w, t_w = ans_words[p], ans_words[t]
                pred_words_by_type[ans_type].append(p_w)
                gt_words_by_type[ans_type].append(t_w)
                gt_words.append(p_w)
                pred_words.append(t_w)

        valid_err = 1. - numpy_floatX(valid_err) / len(data[0])
        valid_err_by_type = 1. - numpy_floatX(corr_by_type) / total_by_type
        wups_0 = compute_wups(gt_words, pred_words, 0)
        wups_9 = compute_wups(gt_words, pred_words, 0.9)
        wups_10 = compute_wups(gt_words, pred_words, -1)
        wups_btyp_0 = [compute_wups(gt_words_by_type[i], pred_words_by_type[i], 0) for i in range(4)]
        wups_btyp_9 = [compute_wups(gt_words_by_type[i], pred_words_by_type[i], 0.9) for i in range(4)]
        wups_btyp_10 = [compute_wups(gt_words_by_type[i], pred_words_by_type[i], -1) for i in range(4)]
        # logging.info('err: %s, err_by_type: (%s), wups(0, 0.9, 1): (%s), wups_type: %s' % (
        #     valid_err, ','.join(map(str, valid_err_by_type)), ','.join(map(str, [wups_0, wups_9, wups_10])),
        #     ','.join(map(str, wups_btyp_0+wups_btyp_9+wups_btyp_10))))
        logging.info('metrics: %s,%s,%s,%s' % (
            valid_err, ','.join(map(str, valid_err_by_type)), ','.join(map(str, [wups_0, wups_9, wups_10])),
            ','.join(map(str, wups_btyp_0+wups_btyp_9+wups_btyp_10))))

        print ('Time used when eval: %s' % (time.time() - s0))

        return valid_err
