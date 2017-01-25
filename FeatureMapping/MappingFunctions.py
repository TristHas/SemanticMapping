#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os
from timeit import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import DenseLayer, InputLayer

from Helpers import get_Sensembed_A_labelmap, normalize
from ..DataProcessing.util.Helpers import Logger
log = Logger()




def compile_hinge(labels, l_r=0.01, reg=0.00001, margin = 0.1, n_in = 2048, batch_size = 256):
    """
        DOC
    """

    input            = T.matrix("x")
    target           = T.vector("y", dtype="uint64")
    pred, params     = linear_model(input, n_in = n_in, n_out = labels.shape[1], batch_size = batch_size)

    labels           = np.tile(labels,(batch_size,1,1)).swapaxes(1,2)
    smbds            = theano.shared(labels)

    positive_samples = smbds[0,:,target.astype("int64")].dimshuffle(0,1,"x")
    negative_samples = smbds

    sample_positive_dist  = cosine_sim(pred.dimshuffle(0,1,"x"), positive_samples)
    sample_negative_dist  = cosine_sim(pred.dimshuffle(0,1,"x"), negative_samples)
    sample_negative_dist  = T.set_subtensor(sample_negative_dist[range(256),target.astype("int64")], 0.0)

    sample_loss      = T.sum(T.maximum(0.0, margin - sample_positive_dist + sample_negative_dist), axis=1)
    loss             = sample_loss.mean()

    penalty      = reg * lasagne.regularization.apply_penalty(params, lasagne.regularization.l2)
    total_loss   = l_r * (loss + penalty)
    updates      = lasagne.updates.adadelta(total_loss, params, l_r)
    train_func   = theano.function(inputs = [input, target], outputs = loss, updates=updates)
    valid_func   = theano.function(inputs = [input, target], outputs = [loss, pred])
    return train_func, valid_func

def compile_struct(labels, l_r=0.01, reg=0.00001, n_in = 2048, batch_size = 256, ):
    """
        DOC
    """

    input            = T.matrix("x")
    target           = T.vector("y", dtype="uint64")
    pred, params     = linear_model(input, n_in = n_in, n_out = labels.shape[1], batch_size = batch_size)

    index_map        = get_Sensembed_A_labelmap(True)
    labels           = labels.sort_values(by="LABEL").drop(["LABEL"], axis=1).get_values().astype("float32")
    smbds            = theano.shared(np.tile(labels,(batch_size,1,1)).swapaxes(1,2))

    dists            = np.dot(labels, labels.transpose())
    norms            = np.linalg.norm(labels, axis=1)
    dists           /= np.expand_dims(norms,1)
    dists           /= np.expand_dims(norms,1).transpose()
    dists            = theano.shared(dists)

    pred_distance    = cosine_sim(pred.dimshuffle(0,1,'x'), smbds)
    sample_loss      = T.sqr(pred_distance - dists[target.astype("int64")])
    loss             = sample_loss.mean()

    penalty      = reg * lasagne.regularization.apply_penalty(params, lasagne.regularization.l2)
    total_loss   = l_r * (loss + penalty)
    updates      = lasagne.updates.sgd(total_loss, params, l_r)

    train_func   = theano.function(inputs = [input, target], outputs = loss, updates=updates)
    valid_func   = theano.function(inputs = [input, target], outputs = [loss, pred])

    return train_func, valid_func

# Math Helpers
def norm(u):
    return u.norm(2, axis=1)#.dimshuffle(0,'x')

def euclidean_dist(u,v):
    return (u-v).norm(2, axis=1)

def cosine_sim(u,v):
    """Cosine Similarity measure"""
    return (u*v).sum(axis=1)/(norm(u) * norm(v))

# Model functions accept a symbolic variable input and return a double (prediction, parameters)
def linear_model(input, batch_size=256, n_in = 2048, n_out=400):
    input_l  = InputLayer((batch_size, n_in), input_var = input)
    output_l = DenseLayer(input_l, num_units=n_out, nonlinearity=None)
    return lasagne.layers.get_output(output_l), lasagne.layers.get_all_params(output_l)[:1]

def dense_model(input, batch_size=256, n_in = 2048, n_out=400):
    input_l  = InputLayer((batch_size, n_in), input_var = input)
    output_l = DenseLayer(input_l, num_units=n_out, nonlinearity=None)
    return lasagne.layers.get_output(output_l), lasagne.layers.get_all_params(output_l)











































































def compile_dot(l_r=0.01, reg=0.01):
    """
        DOC
    """
    x = T.matrix("x")
    y = T.matrix("y")
    input_l  = InputLayer((256, 2048), input_var = x)
    output_l = DenseLayer(input_l, num_units=400, nonlinearity=None)

    pred = lasagne.layers.get_output(output_l)
    sample_loss = - (pred * y).sum(axis=1)
    loss = sample_loss.mean()

    params = lasagne.layers.get_all_params(output_l)
    penalty = reg * lasagne.regularization.apply_penalty(params, lasagne.regularization.l2)
    total_loss = l_r * (loss + penalty)
    updates = lasagne.updates.adadelta(total_loss, params, l_r)

    train_func = theano.function(inputs = [x,y], outputs = loss, updates=updates)
    valid_func = theano.function(inputs = [x,y], outputs = [loss, pred])

    return train_func, valid_func

def compile_dot_nobias(l_r=0.01, reg=0.01):
    """
        DOC
    """
    x = T.matrix("x")
    y = T.matrix("y")

    input_l  = InputLayer((256, 2048), input_var = x)
    output_l = DenseLayer(input_l, num_units=400, nonlinearity=None)

    pred = lasagne.layers.get_output(output_l)
    pred_norm = pred.norm(2, axis=1).dimshuffle(0,'x')
    pred = pred / pred_norm
    sample_loss = - (pred * y).sum(axis=1)
    loss = sample_loss.mean()

    params = lasagne.layers.get_all_params(output_l)[:1]
    penalty = reg * lasagne.regularization.apply_penalty(params, lasagne.regularization.l2)
    total_loss = l_r * (loss + penalty)
    updates = lasagne.updates.sgd(total_loss, params, l_r)

    train_func = theano.function(inputs = [x,y], outputs = loss, updates=updates)
    valid_func = theano.function(inputs = [x,y], outputs = [loss, pred])

    return train_func, valid_func

def compile_square_nobias(l_r=0.01, reg=0.01):
    """
        DOC
    """
    x = T.matrix("x")
    y = T.matrix("y")
    input_l  = InputLayer((256, 2048), input_var = x)
    output_l = DenseLayer(input_l, num_units=400, nonlinearity=None)

    pred = lasagne.layers.get_output(output_l)
    loss = (y-pred).norm(2, axis=1).mean()

    params = lasagne.layers.get_all_params(output_l)[:1]
    penalty = reg * lasagne.regularization.apply_penalty(params, lasagne.regularization.l2)
    total_loss = l_r * (loss + penalty)
    updates = lasagne.updates.sgd(total_loss, params, l_r)

    train_func = theano.function(inputs = [x,y], outputs = loss, updates=updates)
    valid_func = theano.function(inputs = [x,y], outputs = [loss, pred])

    return train_func, valid_func

def compile_square(l_r=0.01, reg=0.01):
    """
        DOC
    """
    x = T.matrix("x")
    y = T.matrix("y")
    input_l  = InputLayer((256, 2048), input_var = x)
    output_l = DenseLayer(input_l, num_units=400, nonlinearity=None)

    pred = lasagne.layers.get_output(output_l)
    loss = (y-pred).norm(2, axis=1).mean()

    params = lasagne.layers.get_all_params(output_l)
    penalty = reg * lasagne.regularization.apply_penalty(params, lasagne.regularization.l2)
    total_loss = l_r * (loss + penalty)
    updates = lasagne.updates.sgd(total_loss, params, l_r)

    train_func = theano.function(inputs = [x,y], outputs = loss, updates=updates)
    valid_func = theano.function(inputs = [x,y], outputs = [loss, pred])

    return train_func, valid_func


def compile_square_hinge(l_r=0.01, reg=0.01, margin=0.1, normal = True):
    """
        DOC
    """

    index_map        = get_Sensembed_A_labelmap(normal)
    data             = index_map.sort_values(by="LABEL").drop(["LABEL"], axis=1).get_values().astype("float32")
    data             = np.tile(data,(256,1,1)).swapaxes(1,2)
    smbds            = theano.shared(data)

    x = T.matrix("x")
    y = T.matrix("y")
    input_l  = InputLayer((256, 2048), input_var = x)
    output_l = DenseLayer(input_l, num_units=400, nonlinearity=None)

    pred = lasagne.layers.get_output(output_l)
    pos_term = (y-pred).norm(2, axis=1)
    neg_term = (pred.dimshuffle(0,1,'x') - smbds).norm(2, axis = 1)
    sample_loss = 2 * pos_term + T.min(0, margin - pos_term - neg_term)



    loss = (y-pred).norm(2, axis=1).mean()

    params = lasagne.layers.get_all_params(output_l)
    penalty = reg * lasagne.regularization.apply_penalty(params, lasagne.regularization.l2)
    total_loss = l_r * (loss + penalty)
    updates = lasagne.updates.sgd(total_loss, params, l_r)

    train_func = theano.function(inputs = [x,y], outputs = loss, updates=updates)
    valid_func = theano.function(inputs = [x,y], outputs = [loss, pred])

    return train_func, valid_func






def compile_different_dot(l_r=0.01, reg=0.00):
    """
        DOC
    """
    x = T.matrix("x")
    y = T.matrix("y")
    input_l  = InputLayer((256, 2048), input_var = x)
    output_l = DenseLayer(input_l, num_units=400, nonlinearity=None)
    pred = lasagne.layers.get_output(output_l)
    sample_loss_1 = - (pred * y).sum(axis=1)
    sample_loss_2 = - T.diagonal(T.dot(pred, y.T))
    comparison_func = theano.function(inputs = [x,y], outputs = [sample_loss_1, sample_loss_2])
    return comparison_func




def angle(u,v):
    cos = (u * v).sum(axis=1)/(np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
    return cos


def test_dot_functions_1(n=1000,l_r=theano.shared(np.array(1., dtype="float32"), name="lr"), reg=theano.shared(np.array(0.00001, dtype="float32")), f = None):
    from Helpers import normalize
    if f is None:
        t,v = compile_dot_nobias(l_r,reg)
    else:
        t,v = f
    x,y,z = get_batch()
    x = normalize(x)
    y = normalize(y)
    pred = v(x,y)[1]
    for i in range(n):
        t(x,y)
        pred = v(x,y)[1]
        print angle(pred,y).mean()
    return t,v


def test_dot_functions_2(n=1000,l_r=theano.shared(np.array(1., dtype="float32"), name="lr"), reg=theano.shared(np.array(0.00001, dtype="float32")), f = None):
    from Helpers import normalize
    if f is None:
        t,v = compile_dot_nobias_norm(l_r,reg)
    else:
        t,v = f
    x,y,z = get_batch()
    x = normalize(x)
    y = normalize(y)
    pred = v(x,y)[1]
    for i in range(n):
        t(x,y)
        pred = v(x,y)[1]
        print angle(pred,y).mean()
    return t,v


def test_square_function_1(n=1000,l_r=theano.shared(np.array(1., dtype="float32"), name="lr"), reg=theano.shared(np.array(0.00001, dtype="float32")), f = None):
    from Helpers import normalize
    if f is None:
        t,v = compile_square(l_r,reg)
    else:
        t,v = f
    x,y,z = get_batch()
    x = normalize(x)
    y = normalize(y)
    pred = v(x,y)[1]
    for i in range(n):
        t(x,y)
        pred = v(x,y)[1]
        print angle(pred,y).mean()
    return t,v
















def get_batch():
    from Loading import ThreadedLoader
    dl = ThreadedLoader()
    batch=dl.next_batch()
    dl.stop()
    return batch

def test_time():
    from Loading import ThreadedLoader
    funcs = {"square":compile_square,
             "square_nob":compile_square_nobias,
             "dot":compile_dot,
             "dot_nob":compile_dot_nobias}
    x,_,y = get_batch()
    log.info("Without loading")
    for name, func in funcs.iteritems():
        train, _ = func()
        start = time.time()
        for i in range(4547):
            train(x,y)
        stop = time.time()
        log.info("{}: {}s".format(name, stop-start))

    log.info("With Threaded loading")
    for name, func in funcs.iteritems():
        dl = ThreadedLoader()
        train, _ = func()
        start = time.time()
        for i in dl.epoch():
            train(x,y)
        stop = time.time()
        log.info("{}: {}s".format(name, stop-start))

def test_acc(nepoch):
    from Loading import ThreadedLoader
    funcs = {"square":compile_square,
             "square_nob":compile_square_nobias,
             "dot":compile_dot,
             "dot_nob":compile_dot_nobias}
    for name, func in funcs.iteritems():
        log.info("Training {}".format(name))
        val_dl = ThreadedLoader(ds_path = "/home/tristan/data/Imagenet/datasets/Sensembed_A/256_224/center_crop/val")
        train_dl = ThreadedLoader(ds_path = "/home/tristan/data/Imagenet/datasets/Sensembed_A/256_224/center_crop/train")
        start = time.time()
        for i in range(nepoch):
            train, valid = func()
            for x,y,z in val_dl.epoch():
                valid(x,z)
            for x,y,z in train_dl.epoch():
                train(x,z)
            log.info("Epoch {}: Training={}. Validation={}".format(name, stop-start))
        for x,y,z in val_dl.epoch():
            valid(x,z)
        stop = time.time()


def chk_compile_square(l_r=0.01, reg=0.01):
    """
        DOC
    """
    x = T.matrix("x")
    y = T.matrix("y")
    input_l  = InputLayer((256, 2048), input_var = x)
    output_l = DenseLayer(input_l, num_units=400, nonlinearity=None)

    pred = lasagne.layers.get_output(output_l)
    loss = (y-pred).norm(2, axis=1).mean()

    params = lasagne.layers.get_all_params(output_l)
    penalty = reg * lasagne.regularization.apply_penalty(params, lasagne.regularization.l2)
    total_loss = l_r * (loss + penalty)
    updates = lasagne.updates.sgd(total_loss, params, l_r)

    train_func = theano.function(inputs = [x,y], outputs = [loss, total_loss, pred, params[0], penalty], updates=updates)
    return train_func

def chk_compile_dot(l_r=0.01, reg=0.01):
    """
        DOC
    """
    x = T.matrix("x")
    y = T.matrix("y")
    input_l  = InputLayer((256, 2048), input_var = x)
    output_l = DenseLayer(input_l, num_units=400, nonlinearity=None)

    pred = lasagne.layers.get_output(output_l)
    sample_loss = - (pred * y).sum(axis=1)
    loss = sample_loss.mean()

    params = lasagne.layers.get_all_params(output_l)
    penalty = reg * lasagne.regularization.apply_penalty(params, lasagne.regularization.l2)
    total_loss = l_r * (loss + penalty)
    updates = lasagne.updates.sgd(total_loss, params, l_r)

    train_func = theano.function(inputs = [x,y],
                                 outputs = [total_loss, loss, pred, params[0], penalty],
                                 updates=updates)
    return train_func
