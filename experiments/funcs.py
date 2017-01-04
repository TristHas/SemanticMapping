#!/usr/bin/env python
# -*- coding: utf-8 -*-

def compile_cosine(l_r=0.01, reg=0.00001):
    """
        DOC
    """
    input        = T.matrix("x")
    target       = T.matrix("y")
    pred, params = linear_model(input)

    sample_loss  = - cosine_sim(pred, target)
    loss         = sample_loss.mean()

    penalty      = reg * lasagne.regularization.apply_penalty(params, lasagne.regularization.l2)
    total_loss   = l_r * (loss + penalty)
    updates      = lasagne.updates.sgd(total_loss, params, l_r)
    train_func = theano.function(inputs = [input, target], outputs = loss, updates=updates)
    valid_func = theano.function(inputs = [input, target], outputs = [loss, pred])

    return train_func, valid_func

def compile_square(l_r=0.01, reg=0.00001):
    """
        DOC
    """
    input        = T.matrix("x")
    target       = T.matrix("y")
    pred, params = linear_model(input)

    sample_loss  = euclidean_dist(pred, target)
    loss         = sample_loss.mean()

    penalty      = reg * lasagne.regularization.apply_penalty(params, lasagne.regularization.l2)
    total_loss   = l_r * (loss + penalty)
    updates      = lasagne.updates.sgd(total_loss, params, l_r)
    train_func = theano.function(inputs = [input, target], outputs = loss, updates=updates)
    valid_func = theano.function(inputs = [input, target], outputs = [loss, pred])

    return train_func, valid_func

def compile_hinge(l_r=0.01, reg=0.00001, margin = 0.1):
    """
        DOC
    """

    input            = T.matrix("x")
    target           = T.vector("y", dtype="uint64")
    pred, params     = linear_model(input)

    # TOBECHANGED
    #pred             = T.matrix("z")

    index_map        = get_Sensembed_A_labelmap()
    data             = index_map.sort_values(by="LABEL").drop(["BN", "POS", "WNID", "gp", "LABEL"], axis=1)
    data             = normalize(data.get_values().astype("float32"))
    data             = np.tile(data,(256,1,1)).swapaxes(1,2)
    smbds            = theano.shared(data)

    positive_samples = smbds[0,:,target.astype("int64")].dimshuffle(0,1,"x")
    negative_samples = smbds

    sample_positive_dist  = cosine_sim(pred.dimshuffle(0,1,"x"), positive_samples)
    sample_negative_dist  = cosine_sim(pred.dimshuffle(0,1,"x"), negative_samples)
    sample_negative_dist  = T.set_subtensor(sample_negative_dist[range(256),target.astype("int64")], 0.0)

    sample_loss      = T.sum(T.maximum(0.0, margin - sample_positive_dist + sample_negative_dist), axis=1)
    loss             = sample_loss.mean()

    penalty      = reg * lasagne.regularization.apply_penalty(params, lasagne.regularization.l2)
    total_loss   = l_r * (loss + penalty)
    updates      = lasagne.updates.sgd(total_loss, params, l_r)
    #train_func   = theano.function(inputs = [target, pred], outputs = [sample_loss,sample_positive_dist, sample_negative_dist, positive_samples, negative_samples])#loss, updates=updates)
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
def linear_model(input):
    input_l  = InputLayer((256, 2048), input_var = input)
    output_l = DenseLayer(input_l, num_units=400, nonlinearity=None)
    return lasagne.layers.get_output(output_l), lasagne.layers.get_all_params(output_l)[:1]

def dense_model(input):
    input_l  = InputLayer((256, 2048), input_var = input)
    output_l = DenseLayer(input_l, num_units=400, nonlinearity=None)
    return lasagne.layers.get_output(output_l), lasagne.layers.get_all_params(output_l)
