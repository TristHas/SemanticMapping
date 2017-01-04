#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import DenseLayer, InputLayer

from Helpers import get_Sensembed_A_labelmap

def compile_smbd_struct(l_r=0.01, reg=0.01):
    """
        We suppose pred is the predicted semantic vector, not input CNN-extracted vector
    """
    # Load dists and sensembed shared variables
    index_map = get_Sensembed_A_labelmap()
    data = index_map.sort_values(by="LABEL").drop(["BN", "POS", "WNID", "gp", "LABEL"], axis=1)
    data = data.get_values().astype("float32")
    smbds = theano.shared(np.tile(data,(256,1,1)).swapaxes(1,2))
    dists = np.zeros((907,907))
    for i in range(data.shape[0]):
        dists[i] = np.square(data - data[i]).mean(axis=1)
    dists = theano.shared(dists)

    input       = T.matrix(dtype="float32")
    input_l     = InputLayer((256, 2048), input_var = input)
    output_l    = DenseLayer(input_l, num_units=400, nonlinearity=lasagne.nonlinearities.softmax)
    pred        = lasagne.layers.get_output(output_l)
    labels      = T.ivector()

    pred_dists  = T.sqrt(T.square(pred.dimshuffle(0,1,'x') - smbds).sum(axis=1))
    batch_dists = dists[labels]
    loss        = T.mean(T.square(pred_dists - batch_dists))

    all_layers = lasagne.layers.get_all_layers(output_l)
    l2_penalty = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2) * reg
    loss = loss + l2_penalty
    params = lasagne.layers.get_all_params(output_l, trainable=True)
    updates = lasagne.updates.sgd(loss, params, l_r)

    train_func = theano.function(inputs = [input,labels], outputs = loss, updates=updates)
    valid_func = theano.function(inputs = [input,labels], outputs = [loss, pred])
    return train_func, valid_func

def compile_smbd_hinge_dist(l_r=0.01, reg=0.01, margin = 0.1):
    """
        DOC
    """
    index_map        = get_Sensembed_A_labelmap()
    data             = index_map.sort_values(by="LABEL").drop(["BN", "POS", "WNID", "gp", "LABEL"], axis=1)
    data             = data.get_values().astype("float32")
    data             = np.tile(data,(256,1,1)).swapaxes(1,2)
    smbds            = theano.shared(data)

    input       = T.matrix(dtype="float32")
    input_l     = InputLayer((256, 2048), input_var = input)
    output_l    = DenseLayer(input_l, num_units=400, nonlinearity=lasagne.nonlinearities.softmax)
    pred        = lasagne.layers.get_output(output_l)
    labels      = T.ivector()

    pred_dists  = T.sqrt(T.square(pred.dimshuffle(0,1,'x') - smbds).sum(axis=1))
    positive_samples_dist = (pred_dists[range(256),labels]).dimshuffle(0,'x')
    sample_losses         = T.maximum( margin + positive_samples_dist - pred_dists, 0)
    loss                  = sample_losses.mean()

    all_layers = lasagne.layers.get_all_layers(output_l)
    l2_penalty = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2) * reg
    loss = loss + l2_penalty
    params = lasagne.layers.get_all_params(output_l, trainable=True)
    updates = lasagne.updates.sgd(loss, params, l_r)

    train_func = theano.function(inputs = [input,labels], outputs = loss, updates=updates)
    valid_func = theano.function(inputs = [input,labels], outputs = [loss, pred])
    return train_func, valid_func

def compile_smbd_hinge_dot(l_r=0.01, reg=0.01, margin = 0.1, coef = 2):
    """
        DOC
    """
    index_map        = get_Sensembed_A_labelmap()
    data             = index_map.sort_values(by="LABEL").drop(["BN", "POS", "WNID", "gp", "LABEL"], axis=1)
    data             = data.get_values().astype("float32")
    data             = np.tile(data,(256,1,1)).swapaxes(1,2)
    smbds            = theano.shared(data)

    input       = T.matrix(dtype="float32")
    input_l     = InputLayer((256, 2048), input_var = input)
    output_l    = DenseLayer(input_l, num_units=400, nonlinearity=lasagne.nonlinearities.softmax)
    pred        = lasagne.layers.get_output(output_l)
    labels      = T.ivector()

    positive_samples      = smbds[0,:,labels]
    positive_dot_prod     = T.sum(positive_samples * pred, axis = 1)
    negative_dot_prod     = T.sum(smbds * pred.dimshuffle(0,1,'x'), axis=(1,2))
    per_sample_losse      = T.maximum( margin - coef * positive_dot_prod + negative_dot_prod, 0)
    loss                  = per_sample_losse.mean()

    all_layers = lasagne.layers.get_all_layers(output_l)
    l2_penalty = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2) * reg
    loss = loss + l2_penalty
    params = lasagne.layers.get_all_params(output_l, trainable=True)
    updates = lasagne.updates.sgd(loss, params, l_r)

    train_func = theano.function(inputs = [input,labels], outputs = loss, updates=updates)
    valid_func = theano.function(inputs = [input,labels], outputs = [loss, pred])
    return train_func, valid_func

def compile_smbd_hinge_dot(l_r=0.01, reg=0.01, margin = 0.1, coef = 2):
    """
        DOC
    """
    index_map        = get_Sensembed_A_labelmap()
    data             = index_map.sort_values(by="LABEL").drop(["BN", "POS", "WNID", "gp", "LABEL"], axis=1)
    data             = data.get_values().astype("float32")
    data             = np.tile(data,(256,1,1)).swapaxes(1,2)
    smbds            = theano.shared(data)

    input       = T.matrix(dtype="float32")
    input_l     = InputLayer((256, 2048), input_var = input)
    output_l    = DenseLayer(input_l, num_units=400, nonlinearity=lasagne.nonlinearities.softmax)
    pred        = lasagne.layers.get_output(output_l)
    labels      = T.ivector()

    positive_samples      = smbds[0,:,labels]
    positive_dot_prod     = T.sum(positive_samples * pred, axis = 1)
    negative_dot_prod     = T.sum(smbds * pred.dimshuffle(0,1,'x'), axis=(1,2))
    per_sample_losse      = T.maximum( margin - coef * positive_dot_prod + negative_dot_prod, 0)
    loss                  = per_sample_losse.mean()

    all_layers = lasagne.layers.get_all_layers(output_l)
    l2_penalty = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2) * reg
    loss = loss + l2_penalty
    params = lasagne.layers.get_all_params(output_l, trainable=True)
    updates = lasagne.updates.sgd(loss, params, l_r)

    train_func = theano.function(inputs = [input,labels], outputs = loss, updates=updates)
    valid_func = theano.function(inputs = [input,labels], outputs = [loss, pred])
    return train_func, valid_func

def compile_smbd_hinge_dot(l_r=0.01, reg=0.01, margin = 0.1, coef = 2):
    """
        DOC
    """
    index_map        = get_Sensembed_A_labelmap()
    data             = index_map.sort_values(by="LABEL").drop(["BN", "POS", "WNID", "gp", "LABEL"], axis=1)
    data             = data.get_values().astype("float32")
    data             = np.tile(data,(256,1,1)).swapaxes(1,2)
    smbds            = theano.shared(data)

    input       = T.matrix(dtype="float32")
    input_l     = InputLayer((256, 2048), input_var = input)
    output_l    = DenseLayer(input_l, num_units=400, nonlinearity=lasagne.nonlinearities.softmax)
    pred        = lasagne.layers.get_output(output_l)
    labels      = T.ivector()

    positive_samples      = smbds[0,:,labels]
    positive_dot_prod     = T.sum(positive_samples * pred, axis = 1)
    negative_dot_prod     = T.sum(smbds * pred.dimshuffle(0,1,'x'), axis=(1,2))
    per_sample_losse      = T.maximum( margin - coef * positive_dot_prod + negative_dot_prod, 0)
    loss                  = per_sample_losse.mean()

    all_layers = lasagne.layers.get_all_layers(output_l)
    l2_penalty = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2) * reg
    loss = loss + l2_penalty
    params = lasagne.layers.get_all_params(output_l, trainable=True)
    updates = lasagne.updates.sgd(loss, params, l_r)

    train_func = theano.function(inputs = [input,labels], outputs = loss, updates=updates)
    valid_func = theano.function(inputs = [input,labels], outputs = [loss, pred])
    return train_func, valid_func


def compile_smbd_square(l_r=0.01, reg=0.01):
    """
        DOC
    """
    x = T.matrix("x")
    y = T.matrix("y")
    input_l  = InputLayer((256, 2048), input_var = x)
    output_l = DenseLayer(input_l, num_units=400, nonlinearity=None)
    output = lasagne.layers.get_output(output_l)
    loss = lasagne.objectives.squared_error(output, y).mean()
    all_layers = lasagne.layers.get_all_layers(output_l)
    l2_penalty = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2) * reg
    loss = loss + l2_penalty
    params = lasagne.layers.get_all_params(output_l, trainable=True)
    updates = lasagne.updates.sgd(loss, params, l_r)
    train_func = theano.function(inputs = [x,y], outputs = loss, updates=updates)
    valid_func = theano.function(inputs = [x,y], outputs = [loss, output])
    return train_func, valid_func

def compile_classification(l_r=0.01, reg=0.01):
    """
        DOC
    """
    x = T.matrix("x")
    y = T.ivector("y")
    input_l  = InputLayer((256, 2048), input_var = x)
    output_l = DenseLayer(input_l, num_units=907, nonlinearity=lasagne.nonlinearities.softmax)
    output = lasagne.layers.get_output(output_l)
    loss = loss = lasagne.objectives.categorical_crossentropy(output, y).mean()
    all_layers = lasagne.layers.get_all_layers(output_l)
    l2_penalty = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2) * reg
    loss = loss + l2_penalty
    params = lasagne.layers.get_all_params(output_l, trainable=True)
    updates = lasagne.updates.sgd(loss, params, l_r)
    train_func = theano.function(inputs = [x,y], outputs = loss, updates=updates)
    valid_func = theano.function(inputs = [x,y], outputs = [loss, output])
    return train_func, valid_func


#### Tests


# Always do numpy functions, theano functions and assert results are the same
# Do more complicated tests on the numpy results
def test_compile_smbd_hinge_numpy(l_r=0.01, reg=0.01, threshold = 1):
    """
        DOC
    """
    # Numpy versions
    from Helpers import BatchIterator
    margin           = 0.1
    ds_path          = "/home/tristan/data/Imagenet/datasets/Sensembed_A/256_224/center_crop/"
    bi               = BatchIterator(ds_path, input_type="features")
    x,y,z            = bi.next_smbd_train()
    index_map        = get_Sensembed_A_labelmap()
    data             = index_map.sort_values(by="LABEL").drop(["BN", "POS", "WNID", "gp", "LABEL"], axis=1)
    data             = data.get_values().astype("float32")
    data             = np.tile(data,(256,1,1)).swapaxes(1,2)
    smbds            = data # Turn to shared variable
    labels           = z  # Change for symbolic variable
    pred             = y # Turn to actual output prediction through lasagne
    pred_dists       = np.sqrt(np.square(np.expand_dims(pred, 2) - smbds).sum(axis=1))# Change for pred.dimshuffle(0,1,'x')
    positive_samples_dist = np.expand_dims(pred_dists[range(256),labels], 1)
    sample_losses   = np.maximum(margin + positive_samples_dist - pred_dists, 0)
    loss            = sample_losses.mean()

def test_compile_smbd_hinge_theano(l_r=0.01, reg=0.01, threshold = 1):
    from Helpers import BatchIterator
    margin           = 0.1
    ds_path          = "/home/tristan/data/Imagenet/datasets/Sensembed_A/256_224/center_crop/"
    bi               = BatchIterator(ds_path, input_type="features")
    x,y,z            = bi.next_smbd_train()

    index_map        = get_Sensembed_A_labelmap()
    data             = index_map.sort_values(by="LABEL").drop(["BN", "POS", "WNID", "gp", "LABEL"], axis=1)
    data             = data.get_values().astype("float32")
    data             = np.tile(data,(256,1,1)).swapaxes(1,2)
    smbds            = theano.shared(data) # Turn to shared variable
    labels           = theano.shared(z)  # Change for symbolic variable
    pred             = theano.shared(y) # Turn to actual output prediction through lasagne

    pred_dists            = T.sqrt(T.square(pred.dimshuffle(0,1,'x') - smbds).sum(axis=1))# Change for pred.dimshuffle(0,1,'x')
    positive_samples_dist = (pred_dists[range(256),labels]).dimshuffle(0,'x')
    sample_losses         = T.maximum(margin - positive_samples_dist + pred_dists, 0)
    loss                  = sample_losses.mean()
    f = theano.function(inputs = [], outputs = [pred_dists, positive_samples_dist, sample_losses, loss])
    a,b,c,d = f()
    assert all(c[range(256),z]==margin)


def test_compile_smbd_struct_theano(l_r=0.01, reg=0.01):
    """
        We suppose pred is the predicted semantic vector, not input CNN-extracted vector
    """
    from Helpers import BatchIterator
    ds_path          = "/home/tristan/data/Imagenet/datasets/Sensembed_A/256_224/center_crop/"
    bi               = BatchIterator(ds_path, input_type="features")
    x,y,z            = bi.next_smbd_train()

    # Load dists and sensembed shared variables
    index_map = get_Sensembed_A_labelmap()
    data = index_map.sort_values(by="LABEL").drop(["BN", "POS", "WNID", "gp", "LABEL"], axis=1)
    data = data.get_values().astype("float32")
    smbds = theano.shared(np.tile(data,(256,1,1)).swapaxes(1,2))
    dists = np.zeros((907,907))
    for i in range(data.shape[0]):
        dists[i] = np.square(data - data[i]).mean(axis=1)
    dists = theano.shared(dists)

    labels           = theano.shared(z)  # Change for symbolic variable
    pred             = theano.shared(y) # Turn to actual output prediction through lasagne

    pred_dists  = T.sqrt(T.square(pred.dimshuffle(0,1,'x') - smbds).sum(axis=1))
    batch_dists = dists[labels]

    f = theano.function(inputs = [], outputs = [batch_dists])
    # Test shapes
    # Test equalities
    # Then find a way to test thast correct calculations are performed


def test_compile_smbd_hinge_dot_numpy(l_r=0.01, reg=0.01, margin = 0.1, coef = 2):
    """
        DOC
    """
    from Helpers import BatchIterator
    ds_path          = "/home/tristan/data/Imagenet/datasets/Sensembed_A/256_224/center_crop/"
    bi               = BatchIterator(ds_path, input_type="features")
    x,y,z            = bi.next_smbd_train()

    index_map           = get_Sensembed_A_labelmap()
    data                = index_map.sort_values(by="LABEL").drop(["BN", "POS", "WNID", "gp", "LABEL"], axis=1)
    data                = data.get_values().astype("float32")
    data                  = np.tile(data,(256,1,1)).swapaxes(1,2)
    smbds                 = data

    pred                  = y
    labels                = z

    positive_samples      = smbds[0,:,labels]
    positive_dot_prod     = np.sum(positive_samples * pred, axis = 1)
    negative_dot_prod     = np.sum(smbds * np.expand_dims(pred, 2), axis=(1,2))
    per_sample_losse      = np.maximum( margin - coef * positive_dot_prod + negative_dot_prod, 0)
    loss                  = per_sample_losse.mean()



