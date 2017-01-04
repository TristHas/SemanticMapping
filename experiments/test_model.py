#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("models")
sys.path.append("..")
import pickle

import numpy as np
import theano
from theano import tensor as T
import lasagne
import hickle as hkl
from util.Helpers import Logger


log = Logger()
log.info("All imports done. logging")

def n_max(array, k):
    return np.argpartition(-array, k, axis=1)[:,:k]

def check_inclusion(maxindices, labels):
    val=[]
    for i, label in enumerate(labels):
        val.append(1 if label in maxindices[i,:] else 0)
    return val

def get_batch_top_k(input, label, k=5):
    p = predict_proba(input)
    topk = check_inclusion(n_max(p, k), label)
    return np.mean(topk)

def load_resnet50():
    from resnet50 import build_model
    net = build_model()
    data=pickle.load(open("/home/tristan/workspace/Resnet/src/data/Recipes_weights/resnet50.pkl","rb"))
    weights = data["values"]
    lasagne.layers.helper.set_all_param_values(output_layer, weights)
    return net, data["mean_image"]

def load_googlenet():
    from googlenet import build_model
    net = build_model()
    data=pickle.load(open("/home/tristan/workspace/Resnet/src/data/Recipes_weights/blvc_googlenet.pkl","rb"))
    weights = data["param values"]
    lasagne.layers.helper.set_all_param_values(net["prob"], weights)
    return net, np.array([104, 117, 123]).reshape(1,3,1,1)


log.info("Building Net")
net, mean = load_googlenet()
output_layer = net["prob"]

Y = T.ivector('y')
X = net["input"].input_var

output_test = lasagne.layers.get_output(output_layer, deterministic=True)
#output_class = T.argmax(output_test, axis=1)

log.info("Compiling")
predict_proba = theano.function(inputs=[X], outputs=output_test)
#predict_class = theano.function(inputs=[X], outputs=output_class)

test_x = hkl.load("/home/tristan/data/Imagenet/datasets/ILSVRC2015/valid/input/1")
test_y = hkl.load("/home/tristan/data/Imagenet/datasets/ILSVRC2015/valid/label/1")

#predict_proba(test_x)
