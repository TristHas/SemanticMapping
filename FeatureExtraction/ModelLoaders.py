#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os, pickle
sys.path.append("models")
sys.path.append("..")

from util.Helpers import Logger
import numpy as np
import theano
from theano import tensor as T
import lasagne
from util.Helpers import Logger

log = Logger()
log.info("All imports done. logging")



def load_resnet50():
    from resnet50 import build_model
    log.info("Building ResNet")
    net = build_model()
    data=pickle.load(open("/home/tristan/workspace/Resnet/src/data/Recipes_weights/resnet50.pkl","rb"))
    weights = data["values"]
    lasagne.layers.helper.set_all_param_values(net["prob"], weights)
    return net, data["mean_image"].astype("float32")

def load_inception_v3():
    from inception_v3 import build_network
    log.info("Building Inception v3 Net")
    net = build_network()
    data=pickle.load(open("/home/tristan/workspace/Resnet/src/data/Recipes_weights/inception_v3.pkl","rb"))
    weights = data["param values"]
    lasagne.layers.helper.set_all_param_values(net["softmax"], weights)
    return net, np.array([104, 117, 123]).reshape(1,3,1,1).astype("float32")

def load_googlenet():
    ## Probably don't need mean because it seems preprocessing only centers the data
    from googlenet import build_model
    log.info("Building GoogleNet")
    net =  build_model()
    data=pickle.load(open("/home/tristan/workspace/Resnet/src/data/Recipes_weights/blvc_googlenet.pkl","rb"))
    weights = data["param values"]
    lasagne.layers.helper.set_all_param_values(net["prob"], weights)
    return net, np.array([104, 117, 123]).reshape(1,3,1,1).astype("float32")

def compile_model_proba(model = "resnet50"):
    if model == "resnet50":
        net, mean = load_resnet50()
        output_layer = net["prob"]
    elif model == "inception_v3":
        net, mean = load_inception_v3()
        output_layer = net["prob"]
    elif model == "googlenet":
        net, mean = load_googlenet()
        output_layer = net["prob"]
    Y = T.ivector('y')
    X = net["input"].input_var
    output_test = lasagne.layers.get_output(output_layer, deterministic=True)
    log.info("Compiling")
    predict_proba = theano.function(inputs=[X], outputs=output_test)
    return mean, predict_proba


###
### Check layers outputs
###
def compile_model_features(model = "resnet50"):
    if model == "resnet50":
        net, mean = load_resnet50()
        output_layer = net["pool5"]
    elif model == "inception_v3":
        net, mean = load_inception_v3()
        output_layer = net["pool3"]
    elif model == "googlenet":
        net, mean = load_googlenet()
        output_layer = net["pool5/7x7_s1"]
    Y = T.ivector('y')
    X = net["input"].input_var
    output_test = lasagne.layers.get_output(output_layer, deterministic=True)
    log.info("Compiling")
    compute_feature = theano.function(inputs=[X], outputs=output_test)
    return mean, compute_feature

