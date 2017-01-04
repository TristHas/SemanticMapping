#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.append("models")
sys.path.append("..")

import numpy as np
import theano
from theano import tensor as T
import lasagne
import hickle as hkl
from util.Helpers import Logger

log = Logger()
log.info("All imports done. logging")


log.info("Building Net")
#from models import resnet_idmapping as ResNet
#output_layer = ResNet(X, n_out=1000)

from resnet50 import build_model
#from googlenet import build_model
#from caffe_reference import build_model
net = build_model()
Y = T.ivector('y')

# For non cafe reference model
output_layer = net["prob"]
X = net["input"].input_var

# For reference model
#output_layer = net["fc8"]
#X = net["data"].input_var

output_train = lasagne.layers.get_output(output_layer)
output_test = lasagne.layers.get_output(output_layer, deterministic=True)

loss = lasagne.objectives.categorical_crossentropy(output_train, Y)
loss = loss.mean()

all_layers = lasagne.layers.get_all_layers(output_layer)
l2_penalty = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2) * 0.0001
loss = loss + l2_penalty

l_r = theano.shared(np.array(0.01, dtype=theano.config.floatX))
params = lasagne.layers.get_all_params(output_layer, trainable=True)
updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=l_r, momentum=0.9)

log.info("Compiling function")
train_fn = theano.function(inputs=[X,Y], outputs=loss, updates=updates)

log.info("Loading hickles")
train_x = hkl.load("/home/tristan/data/Imagenet/datasets/ILSVRC2015/train/input/1")[:,:,:224,:224]
train_y = hkl.load("/home/tristan/data/Imagenet/datasets/ILSVRC2015/train/label/1")

log.info("Training")
train_fn(train_x, train_y)
