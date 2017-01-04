#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
sys.path.append("models")
sys.path.append("..")
import pickle

import matplotlib.pyplot as plt
import numpy as np
import theano
from theano import tensor as T
import lasagne
import hickle as hkl
from util.Helpers import Logger
from util.preprocessing import get_image, resize, center_crop

import skimage

log = Logger()
log.info("All imports done. logging")


def n_max(array, k):
    return np.argpartition(-array, k, axis=1)[:,:k]

def check_inclusion(maxindices, labels):
    val=[]
    for i, label in enumerate(labels):
        val.append(1 if label in maxindices[i,:] else 0)
    return val

def top_k_result(input, k=5):
    p = predict_proba(input)
    return n_max(p, k)

def get_batch_top_k_scores(input, label, k=5):
    p = predict_proba(input)
    topk = check_inclusion(n_max(p, k), label)
    return np.mean(topk)

def load_resnet50():
    from resnet50 import build_model
    net = build_model()
    data=pickle.load(open("/home/tristan/workspace/Resnet/src/data/Recipes_weights/resnet50.pkl","rb"))
    weights = data["values"]
    lasagne.layers.helper.set_all_param_values(net["prob"], weights)
    return net, data["mean_image"].astype("float32")

def load_inception_v3():
    from inception_v3 import build_network
    net = build_network()
    data=pickle.load(open("/home/tristan/workspace/Resnet/src/data/Recipes_weights/inception_v3.pkl","rb"))
    weights = data["param values"]
    lasagne.layers.helper.set_all_param_values(net["softmax"], weights)
    return net, np.array([104, 117, 123]).reshape(1,3,1,1).astype("float32")

def load_googlenet():
    from googlenet import build_model
    net =  build_model()
    data=pickle.load(open("/home/tristan/workspace/Resnet/src/data/Recipes_weights/blvc_googlenet.pkl","rb"))
    weights = data["param values"]
    lasagne.layers.helper.set_all_param_values(net["prob"], weights)
    return net, np.array([104, 117, 123]).reshape(1,3,1,1).astype("float32")


def compile_func(loading = load_resnet50):
    log.info("Building Net")
    net, mean = loading()
    output_layer = net["prob"]
    Y = T.ivector('y')
    X = net["input"].input_var
    output_test = lasagne.layers.get_output(output_layer, deterministic=True)
    log.info("Compiling")
    predict_proba = theano.function(inputs=[X], outputs=output_test)
    return mean, predict_proba

def old():
    synset_dir = "/home/tristan/data/ILSVRC2015/Data/CLS-LOC/train/n01440764"
    photos = os.listdir("/home/tristan/data/ILSVRC2015/Data/CLS-LOC/train/n01440764")
    photo = os.path.join(synset_dir, photos[0])
    img = get_image(photo, RGB_TO_GBR = False, swapaxing = False)
    plt.imshow(img)
    plt.show()
    img = get_image(photo, RGB_TO_GBR = True, swapaxing = True)[np.newaxis]


def batch_of_syns(wnid = "n02012849", id = 429, batch_size = 256, img_size=224, resize_func = resize):
    synset_dir = os.path.join("/home/tristan/data/ILSVRC2015/Data/CLS-LOC/train", wnid)
    photos = os.listdir(synset_dir)
    input_array = np.zeros((batch_size, 3, img_size, img_size), np.uint8)
    for i, photo in enumerate(photos[:batch_size]):
        print i, photo
        input_array[i,:,:,:] = get_image(os.path.join(synset_dir,photo),
                                               img_size = img_size,
                                               RGB_TO_GBR = True, swapaxing = True)
        #input_array[i,:,:,:] = inception_preprocess(os.path.join(synset_dir,photo),img_size)
    return input_array, batch_size * [id]

from scipy.misc import imread
def inception_preprocess(path, imsize):
    # Expected input: RGB uint8 image
    # Input to network should be bc01, 299x299 pixels, scaled to [-1, 1].
    import skimage.transform
    import numpy as np
    im = imread(path)
    if len(im.shape) == 2:
        #img =  imresize(img, (img_size, img_size))
        log.debug("img.shape={}".format(im.shape))
        log.debug("Image has dimension 2")
        im = np.expand_dims(im, axis=2)
        im = np.tile(im, (1,1,3))
        log.debug("Image shape: {}".format(im.shape))
    elif im.shape[2] > 3:
        im = im[:, :, :3]
        log.debug("Image has dimension superior to 3")
    im = skimage.transform.resize(im, (imsize, imsize), preserve_range=True)
    im = (im - 128) / 128.
    im = np.rollaxis(im, 2)[np.newaxis].astype('float32')
    return im


def tensor_to_img(tensor, RGB_TO_GBR = True, swapaxing = True):
    if swapaxing:
        tensor = np.rollaxis(tensor, 0, 3)
        if RGB_TO_GBR:
            tensor = tensor[:, :, ::-1]
    else:
        if RGB_TO_GBR:
            tensor = tensor[::-1, :, :]
    return tensor


#mean, predict_proba = compile_func()
#x = hkl.load("/home/tristan/data/dummy/valid/input/0") - mean
#y = hkl.load("/home/tristan/data/dummy/valid/label/0")
#syns = pickle.load(open("/home/tristan/workspace/Resnet/src/data/Recipes_weights/blvc_googlenet.pkl","rb"))['synset words']


def show(i):
    print syns[y[i]]
    z = x + mean
    assert z.min() >= 0 and z.max() <= 255
    z = z.astype("int")
    plt.imshow(tensor_to_img(z[i]).astype("uint8"))
    plt.show()


def train_func(nepoch=15, lr=None, reg=None):
    index_map = get_label_sembed_map()
    v = Validator()

    reg_coef = theano.shared(np.array(0.001, dtype=theano.config.floatX))
    l_r = theano.shared(np.array(0.0001, dtype=theano.config.floatX))
    nepoch = 15

    train_dir     = "/home/tristan/data/dummysensembed/train/"
    train_batches = os.listdir(os.path.join(train_dir, "features"))
    valid_dir     = "/home/tristan/data/dummysensembed/test/"
    valid_batches = os.listdir(os.path.join(valid_dir, "features"))
    x = T.matrix("x")
    y = T.matrix("y")
    input_l  = InputLayer((256, 2048), input_var = x)
    output_l = DenseLayer(input_l, num_units=400, nonlinearity=None)
    output = lasagne.layers.get_output(output_l)

    # Not sure about the squared error, shouldn't an axis be specified for the squaring?
    loss = lasagne.objectives.squared_error(output, y).mean()
    all_layers = lasagne.layers.get_all_layers(output_l)
    l2_penalty = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2) * reg_coef
    loss = loss + l2_penalty

    params = lasagne.layers.get_all_params(output_l, trainable=True)
    updates = lasagne.updates.sgd(loss, params, l_r)

    train_func = theano.function(inputs = [x,y], outputs = loss, updates=updates)
    valid_func = theano.function(inputs = [x,y], outputs = loss)

    if lr is not None:
        l_r.set_value(np.array(lr, dtype=theano.config.floatX))
    if reg is not None:
        reg_coef.set_value(np.array(reg, dtype=theano.config.floatX))

    val_res = []
    train_res = []
    for epoch in range(nepoch):
        train_score = 0
        valid_score = 0
        start = time.time()
        update_time = 0
        load_time   = 0
        for batch in valid_batches:
            x,y,z = get_batch(batch, valid_dir)
            score = valid_func(x,y)
            valid_score += score
        log.debug("Epoch {} validation executed in {}s: Validation score: {}".format(epoch, time.time() - start, valid_score/len(valid_batches)))
        start = time.time()
        for batch in train_batches:
            start_load = time.time()
            x,y,_ = get_batch(batch, train_dir)
            load_time += time.time() - start_load
            start_update = time.time()
            score = train_func(x,y)
            update_time += time.time() - start_update
            train_score += score
        train_time = time.time() - start
        log.info("Epoch {} training executed in {}s: Training score: {}".format(epoch, train_time, train_score/len(train_batches)))
        log.info("{}s spent in loading ({}%), {}s spent in computing ({}%)".format(load_time,100*load_time/train_time,
                                                                                   update_time, 100*update_time/train_time))
        val_res.append(valid_score/len(valid_batches))
        train_res.append(train_score/len(valid_batches))
        return val_res, train_res

def run_params_tests(nepoch = 15):
    lrs  = [0.0001, 0.001, 0.01, 0.1]
    regs = [0, 0.1, 0.01, 0.001]
    columns = ["lr", "reg", "ds"] + [str(i) for i in range(nepoch)]
    results = pd.DataFrame(columns=columns)
    for lr in lrs:
        for reg in regs:
            log.info("lr:{} reg:{}".format(lr,reg))
            val, train = train_func(nepoch, lr, reg)
            val_data = [lr, reg, "val"] + val
            train_data = [lr, reg, "train"] + train
            results.append(pd.DataFrame(data = val_data, columns=columns))
            results.append(pd.DataFrame(data = train_data, columns=columns))
