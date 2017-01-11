#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
from timeit import time
sys.path.append("..")

import numpy as np
import hickle as hkl
from DataProcessing.util.Helpers import Logger, check_file_path
from model_loaders import compile_model_features

log = Logger()
log.info("All imports done. logging")

def extract_features(feature_func, mean, dataset_dir):
    batch_names = os.listdir(os.path.join(dataset_dir, "input"))
    inputs =  [os.path.join(dataset_dir, "input", x) for x in batch_names]
    outputs = [os.path.join(dataset_dir, "features", x) for x in batch_names]
    for input, output in zip(inputs, outputs):
        check_file_path(output)
        log.debug("output = {}\n input= {}".format(output, input))
        x = hkl.load(input) - mean
        feature =  np.squeeze(feature_func(x))
        log.debug("feature shape = {}".format(feature.shape))
        hkl.dump(feature, output, mode = "w")
        log.info("Serialized {}".format(output))

def extract_resnet(dataset_dir):
    from model_loaders import compile_model_features
    mean, feature_func = compile_model_features(model = "resnet50")
    extract_features(feature_func, mean, dataset_dir)


def extract_unit_normed_labels(dataset_dir):
    batch_names = os.listdir(os.path.join(dataset_dir, "label"))
    inputs =  [os.path.join(dataset_dir, "label", x) for x in batch_names]
    outputs = [os.path.join(dataset_dir, "UN_labels", x) for x in batch_names]
    for input, output in zip(inputs, outputs):
        check_file_path(output)
        log.debug("output = {}\n input= {}".format(output, input))
        x = hkl.load(input)
        norms = np.linalg.norm(x,axis=0)
        y = x / norms
        hkl.dump(y, output, mode = "w")
        log.info("Serialized {}".format(output))




### Failed Extractions
def check_ds_hickles(ds = "/home/tristan/data/Imagenet/datasets/Sensembed_A/256_224/center_crop/valid"):
    print "Features"
    start = time.time()
    check_failed_hickles(os.path.join(ds,"features"))
    print "Checked in {}".format(time.time() - start)
    print "Labels"
    start = time.time()
    check_failed_hickles(os.path.join(ds,"label"))
    print "Checked in {}".format(time.time() - start)
    #print "Inputs"
    #start = time.time()
    #check_failed_hickles(os.path.join(ds,"input"))
    #print "Checked in {}".format(time.time() - start)

def check_failed_hickles(folder):
    failed_files = []
    for fname in os.listdir(folder):
        try:
            x=hkl.load(os.path.join(folder, fname))
        except IOError as e:
            failed_files.append(fname)
            print fname
    return failed_files

def fix_failed_features(failed_features = [3557], folder = "/home/tristan/data/Imagenet/datasets/Sensembed_A/256_224/center_crop/train"):
    mean, feature_func = compile_model_features(model = "resnet50")
    for i in failed_features:
        input = os.path.join(folder, "input", str(i))
        output = os.path.join(folder, "features",str(i))
        x = hkl.load(input) - mean
        feature =  np.squeeze(feature_func(x))
        hkl.dump(feature, output, mode = "w")

check_ds_hickles()
fix_failed_features()
