#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
from timeit import time
sys.path.append("..")
from Helpers import BatchIterator
from Loading import ThreadedLoader, MultiprocessedLoader
from MappingFunctions import *

from DataProcessing.util.Helpers import Logger

log = Logger()

def test_loadings_square(legacy=True, threaded=True, multiprocessed=True):
    """
    """
    ds_path="/home/tristan/data/Imagenet/datasets/Sensembed_A/256_224/center_crop/"
    log.info("TESTING LOADING WITH SQUARE LOSS")
    square_train, square_valid = compile_smbd_square()

    if legacy:
        log.info("Legacy serial batch iterator")
        bi     = BatchIterator(ds_path, input_type="features")
        start = time.time()
        for x,y,z in bi.epoch_smbd_train():
            z=square_train(x,y)
        stop = time.time()
        log.info("Serial loading training epoch time={}s".format(stop - start))

    if threaded:
        log.info("Threaded batch loader")
        dl = ThreadedLoader(ds_path+"train/")
        start = time.time()
        for x,y,z in dl.epoch():
            z=square_train(x,z)
        stop = time.time()
        time.sleep(1)
        dl.stop()
        log.info("Threaded loading training epoch time={}s".format(stop - start))

    if multiprocessed:
        log.info("Multiprocessed batch loader")
        dl = MultiprocessedLoader(ds_path+"train/")
        start = time.time()
        for x,y,z in dl.epoch():
            z=square_train(x,z)
        stop = time.time()
        time.sleep(1)
        dl.stop()
        log.info("Multiprocessed loading training epoch time={}s".format(stop - start))

def test_loadings_passing(legacy=True, threaded=True, multiprocessed=True):
    ds_path="/home/tristan/data/Imagenet/datasets/Sensembed_A/256_224/center_crop/"
    log.info("TESTING LOADING WITH PASSING")

    if legacy:
        bi     = BatchIterator(ds_path, input_type="features")
        log.info("Legacy serial batch iterator")
        start = time.time()
        for x,y,z in bi.epoch_smbd_train():
            pass
        stop = time.time()
        log.info("Serial loading training epoch time={}s".format(stop - start))

    if threaded:
        log.info("Threaded batch loader")
        dl = ThreadedLoader(ds_path+"train/")
        start = time.time()
        for x,y,z in dl.epoch():
            pass
        stop = time.time()
        time.sleep(1)
        dl.stop()
        log.info("Threaded loading training epoch time={}s".format(stop - start))

    if multiprocessed:
        log.info("Multiprocessed batch loader")
        dl = MultiprocessedLoader(ds_path+"train/")
        start = time.time()
        for x,y,z in dl.epoch():
            pass
        stop = time.time()
        time.sleep(1)
        dl.stop()
        log.info("Multiprocessed loading training epoch time={}s".format(stop - start))

def test_loadings_hinge(legacy=True, threaded=True, multiprocessed=True):
    ds_path="/home/tristan/data/Imagenet/datasets/Sensembed_A/256_224/center_crop/"
    log.info("TESTING LOADING WITH HINGE LOSS")
    hinge_train, hinge_valid = compile_smbd_hinge_dist()

    if legacy:
        log.info("Legacy serial batch iterator")
        bi     = BatchIterator(ds_path, input_type="features")
        start = time.time()
        for x,y,z in bi.epoch_smbd_train():
            z=hinge_train(x,z)
        stop = time.time()
        log.info("Serial loading training epoch time={}s".format(stop - start))

    if threaded:
        log.info("Threaded batch loader")
        dl = ThreadedLoader(ds_path+"train/")
        start = time.time()
        for x,y,z in dl.epoch():
            z=hinge_train(x,y)
        stop = time.time()
        time.sleep(1)
        dl.stop()
        log.info("Threaded loading training epoch time={}s".format(stop - start))

    if multiprocessed:
        log.info("Multiprocessed batch loader")
        dl = MultiprocessedLoader(ds_path+"train/")
        start = time.time()
        for x,y,z in dl.epoch():
            z=hinge_train(x,y)
        stop = time.time()
        time.sleep(1)
        dl.stop()
        log.info("Multiprocessed loading training epoch time={}s".format(stop - start))

def run_all_tests(n_epoch=1, legacy=True, threaded=True, multiprocessed=True):
    for i in range(n_epoch):
        test_loadings_passing(legacy, threaded, multiprocessed)
        test_loadings_square(legacy, threaded, multiprocessed)
        test_loadings_hinge(legacy, threaded, multiprocessed)
