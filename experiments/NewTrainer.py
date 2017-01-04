#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
from timeit import time
sys.path.append("..")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import theano

from Loading import ThreadedLoader
from DataProcessing.util.Helpers import Logger
from Helpers import Validator
from NewMappingFunctions import *

log = Logger()

class SMBDTrainer(object):
    def __init__(self, ds_path="/home/tristan/data/Imagenet/datasets/Sensembed_A/256_224/center_crop/",
                       input_unit_norm = True, label_unit_norm = True, distance = "cosine",
                       load_parallel = True, map_parallel = True,
                       lr = 0.1, reg = 0.0, margin = 0.1, mode = "hinge"):
        """
        """
        assert mode in ["dot", "square", "dot_nob", "square_nob", "hinge", "struct"]
        self.lr            = theano.shared(np.array(lr, dtype="float32"))
        self.reg           = theano.shared(np.array(reg, dtype="float32"))
        self.margin        = theano.shared(np.array(margin, dtype="float32"))

        self.train_dl      = ThreadedLoader(os.path.join(ds_path,"train"), input_type="features",
                                            input_unit_norm=input_unit_norm, label_unit_norm=label_unit_norm)
        self.valid_dl      = ThreadedLoader(os.path.join(ds_path,"valid"), input_type="features",
                                            input_unit_norm=input_unit_norm, label_unit_norm=label_unit_norm)

        self.v             = Validator(distance=distance)
        self.hist          = pd.DataFrame(columns=["reg", "lr", "score_type", "scores", "epoch"])

        self.epoch_count   = 0
        self.mode          = mode
        self.compile_functions()

    def compile_functions(self):
        if self.mode == "struct":
            self.train_func, self.valid_func = compile_struct(self.lr, self.reg)
        if self.mode == "hinge":
            self.train_func, self.valid_func = compile_hinge(self.lr, self.reg, self.margin)
        if self.mode == "square":
            self.train_func, self.valid_func = compile_square(self.lr, self.reg)
        if self.mode == "dot":
            self.train_func, self.valid_func = compile_dot(self.lr, self.reg)
        if self.mode == "dot_nob":
            self.train_func, self.valid_func = compile_dot_nobias(self.lr, self.reg)
        if self.mode == "square_nob":
            self.train_func, self.valid_func = compile_square_nobias(self.lr, self.reg)

    def set_lr(self, lr):
        """
        """
        log.info("Setting lr to {}".format(lr))
        self.lr.set_value(lr)

    def set_reg(self, reg):
        """
        """
        log.info("Setting reg to {}".format(reg))
        self.reg.set_value(reg)

    def set_margin(self, margin):
        """
        """
        log.info("Setting margin to {}".format(margin))
        self.margin.set_value(margin)

    def set_hinge_coef(self, hinge_coef):
        """
        """
        log.info("Setting hinge_coef to {}".format(hinge_coef))
        self.hinge_coef.set_value(hinge_coef)

    def run_valid_epoch(self):
        """
        """
        val_sc1, val_sc2 = 0,0
        start = time.time()
        for x,y,z in self.valid_dl.epoch():
            if self.mode in ["hinge", "struct"]:
                val,out = self.valid_func(x,z)
            else:
                val,out = self.valid_func(x,y)
            val_sc1 += val
            val_sc2 += self.v.smbd_top_k_scores(out,z)
        val_sc1 /= self.valid_dl.nbatches
        val_sc2 /= self.valid_dl.nbatches
        log.info("Average valid loss= {}. Average top-5 error= {}% Computed in {}".format(val_sc1, 100 * val_sc2, time.time() - start))
        df1 = pd.DataFrame.from_records([{"reg":self.reg.get_value(), "lr":self.lr.get_value(),
                                "score_type": "valid", "scores":val_sc1, "epoch":self.epoch_count}])
        df2 = pd.DataFrame.from_records([{"reg":self.reg.get_value(), "lr":self.lr.get_value(),
                                "score_type": "topk", "scores":val_sc2, "epoch":self.epoch_count}])
        self.hist = pd.concat([self.hist, df1, df2])

    def run_train_epoch(self):
        """
        """
        tr_sc = 0
        self.epoch_count+=1
        start = time.time()
        for x,y,z in self.train_dl.epoch():
            if self.mode in ["hinge", "struct"]:
                tr_sc += self.train_func(x,z)
            else:
                tr_sc += self.train_func(x,y)
        tr_sc /= self.train_dl.nbatches
        log.info("Average training loss= {}. Computed in {}s".format(tr_sc, time.time() - start))
        df = pd.DataFrame.from_records([{"reg":self.reg.get_value(), "lr":self.lr.get_value(),
                                "score_type": "training", "scores":tr_sc, "epoch":self.epoch_count}])
        self.hist = pd.concat([self.hist, df])

    def run_training(self, nepoch):
        """
        """
        log.info("Training launched for {} epochs. Mode {}".format(nepoch, self.mode))
        self.run_valid_epoch()
        for i in range(nepoch):
            log.info("{}th Epoch! Total {} epochs done".format(i, self.epoch_count))
            self.run_train_epoch()
            self.run_valid_epoch()

    def drop_hist(self, file_path = "/home/tristan/Desktop/default_smbd_hist"):
        """
        """
        self.hist.to_pickle(file_path)

    def plot_hist(self,training=True, valid=True, topk=True):
        hist = self.hist.set_index(self.hist.epoch)
        if training:
            hist[hist.score_type=="training"].scores.plot(color = "r")
        if valid:
            hist[hist.score_type=="valid"].scores.plot(color = "b")
        if topk:
            hist[hist.score_type=="topk"].scores.plot(color = "g")
        plt.show()

    def run_training_adjusted(self, nepoch=70, table = {0:0.1,10:0.01,25:0.001,35:0.0001,50:0.00001}):
        self.run_valid_epoch()
        for i in range(nepoch):
            log.info("{}th Epoch! Total {} epochs done".format(i, self.epoch_count))
            if i in table.keys():
                log.info("Setting lr to {}".format(table[i]))
                self.set_lr(table[i])
            self.run_train_epoch()
            self.run_valid_epoch()

    def stop(self):
        self.train_dl.stop()
        self.valid_dl.stop()


def plot_hist(trainer,training=True, valid=True, topk=True):
    plt.ion()
    hist = trainer.hist.set_index(trainer.hist.epoch)
    if training:
        hist[hist.score_type=="training"].scores.plot(color = "r")
    if valid:
        hist[hist.score_type=="valid"].scores.plot(color = "b")
    if topk:
        hist[hist.score_type=="topk"].scores.plot(color = "g")
    if topk:
        hist.lr.apply( lambda x: np.log10(1/x) / 10).plot(color = "k")
    if topk:
        hist.reg.apply( lambda x: np.log10(1/x)/ 10).plot(color = "m")
    plt.show(False)
