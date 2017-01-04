#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
from timeit import time
sys.path.append("..")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import theano

from DataProcessing.util.Helpers import Logger
from Helpers import Validator, BatchIterator, get_Sensembed_A_labelmap
from MappingFunctions import *

log = Logger()

class SMBDTrainer(object):
    def __init__(self, ds_path="/home/tristan/data/Imagenet/datasets/Sensembed_A/256_224/center_crop/",
                       lr = 0.1, reg = 0.0, margin = 0.1, hinge_coef = 2, mode = "struct"):
        """
        """
        assert mode in ["struct", "square", "hinge_dist", "hinge_dot"]
        self.lr     = theano.shared(np.array(lr, dtype="float32"))
        self.reg    = theano.shared(np.array(reg, dtype="float32"))
        self.margin     = theano.shared(np.array(margin, dtype="float32"))
        self.hinge_coef    = theano.shared(np.array(hinge_coef, dtype="float32"))
        self.bi     = BatchIterator(ds_path, input_type="features")
        self.v      = Validator()
        self.hist   = pd.DataFrame(columns=["reg", "lr", "score_type", "scores", "epoch"])
        self.epoch_count = 0
        self.mode   = mode
        if self.mode == "square":
            self.train_func, self.valid_func = compile_smbd_square(self.lr, self.reg)
        if self.mode == "struct":
            self.train_func, self.valid_func = compile_smbd_struct(self.lr, self.reg)
        if self.mode == "hinge_dist":
            self.train_func, self.valid_func = compile_smbd_hinge_dist(self.lr, self.reg, self.margin)
        if self.mode == "hinge_dot":
            self.train_func, self.valid_func = compile_smbd_hinge_dot(self.lr, self.reg,
                                                                      self.margin, self.hinge_coef)

    def set_lr(self, lr):
        """
        """
        self.lr.set_value(lr)

    def set_reg(self, reg):
        """
        """
        self.reg.set_value(reg)

    def set_margin(self, margin):
        """
        """
        self.margin.set_value(margin)

    def set_hinge_coef(self, hinge_coef):
        """
        """
        self.hinge_coef.set_value(hinge_coef)

    def run_valid_epoch(self):
        """
        """
        val_sc1, val_sc2 = 0,0
        start = time.time()
        for x,y,z in self.bi.epoch_smbd_valid():
            if self.mode == "square":
                val,out = self.valid_func(x,y)
            if self.mode in ["struct", "hinge_dist", "hinge_dot"]:
                val,out = self.valid_func(x,z)
            val_sc1 += val
            val_sc2 += self.v.smbd_top_k_scores(out,z)
        val_sc1 /= len(self.bi.valid_batches)
        val_sc2 /= len(self.bi.valid_batches)
        log.info("Average valid score= {}. Average top-5 error= {}% Computed in {}".format(val_sc1, 100 * val_sc2, time.time() - start))
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
        for x,y,z in self.bi.epoch_smbd_train():
            if self.mode == "square":
                tr_sc += self.train_func(x,y)
            if self.mode in ["struct", "hinge_dist", "hinge_dot"]:
                tr_sc += self.train_func(x,z)
        tr_sc /= len(self.bi.train_batches)
        log.info("Training average score= {}. Computed in {}s".format(tr_sc, time.time() - start))
        df = pd.DataFrame.from_records([{"reg":self.reg.get_value(), "lr":self.lr.get_value(),
                                "score_type": "training", "scores":tr_sc, "epoch":self.epoch_count}])
        self.hist = pd.concat([self.hist, df])

    def run_training(self, nepoch):
        """
        """
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


class CLASTrainer(object):
    def __init__(self, ds_path="/home/tristan/data/Imagenet/datasets/Sensembed_A/256_224/center_crop/",
                       lr = 0.1, reg = 0.0, compile_func = compile_classification):
        """
        """
        self.lr     = theano.shared(np.array(lr, dtype="float32"))
        self.reg    = theano.shared(np.array(reg, dtype="float32"))
        self.bi     = BatchIterator(ds_path, input_type="features")
        self.v      = Validator()
        self.hist   = pd.DataFrame(columns=["reg", "lr", "score_type", "scores", "epoch"])
        self.epoch_count = 0
        self.train_func, self.valid_func = compile_func(self.lr, self.reg)

    def set_lr(self, lr):
        """
        """
        self.lr.set_value(lr)

    def set_reg(self, reg):
        """
        """
        self.reg.set_value(reg)

    def run_valid_epoch(self):
        """
        """
        val_sc1, val_sc2 = 0,0
        start = time.time()
        for x,y in self.bi.epoch_clas_valid():
            val,out = self.valid_func(x,y)
            val_sc1 += val
            val_sc2 += self.v.clas_top_k_scores(out,y)
        val_sc1 /= len(self.bi.valid_batches)
        val_sc2 /= len(self.bi.valid_batches)
        log.info("Average valid score= {}. Average top-5 error= {}% Computed in {}".format(val_sc1, 100 * val_sc2, time.time() - start))
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
        for x,y in self.bi.epoch_clas_train():
            tr_sc += self.train_func(x,y)
        tr_sc /= len(self.bi.train_batches)
        log.info("Training average score= {}. Computed in {}s".format(tr_sc, time.time() - start))
        df = pd.DataFrame.from_records([{"reg":self.reg.get_value(), "lr":self.lr.get_value(),
                                "score_type": "training", "scores":tr_sc, "epoch":self.epoch_count}])
        self.hist = pd.concat([self.hist, df])

    def run_training(self, nepoch):
        """
        """
        self.run_valid_epoch()
        for i in range(nepoch):
            log.info("{}th Epoch! Total {} epochs done".format(i, self.epoch_count))
            self.run_train_epoch()
            self.run_valid_epoch()

    def drop_hist(self, file_path = "/home/tristan/Desktop/default_clas_hist"):
        """
        """
        self.hist.to_pickle(file_path)

    def plot_hist(self,training=True, valid=True, topk=True):
        """
        """
        hist = self.hist.set_index(self.hist.epoch)
        if training:
            hist[hist.score_type=="training"].scores.plot(color = "r")
        if valid:
            hist[hist.score_type=="valid"].scores.plot(color = "b")
        if topk:
            hist[hist.score_type=="topk"].scores.plot(color = "g")
        plt.show()

    def run_training_adjusted(self, nepoch=70, table = {0:0.1,5:0.01,15:0.001,30:0.0001, 50:0.00001}):
        self.run_valid_epoch()
        for i in range(nepoch):
            log.info("{}th Epoch! Total {} epochs done".format(i, self.epoch_count))
            if i in table.keys():
                log.info("Setting lr to {}".format(table[i]))
                self.set_lr(table[i])
            self.run_train_epoch()
            self.run_valid_epoch()
