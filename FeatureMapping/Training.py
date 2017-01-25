#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
from timeit import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from DataProcessing.util.Helpers import Logger
from FeatureMapping.Trainer import SMBDTrainer, plot_hist

log = Logger()

def run_tests_smbd(nepoch=70):
    """
    """
    log.info("TESTING SMBD")
    regs = [0.0001, 0.000001, 0.00000001]
    lrs  = [0.1, 0.01, 0.001]
    for lr in lrs:
        for reg in regs:
            log.info("LR:{} || REG:{}".format(lr,reg))
            log.info("Hinge dist training")
            tr = SMBDTrainer(reg=reg, lr=lr, mode="hinge_dist")
            tr.run_training(nepoch=nepoch)
            tr.drop_hist("/home/tristan/Desktop/smbd_tests/reg_test_smbd_hinge_dist_" + str(reg) + "_" + str(lr))
            log.info("Struct training")
            tr = SMBDTrainer(reg=reg, lr=lr, mode="struct")
            tr.run_training(nepoch=nepoch)
            tr.drop_hist("/home/tristan/Desktop/smbd_tests/reg_test_smbd_struct_" + str(reg) + "_" + str(lr))
            log.info("Square training")
            tr = SMBDTrainer(reg=reg, lr=lr, mode="square")
            tr.run_training(nepoch=nepoch)
            tr.drop_hist("/home/tristan/Desktop/smbd_tests/reg_test_smbd_square_" + str(reg) + "_" + str(lr))
    for lr in lrs:
        for reg in regs:
            log.info("LR:{}    ||  REG:{}".format(lr, reg))
            log.info("Hinge dot training")
            tr = SMBDTrainer(reg=reg, lr=lr, mode="hinge_dot")
            tr.run_training(nepoch=nepoch)
            tr.drop_hist("/home/tristan/Desktop/smbd_tests/reg_test_smbd_hinge_dot_" + str(reg) + "_" + str(lr))




tr = SMBDTrainer(lr = 1, reg = 0.001, mode = "hinge")
tr.run_training_adjusted(10, {0:1,1:0.1,5:0.01,10:None})
tr.drop_hist("/media/tristan/41d01b1d-062b-48dc-997b-b029783eca9f/Imagenet/datasets/Sensembed_A/hist/hinge_sembed")
tr.stop()

dims = [10,20,50,100,200,400,800,1600]
for dim in dims:
    tr = SMBDTrainer(labels = dim, lr = 1, reg = 0.001, mode = "hinge")
    tr.run_training_adjusted(10,{0:1,1:0.1,5:0.01,10:None})
    tr.stop()
    tr.drop_hist("/media/tristan/41d01b1d-062b-48dc-997b-b029783eca9f/Imagenet/datasets/Sensembed_A/hist/hinge_random_{}".format(dim))



def pred_distribution(trainer):
    counter = np.zeros(908).astype("uint64")
    for x,y,z in tr.valid_dl.epoch():
        score, pred = tr.valid_func(x,y)
        res=tr.v.smd_k_best_pred(pred)
        count = np.bincount(res.flatten()).astype("uint64")
        count = np.pad(count, (0,908 - count.shape[0]), "constant")
        counter += count
    return counter

def valid_label_distribution(trainer):
    counter = np.zeros(908).astype("uint64")
    for x,y,z in tr.valid_dl.epoch():
        count = np.bincount(z.flatten().astype("int64")).astype("uint64")
        count = np.pad(count, (0,908 - count.shape[0]), "constant")
        counter += count
    return counter

def train_label_distribution(trainer):
    counter = np.zeros(908).astype("uint64")
    for x,y,z in tr.train_dl.epoch():
        count = np.bincount(z.flatten().astype("int64")).astype("uint64")
        count = np.pad(count, (0,908 - count.shape[0]), "constant")
        counter += count
    return counter
