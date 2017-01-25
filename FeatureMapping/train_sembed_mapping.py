#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
from timeit import time

import numpy as np
import pandas as pd
import hickle as hkl
import matplotlib.pyplot as plt
import theano
import theano.tensor as T

from ..DataProcessing.util.Helpers import Logger
from MappingFunctions import *
from Trainers import SMBDTrainer, CLASTrainer

log = Logger()

def plot_df_by_reg_lr(df, training=True, valid=True, topk=True):
    lrs = df.lr.unique()
    regs = df.reg.unique()
    fig, axes = plt.subplots(nrows=len(lrs), ncols=len(regs))
    gps = df.groupby(["lr","reg"])
    for i, lr in enumerate(lrs):
        for j, reg in enumerate(regs):
            if (lr,reg) in gps.groups:
                sdf = gps.get_group((lr,reg)).set_index("epoch")
                if training:
                    sdf[sdf.score_type=="training"].scores.plot(ax=axes[i,j], color = "r")
                if valid:
                    sdf[sdf.score_type=="valid"].scores.plot(ax=axes[i,j], color = "b")
                if topk:
                    sdf[sdf.score_type=="topk"].scores.plot(ax=axes[i,j], color = "g")
                axes[i,j].set_title((lr,reg))

def plot_df_by_reg_type(df, training=True, valid=True, topk=True):
    df.reg=df.reg.apply(float)
    types = df.type.unique()
    regs = df.reg.unique()
    fig, axes = plt.subplots(nrows=len(types), ncols=len(regs))
    gps = df.groupby(["type","reg"])
    for i, t in enumerate(types):
        for j, reg in enumerate(regs):
            if (t,reg) in gps.groups:
                print (t,reg)
                sdf = gps.get_group((t,reg)).set_index("epoch")
                if training:
                    sdf[sdf.score_type=="training"].scores.plot(ax=axes[i,j], color = "r")
                if valid:
                    sdf[sdf.score_type=="valid"].scores.plot(ax=axes[i,j], color = "b")
                if topk:
                    sdf[sdf.score_type=="topk"].scores.plot(ax=axes[i,j], color = "g")
                axes[i,j].set_title((t,reg))

def run_reg_tests_clas(nepoch=2, table = {0:0.1,10:0.01,25:0.001,35:0.0001,50:0.00001}):
    """
    """
    log.info("TESTING CLASSIFICATION REGULARIZATION WITH ADJUSTABLE WEIGHTS")
    regs = [0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001]
    for reg in regs:
        tr = CLASTrainer(reg=reg)
        tr.run_training_adjusted(nepoch, table)
        tr.drop_hist("/home/tristan/Desktop/reg_test_clas_" + str(reg))

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


def mashup_reg_results(dir = "/home/tristan/Desktop/smbd_tests/"):
    """
    """
    files = [os.path.join(dir,x) for x in os.listdir(dir)]
    smbd_prefix ="reg_test_smbd_square_"
    struct_prefix="reg_test_smbd_struct_"
    clas_prefix ="reg_test_clas_"
    df = None
    for f in files:
        t = f.split("_")[-3]
        x = pd.read_pickle(f)
        x["type"] = t
        if df is None:
            df=x
        else:
            df = pd.concat([df,x])
    df.reg=df.reg.apply(float)
    df.lr=df.lr.apply(float)
    return df

if __name__ == "__main__2":
    nepoch= 30
    #table = {0:0.1,10:0.01,25:0.001,35:0.0001,50:0.00001}
    #run_reg_tests_clas(nepoch, table)
    #table = {0:0.1,25:0.01,45:0.001,70:0.0001}
    run_tests_smbd(nepoch)
