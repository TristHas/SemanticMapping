#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
sys.path.append("..")


from timeit import time
import cProfile
import hickle as hkl
import pandas as pd

from DataProcessing.DSAssemble.ILSVRC_Semantic import load_SEMBED_labelmap
from DataProcessing.Linking.Linker import merge_wn_sembed


s_load_x = 0
t_load_x = 0

s_load_y = 0
t_load_y = 0

s_pd_merge = 0
t_pd_merge = 0

s_pd_select = 0
t_pd_select = 0

s_pd_else = 0
t_pd_else = 0

def get_label_sembed_map():
    imnet_sembed   = merge_wn_sembed()
    imnet_labels = load_SEMBED_labelmap()
    return imnet_labels.merge(imnet_sembed)


index_map = get_label_sembed_map()


def get_batch(i, ds_path):
    s_load_x = time.time()
    x = hkl.load(os.path.join(ds_path, "features", str(i)))
    global t_load_x
    t_load_x += time.time() - s_load_x

    s_load_y = time.time()
    y = hkl.load(os.path.join(ds_path, "label", str(i)))
    global t_load_y
    t_load_y += time.time() - s_load_y

    s_pd_select = time.time()
    y = pd.DataFrame(data={"LABEL":y})
    index = index_map[index_map.LABEL.isin(y.LABEL)]
    global t_pd_select
    t_pd_select += time.time() - s_pd_select

    s_pd_merge = time.time()
    y = y.reset_index().merge(index, how="left").set_index('index')
    global t_pd_merge
    t_pd_merge += time.time() - s_pd_merge

    s_pd_else = time.time()
    y = y.drop(["LABEL", "BN", "POS", "WNID", "gp"], axis=1).get_values().astype("float32")
    global t_pd_else
    t_pd_else += time.time() - s_pd_else

    return x,y


def speed_loading(n=1000):
    valid_dir     = "/home/tristan/data/dummysensembed/test/"
    for i in range(n):
        x,y = get_batch(i, valid_dir)

if __name__ == '__main__':
    #cProfile.run('speed_loading()')
    n = 1000
    print "{} Epochs".format(n)
    start = time.time()
    speed_loading(n)
    stop = time.time()
    print "Loading x = {}% \t\t Time per epoch={}".format(int(100*t_load_x/(stop - start)), t_load_x/n)
    print "Loading y = {}% \t\t Time per epoch={}".format(int(100*t_load_y/(stop - start)), t_load_y/n)
    print "Pandas select = {}% \t\t Time per epoch={}".format(int(100*t_pd_select/(stop - start)), t_pd_select/n)
    print "Pandas merge = {}% \t\t Time per epoch={}".format(int(100*t_pd_merge/(stop - start)), t_pd_merge/n)
    print "Pandas else = {}% \t\t Time per epoch={}".format(int(100*t_pd_else/(stop - start)), t_pd_else/n)

