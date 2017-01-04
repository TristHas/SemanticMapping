#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import yaml
import numpy as np
import pandas as pd

root_paths_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "..",
                                "paths.yaml")
module_paths_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "paths.yaml")

with open(root_paths_file, "r") as f_root:
    root_path = yaml.load(f_root)
    root = root_path["linking_root"]
    imagenet_metadata_frame = root_path["imagenet_synset_metadata_frame"]

with open(module_paths_file, 'r') as f_imnet:
    link_path = yaml.load(f_imnet)
    file_wn31_bn   = os.path.join(root,link_path["file_wn31_bn"])
    file_wn30_bn   = os.path.join(root,link_path["file_wn30_bn"])
    file_wn30_wn31        = os.path.join(root,link_path["file_wn30_wn31"])
    file_wn20_wn30       = os.path.join(root,link_path["file_wn20_wn30"])
    file_wn31_dbp      = os.path.join(root,link_path["file_wn31_dbp"])
    file_bn_dbp    = os.path.join(root,link_path["file_bn_dbp"])
    file_merged     = os.path.join(root,link_path["file_merged"])
    file_bn_sembed      = os.path.join(root,link_path["file_bn_sembed"])
    file_wn20_dbp = os.path.join(root,link_path["file_wn20_dbp"])
    file_map_wn30_wn31    = os.path.join(root,link_path["file_map_wn30_wn31"])
    file_map_wn31_bn     = os.path.join(root,link_path["file_map_wn31_bn"])
    file_map_wn31_dbpres      = os.path.join(root,link_path["file_map_wn31_dbpres"])
    file_map_bn_dbpres = os.path.join(root,link_path["file_map_bn_dbpres"])


###################################################
###                   METHODS                   ###
###################################################

# Loaders
def load_wn30_bn():
    return pd.read_pickle(file_wn30_bn)

def merge_wn_sembed():
    with open(root_paths_file, "r") as f_root:
        root_path = yaml.load(f_root)
        store_path = root_path["sensembed_store"]
        imsynsets_vectors = root_path["sensembed_imsynsets"]
    with pd.get_store(store_path) as store:
        df = store[imsynsets_vectors]
        df2 = load_wn30_bn().rename(columns={"BNID":"BN"})
    return df.merge(df2)






# OLD ONES
def load_wn31_bn():
    return pd.read_pickle(file_wn31_bn)

def load_wn30_wn31():
    return pd.read_pickle(file_wn30_wn31)

def load_wn20_wn30():
    return pd.read_pickle(file_wn20_wn30)

def load_bn_dbpres():
    return pd.read_pickle(file_bn_dbp)

def load_wn31_dbpres():
    return pd.read_pickle(file_wn31_dbp)

def load_merged_links():
    return pd.read_pickle(file_merged)

# Mergers
def merge_imnet_bnet():
    x = load_wn31_bn()
    y = load_wn30_wn31()
    xy = x.merge(y)
    xy = xy.set_index(xy.WN30.apply(lambda x:"n"+x))
    z = pd.read_pickle(imagenet_metadata_frame).numImage
    return xy.join(z)

def merge_imnet_dbpres():
    y = load_wn31_dbpres()
    x = load_wn30_wn31()
    xy = x.merge(y)
    xy = xy.set_index(xy.WN30.apply(lambda x:"n"+x))
    z = pd.read_pickle(imagenet_metadata_frame).numImage
    return xy.join(z)



# Syncers
def sync_bn_sensembed():
    tab = [[],[]]
    with open(file, "r") as f:
        for line in f:
            data = line.split()
            cond = data[0].split(":")
            if len(cond) == 2:
                if cond[1].endswith("n"):
                    bns = cond[1][:-1]
                    if bns in index:
                        tab[0].append(bns), tab[1].append(data[1:])
    return pd.DataFrame(index = tab[0],data = tab[1])

def sync_wn31_bn():
    tmp = pd.DataFrame.from_csv(file_map_wn31_bn)
    res = pd.DataFrame(data = {
        "BN":tmp.index.to_series().apply(lambda x:x.split(":")[-1][:-1]),
        "WN31":tmp.wn.apply(lambda x:x.split("/")[-1][1:-2])}
    ).reset_index()
    del res["synsID"]
    res.to_pickle(file_wn31_bn)

def sync_wn30_wn31():
    tab = [[],[]]
    with open(file_map_wn30_wn31, "r") as f:
        for line in f:
            tmp = line.split()
            tab[0].append(tmp[1])
            tab[1].append(tmp[2])
    res = pd.DataFrame(data={"WN30":tab[0], "WN31":tab[1]})
    res.to_pickle(file_wn30_wn31)

def sync_wn31_dbpres():
    tmp = pd.DataFrame.from_csv(file_map_wn31_dbpres).reset_index()
    res = pd.DataFrame(data = {
        "WN31": tmp.wn.apply(lambda x:x.split("/")[-1][1:-2]),
        "DBPRES": tmp.dbpres,
        })
    res.to_pickle(file_wn31_dbp)

def sync_bn_dbpres():
    tmp = pd.DataFrame.from_csv(file_map_bn_dbpres).reset_index()
    res = pd.DataFrame(data = {
        "BN": tmp.BN.apply(lambda x:x.split(":")[-1][:-1]),
        "DBPRES": tmp.DBP,
        })
    res.to_pickle(file_bn_dbp)





