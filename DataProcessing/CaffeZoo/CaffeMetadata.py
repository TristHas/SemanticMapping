#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys, tarfile, pickle
from ..util.Helpers import Logger, download, open_remote
log = Logger()

import numpy as np
import pandas as pd
import requests
import yaml

root_paths_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "..",
                                "paths.yaml")


module_paths_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "paths.yaml")


with open(root_paths_file, "r") as f_root:
    root = yaml.load(f_root)["caffezoo_root"]


### Loaders
def targets_2015():
    terms = load_terms_wnid()
    indices = load_target_2015()
    return indices.merge(terms)

def targets_2012():
    terms = load_terms_wnid()
    indices = load_target_2012()
    labels = indices.merge(terms)
    labels = labels[np.logical_not(np.logical_and(labels.WNID=="n02012849", labels.CaffeLabel == 517))]
    labels = labels[np.logical_not(np.logical_and(labels.WNID=="n03126707", labels.CaffeLabel == 134))]
    return labels

def load_terms_wnid():
    with open(module_paths_file, "r") as f_caffe:
        paths = yaml.load(f_caffe)
        df_syns_words = os.path.join(root, paths["words_synset_df"])
    return pd.read_pickle(df_syns_words)

def load_target_2012():
    with open(module_paths_file, "r") as f_caffe:
        paths = yaml.load(f_caffe)
        df_syns_words = os.path.join(root, paths["2012_target_df"])
    return pd.read_pickle(df_syns_words)

def load_target_2015():
    with open(module_paths_file, "r") as f_caffe:
        paths = yaml.load(f_caffe)
        df_syns_words = os.path.join(root, paths["2015_target_df"])
    return pd.read_pickle(df_syns_words)


### Download and init
def init_directory():
    """
        DOC
    """
    with open(module_paths_file, "r") as f_caffe:
        paths = yaml.load(f_caffe)
        f_syns_words = os.path.join(root, paths["words_synset_f"])
        df_syns_words = os.path.join(root, paths["words_synset_df"])
        url_syns_words = paths["words_synset_url"]
        f_target_2012 = os.path.join(root,paths["2012_target_f"])
        url_target_2012 = paths["2012_target_url"]
        f_target_2015 = os.path.join(root,paths["2015_target_f"])
        url_target_2015 = paths["2015_target_url"]
    log.info("Creating CaffeZoo data directory structure")
    if not os.path.isdir(os.path.dirname(f_syns_words)):
        os.makedirs(os.path.dirname(f_syns_words))
    if not os.path.isdir(os.path.dirname(df_syns_words)):
        os.makedirs(os.path.dirname(df_syns_words))
    log.info("Downloading synset_words from {} to {} ...".format(url_syns_words, f_syns_words))
    tar_obj = tarfile.open(fileobj=open_remote(url_syns_words))
    tar_obj.extract("synset_words.txt", os.path.dirname(f_syns_words))
    log.info("Downloading 2012 weights from {} to {} ...".format(url_target_2012, f_target_2012))
    download(url_target_2012, f_target_2012)
    log.info("Downloading 2015 weights from {} to {} ...".format(url_target_2015, f_target_2015))
    download(url_target_2015, f_target_2015)

### Syncers
def sync_caffeindex_wnid():
    """
    """
    with open(module_paths_file, "r") as f_caffe:
        paths = yaml.load(f_caffe)
        f_syns_words = os.path.join(root, paths["words_synset_f"])
        df_syns_words = os.path.join(root, paths["words_synset_df"])
    data = {"WNID":[], "terms":[]}
    with open(f_syns_words, "r") as rawfile:
        for line in rawfile:
            parsed = line.split(" ", 1)
            assert len(parsed) == 2
            data["WNID"].append(parsed[0])
            data["terms"].append(parsed[1].strip())
    df = pd.DataFrame(data=data)
    df.to_pickle(df_syns_words)

def sync_2012_targets():
    """
    """
    with open(module_paths_file, "r") as f_caffe:
        paths = yaml.load(f_caffe)
        df_file = os.path.join(root,paths["2012_target_df"])
        data_file = os.path.join(root,paths["2012_target_f"])
    data = pickle.load(open(data_file, "r"))["synset_words"]
    df = pd.DataFrame(data={"terms": data, "CaffeLabel": range(len(data))})
    df.to_pickle(df_file)

def sync_2015_targets():
    """
        MIGHT NEED TO TRIM The first and last 8 dimensions.
    """
    with open(module_paths_file, "r") as f_caffe:
        paths = yaml.load(f_caffe)
        df_file = os.path.join(root,paths["2015_target_df"])
        data_file = os.path.join(root,paths["2015_target_f"])
    data = pickle.load(open(data_file, "r"))["synset words"]
    df = pd.DataFrame(data={"terms": data, "CaffeLabel": range(len(data))})
    df.to_pickle(df_file)




