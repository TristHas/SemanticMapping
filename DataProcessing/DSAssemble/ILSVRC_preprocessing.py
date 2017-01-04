#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os, time
import yaml
from build_ds import make_minibatches, mapFL_subfolders, split_dataset_ratio
import pandas as pd
from ..util.Helpers import Logger
from ..CaffeZoo.CaffeMetadata import targets_2012
log = Logger()


module_paths_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "paths.yaml")



def label_map_2012():
    """
    """
    df1 = load_ILSVRC_labelmap()
    df2 = targets_2012()
    return df1.merge(df2)[["LABEL", "CaffeLabel"]]

def sync_ILSVRC_labelmap():
    """
    """
    with open(module_paths_file, 'r') as f:
        paths = yaml.load(f)
        raw_map = paths["raw_label_map"]
        label_map = paths["label_map"]

    clsloc_map = pd.read_table(raw_map,
                               names=["WNID", "LABEL", "lemma"],
                               delim_whitespace=True)
    del clsloc_map["lemma"]
    clsloc_map.to_pickle(label_map)

def load_ILSVRC_labelmap():
    """
    """
    with open(module_paths_file, 'r') as f:
        paths = yaml.load(f)
        label_map = paths["label_map"]
    return pd.read_pickle(label_map)

def build_train_ds(batch_size=256, img_size=224, dest_dir = "/home/tristan/data/", resizing = "center_crop"):
    """
    """
    log.info("Building train dataset")
    with open(module_paths_file, 'r') as f:
        paths = yaml.load(f)
        dest_dir = paths["ilsvrc_ds"]
        img_dir = paths["train_photo_dir"]
    dest_dir = os.path.join(dest_dir, "_".join([str(batch_size),str(img_size)]), str(resizing))
    log.info("Taking images from {}. Serializing into {}...".format(img_dir, dest_dir))
    log.info("mapping filepaths to label...")
    start = time.time()
    df1 = mapFL_subfolders(img_dir)
    df2 = load_ILSVRC_labelmap()
    df3 = df1.merge(df2)[["FP", "LABEL"]]
    log.info("labels mapped in {} ms. /n Associating numerical labels".format(time.time() - start))
    log.info("Fake splitting...")
    start = time.time()
    split_dataset_ratio(df3, ratio = (2,0,0))
    log.info("labels fake splitted in {} ms".format(time.time() - start))
    log.info("Serializing Minibatches...")
    start = time.time()
    make_minibatches(df3, dest_dir, batch_size = batch_size,
                    img_size=img_size, mean_filename="val_mean",
                    resizing = resizing)
    log.info("Minibatches serialized in {} ms".format(time.time() - start))

def build_val_ds(batch_size, img_size, resizing = "center_crop"):
    """
    """
    log.info("Building val dataset")
    with open(module_paths_file, 'r') as f:
        dest_dir = yaml.load(f)["ilsvrc_ds"]
    dest_dir = os.path.join(dest_dir, "_".join([str(batch_size),str(img_size)]), str(resizing))
    log.info("mapping filepaths to label...")
    start = time.time()
    df = map_val_gorlabel_imfilepath()
    log.info("labels mapped in {} ms".format(time.time() - start))
    log.info("Fake splitting...")
    start = time.time()
    split_dataset_ratio(df, ratio = (0,2,0))
    log.info("labels fake splitted in {} ms".format(time.time() - start))
    log.info("Serializing Minibatches...")
    start = time.time()
    make_minibatches(df, dest_dir, batch_size = batch_size, img_size=img_size,
                     mean_filename="val_mean", resizing = resizing)
    log.info("Minibatches serialized in {} ms".format(time.time() - start))

def build_toy_val_ds(batch_size, img_size, dest_dir = "/home/tristan/data/dummy_crop", resizing = "crop"):
    """
    """
    log.info("Building val dataset")
    log.info("mapping filepaths to label...")
    start = time.time()
    df = map_val_gorlabel_imfilepath()
    log.info("labels mapped in {} ms".format(time.time() - start))
    log.info("Fake splitting...")
    start = time.time()
    split_dataset_ratio(df, ratio = (0,2,0))
    df = df.loc[:batch_size-1]
    log.info("labels fake splitted in {} ms".format(time.time() - start))
    log.info("Serializing Minibatches...")
    start = time.time()
    make_minibatches(df, dest_dir, batch_size = batch_size,
                     img_size=img_size, mean_filename="val_mean",
                     resizing = resizing)
    log.info("Minibatches serialized in {} ms".format(time.time() - start))

def map_val_gorlabel_imfilepath():
    """
    """
    with open(module_paths_file, 'r') as f:
        paths = yaml.load(f)
        img_dir = paths["val_photo_dir"]
        ground_truth = paths["val_ground_truth"]
        numerotation = paths["val_numerotation"]
    filepath_imageId = pd.read_table(   numerotation,
                                        delim_whitespace=True,
                                        names = ["FN", "image_id"])
    filepath_imageId["FP"] = filepath_imageId["FN"].apply(lambda x:os.path.join(img_dir, x) + ".JPEG")
    del filepath_imageId["FN"]
    valid_wnid = pd.read_table(ground_truth,
                                names=["LABEL"])
    valid_wnid["image_id"] = valid_wnid.index + 1
    filepath_label = filepath_imageId.merge(valid_wnid)
    del filepath_label["image_id"]
    return filepath_label

def mapFP_val_wnid():
    """
    """
    df1 = map_val_gorlabel_imfilepath()
    df2 = load_ILSVRC_labelmap()
    df = df1.merge(df2)[["FP", "WNID"]]
    return df





























