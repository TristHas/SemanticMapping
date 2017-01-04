#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, time
import yaml

import pandas as pd
import hickle as hkl

from ..util.Helpers import Logger, check_file_path
from ..CaffeZoo.CaffeMetadata import targets_2012
from ..Linking.Linker import merge_wn_sembed

from build_ds import make_minibatches, mapFL_subfolders, split_dataset_ratio, deal_with_corrupted_images
from ILSVRC_preprocessing import mapFP_val_wnid
    ###
    ### I. GLOBALS
    ###

log = Logger()
root_paths_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "..",
                                "paths.yaml")

module_paths_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "paths.yaml")

with open(module_paths_file, 'r') as f:
    paths = yaml.load(f)
    ds_dir           = paths["ds_path"]
    ilsvrc_image_dir = paths["ilsvrc_photos"]

with open(root_paths_file, 'r') as f:
    paths = yaml.load(f)
    # We have to share Imagenet photo dir in root_path because Imagenet module also uses it for Download
    imnet_image_dir  = paths["default_photo_dir"]





sensembedA_name = "Sensembed_A"
sensembedB_name = "Sensembed_B"
labelmap_name   = "wnidlabelmap"
    ###
    ### II. HELPERS
    ###
    ### Helpers SHOULD BE IN BUILD_DS? Makes use of ds_dir so that would be weird?
def serialize_dataset_labelmap(values, ds_name, dest_dir = ds_dir):
    """
        Given a set of WNID, serialize a WNID <-> LABEL mapping in the dataset folder.
    """
    ds_index_path = os.path.join(dest_dir, ds_name, labelmap_name)
    check_file_path(ds_index_path)
    indice_map    = pd.DataFrame(data={"WNID":values, "LABEL":range(len(values))})
    indice_map.to_pickle(ds_index_path)

def sensembed_available_synsets():
    return merge_wn_sembed().WNID



    ###
    ### III. SENSEMBED datasets
    ###

### Dataset A
def sync_Sensembed_A_labelmap(dest_dir = ds_dir):
    """
    """
    train_dir = os.path.join(ilsvrc_image_dir, "train")
    df = mapFL_subfolders(train_dir)
    df = df[df.WNID.isin(sensembed_available_synsets())]
    log.info("Serializing index map {}. Contains {} synsets".format(sensembedA_name, len(df.WNID.unique())))
    serialize_dataset_labelmap(df.WNID.unique(), sensembedA_name, dest_dir)


def load_Sensembed_A_labelmap(dest_dir = ds_dir):
    """
    """
    ds_index_path = os.path.join(dest_dir, sensembedA_name, labelmap_name)
    return pd.read_pickle(ds_index_path)

def build_Sensembed_A_ds(batch_size=256, img_size=224, resizing = "center_crop", dest_dir = ds_dir):
    """
    """
    #sync_Sensembed_A_labelmap(dest_dir)
    label_wnid_df = load_Sensembed_A_labelmap(dest_dir)
    dest_dir = os.path.join(dest_dir, sensembedA_name, "_".join([str(batch_size),str(img_size)]), str(resizing))
    log.info("Building training for ILSVRC2015 sensembed into {}".format(dest_dir))
    #
    #
    #log.info("Building training dataset")
    #train_dir  = os.path.join(ilsvrc_image_dir, "train")
    #log.info("Mapping  {} file paths...".format(train_dir))
    #fp_wnid_df = mapFL_subfolders(train_dir)
    #log.info("Removing non available sensembed vectors")
    #fp_wnid_df = fp_wnid_df[fp_wnid_df.WNID.isin(label_wnid_df.WNID)]
    #log.info("Merging labels")
    #df = fp_wnid_df.merge(label_wnid_df)
    #split_dataset_ratio(df, ratio = (1,0,0))
    #log.info("Serializing {} images for training dataset".format(len(fp_wnid_df)))
    #make_minibatches(df, dest_dir, batch_size = batch_size,
    #                img_size=img_size, mean_filename="val_mean",
    #                resizing = resizing)
    #log.info("Training set serialized in {}".format(dest_dir))

    log.info("Building validation dataset")
    valid_dir  = os.path.join(ilsvrc_image_dir, "val")
    log.info("Mapping  {} file paths...".format(valid_dir))
    fp_wnid_df = mapFP_val_wnid()
    log.info("{} Pictures mapped. Removing non available sensembed vectors...".format(len(fp_wnid_df)))
    fp_wnid_df = fp_wnid_df[fp_wnid_df.WNID.isin(label_wnid_df.WNID)]
    log.info("Remains {} pictures. Merging labels...".format(len(fp_wnid_df)))
    df = fp_wnid_df.merge(label_wnid_df)
    split_dataset_ratio(df, ratio = (0,1,0))
    log.info("Serializing {} images for valid dataset".format(len(fp_wnid_df)))
    make_minibatches(df, dest_dir, batch_size = batch_size,
                    img_size=img_size, mean_filename="val_mean",
                    resizing = resizing)
    log.info("Training set serialized in {}".format(dest_dir))



### Dataset B
def sync_Sensembed_B_labelmap(dest_dir = ds_dir):
    """
    """
    df1 = mapFL_subfolders(ilsvrc_img_dir)
    df1 = df1[df1.WNID.isin(sensembed_available_synsets())]
    df2 = mapFL_subfolders(imnet_image_dir)
    df2 = df2[df2.WNID.isin(sensembed_available_synsets())]
    df2 = df2[-df2.WNID.isin(df1)]
    df  = pd.concat([df1, df2])
    log.info("Serializing index map {}. Contains {} synsets".format(sensembedB_name, len(df.WNID.unique())))
    serialize_dataset_labelmap(df.WNID.unique(), sensembedB_name, dest_dir)

def load_Sensembed_B_labelmap(dest_dir = ds_dir):
    """
    """
    ds_index_path = os.path.join(dest_dir, sensembedB_name, labelmap_name)
    return pd.read_pickle(ds_index_path)

def build_Sensembed_B_ds(batch_size=256, img_size=224, resizing = "center_crop", dest_dir = ds_dir):
    """
    """
    sync_Sensembed_A_labelmap(dest_dir)
    dest_dir = os.path.join(dest_dir, "_".join([str(batch_size),str(img_size)]), str(resizing))
    log.info("Building training dataset for ILSVRC2015 sensembed into {}".format(dest_dir))
    log.info("Mapping  {} file paths".format(ilsvrc_img_dir))
    df1 = mapFL_subfolders(ilsvrc_img_dir)
    log.info("Mapping  {} file paths".format(imnet_image_dir))
    df2 = mapFL_subfolders(imnet_image_dir)
    log.info("Removing non available sensembed vectors")
    df1 = df1[df1.WNID.isin(sensembed_available_synsets())]
    df2 = df2[df2.WNID.isin(sensembed_available_synsets())]
    log.info("Removing duplicate synsets from imagenet")
    df2 = df2[-df2.WNID.isin(df1)]
    log.info("Concatenating both sources")
    df  = pd.concat([df1, df2])
    log.info("Dealing with corrupted images")
    df  = deal_with_corrupted_images(df, dest_dir)
    log.info("Merging index map index")
    df = df.merge(load_SEMBED_labelmap())
    log.info("Mapped {} target labels".format(len(df.LABEL.unique())))
    split_dataset_ratio(df, ratio = (8,1,1))
    make_minibatches(df, dest_dir, batch_size = batch_size,
                    img_size=img_size, mean_filename="val_mean",
                    resizing = resizing)





