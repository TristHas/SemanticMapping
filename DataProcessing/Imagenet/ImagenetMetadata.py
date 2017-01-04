#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import yaml
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from ..util.Helpers import *
log = Logger()

# Metadata file paths and url
root_paths_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "..",
                                "paths.yaml")
module_paths_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 "paths.yaml")

with open(root_paths_file, "r") as f_root:
    root_path = yaml.load(f_root)
    root = root_path["imagenet_root"]
    syns_metadata_file = root_path["imagenet_synset_metadata_frame"]

with open(module_paths_file, 'r') as f_imnet:
    imnet_path = yaml.load(f_imnet)
    f_wnid_names_pair   = os.path.join(root,imnet_path["f_wnid_names_pair"])
    f_is_a_pairs        = os.path.join(root,imnet_path["f_is_a_pairs"])
    f_syns_struct       = os.path.join(root,imnet_path["f_syns_struct"])
    f_syns_release      = os.path.join(root,imnet_path["f_syns_release"])
    url_syns_release    = imnet_path["url_syns_release"]
    url_syns_struct     = imnet_path["url_syns_struct"]
    url_is_a_pairs      = imnet_path["url_is_a_pairs"]
    url_wnid_names_pair = imnet_path["url_wnid_names_pair"]

###################################################
###             METADATA EXTRACTION             ###
###################################################

def extract_all_metadata(access = "local"):
    """
        Extract Fron Imagenet Metadat a pandas DataFrame with:
            index   = wnids
            columns =   has_son (tuple of child synset)
                        is_a    (tuple of parent synset)
                        numImage(number of photos for synset)
                        terms   (tuple of strings from wordnet)
                        gloss   (string short definition)

    """
    isa = wnids_parents(access)
    gloss = wnids_gloss(access)
    ni = wnids_numImage(access)
    t = wnids_terms(access)
    return pd.DataFrame(data = {
         "has_son"  : isa.has_son,
         "is_a"     : isa.is_a,
         "numImage" : ni.numImage,
         "terms"    : t.terms,
         "gloss"    : gloss.gloss
    })

def sync_syns_metadata(synset = True, raw = True):
    """

    """
    if raw:
        log.debug("Downloading {} to {}".format(url_wnid_names_pair, f_wnid_names_pair))
        download(url_wnid_names_pair, f_wnid_names_pair)
        log.debug("Downloading {} to {}".format(url_syns_release, f_syns_release))
        download(url_syns_release, f_syns_release)
        log.debug("Downloading {} to {}".format(url_is_a_pairs, f_is_a_pairs))
        download(url_is_a_pairs, f_is_a_pairs)
        log.debug("Downloading {} to {}".format(url_syns_struct, f_syns_struct))
        download(url_syns_struct, f_syns_struct)
    if synset:
        if raw:
            df = extract_all_metadata(access = "local")
        else:
            df = extract_all_metadata(access = "remote")
    df.to_pickle(syns_metadata_file)

def load_syns_metadata():
    return pd.read_pickle(syns_metadata_file)

def wnids_parents(access = "local"):
    """
        Parser for wordnet.is_a.txt
        Returns a Pandas DataFrame with index wnid
        and column is-a
    """
    if access == "local":
        f = open(f_is_a_pairs)
    else:
        f = open_remote(url_is_a_pairs)
    wnids, fathers = [],[]
    for line in f.readlines():
        father, wnid = line.split()
        wnids.append(wnid)
        fathers.append(father)
    df      = pd.DataFrame(data= {"wnid":wnids,"is_a":fathers})
    isa     = df.is_a.groupby(df.wnid).agg(lambda x : tuple(x))
    has_son = df.wnid.groupby(df.is_a).agg(lambda x : tuple(x))
    return pd.DataFrame({"is_a":isa,"has_son":has_son})

def wnids_gloss(access = "local"):
    """
        Parses structure_released.xml
        Returns a Pandas DataFrame with index wnid
        and columns terms (tuples of striped strings)
        and gloss (short def)
    """
    if access == "local":
        f = open(f_syns_struct)
    else:
        f = open_remote(url_syns_struct)
    tree = ET.parse(f)
    root = tree.getroot()

    wnids, gloss = [],[]
    for item in root.findall( './/synset'):
        wnids.append(item.attrib["wnid"])
        gloss.append(item.attrib["gloss"])
    df = pd.DataFrame(data = {"wnid": wnids, "gloss":gloss})
    return df.groupby(df.wnid).agg(lambda x:str(x))

def wnids_numImage(access = "local"):
    """
        Parse ReleaseStatus.xml
        Returns a Pandas DataFrame with index wnid
        and column numImages (number of image for synset wnid)
    """
    if access == "local":
        f = open(f_syns_release)
    else:
        f = open_remote(url_syns_release)
    tree = ET.parse(f)
    root = tree.getroot().getchildren()[1]
    wnids, numIm = [], []
    for item in root.findall( './/synset'):
        wnids.append(item.attrib["wnid"])
        numIm.append(item.attrib["numImages"])
    df = pd.DataFrame({"wnid": wnids, "numImage":numIm})
    return df.groupby(df.wnid).agg(lambda x: int(x))

def wnids_terms(access = "local"):
    """
        Parses words.txt
        Returns a Pandas DataFrame with index wnid
        and column terms (tuples of striped strings)
    """
    wnids, terms = [],[]
    if access == "local":
        f = open(f_wnid_names_pair)
    else:
        f= open_remote(url_wnid_names_pair)
    for line in f.readlines():
        wnid, term = line.split("\t", 1)
        wnids.append(wnid)
        terms.append(tuple(word.strip() for word in term.split(',')))
    return pd.DataFrame(data={"terms":terms}, index=wnids)

