#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, yaml
import pandas as pd
from ..Babelnet.Sensembed import select_synset_vectors
from Linker import load_wn30_bn

root_paths_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "..",
                                "paths.yaml")
module_paths_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "paths.yaml")


with open(root_paths_file, "r") as f_root:
    root_path = yaml.load(f_root)
    root = root_path["linking_root"]


def extract_wn30_wn31_mappings():
    with open(module_paths_file, 'r') as f_linker:
        link_path = yaml.load(f_linker)
        f_wn30_wn31   = os.path.join(root,link_path["file_map_wn30_wn31"])
        df_wn30_wn31   = os.path.join(root,link_path["file_wn30_wn31"])
    raw_map   = pd.read_table(f_wn30_wn31)
    raw_map   = raw_map[raw_map["#PoS"]=="n"]
    new_map   = pd.DataFrame(data={"WN30": raw_map['WordNet 3.0'].apply(lambda x: "n"+str(x).rjust(8, '0')),
                                   "WN31": raw_map['WordNet 3.1'].apply(lambda x: "n"+str(x).rjust(8, '0'))})
    new_map.to_pickle(df_wn30_wn31)

def extract_wn30_bn_mappings():
    with open(module_paths_file, 'r') as f_linker:
        link_path = yaml.load(f_linker)
        f_wn31_bn   = os.path.join(root,link_path["file_map_wn30_bn"])
        df_wn31_bn   = os.path.join(root,link_path["file_wn30_bn"])
    df = pd.read_csv(f_wn31_bn)
    df.BNID = df.BNID.apply(lambda x: x.split(":")[1][:-1])
    df.to_pickle(df_wn31_bn)

def extract_image_sensembed():
    with open(root_paths_file, "r") as f_root:
        root_path = yaml.load(f_root)
        store_path = root_path["sensembed_store"]
        imsynsets_vectors = root_path["sensembed_imsynsets"]
    df = load_wn30_bn().rename(columns={"BNID":"BN"})
    df = select_synset_vectors(df)
    with pd.get_store(store_path) as store:
            if imsynsets_vectors in store:
                del store[imsynsets_vectors]
            store.append(imsynsets_vectors, df, data_columns = ["BN", "POS"])
