#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, yaml
import pandas as pd

root_paths_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "..",
                                "paths.yaml")
module_paths_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "paths.yaml")

# Edges (21G), Categories (11G) will probably need to be iterated

with open(root_paths_file, "r") as f_root:
    root_path = yaml.load(f_root)
    root = root_path["babelnet_root"]

with open(module_paths_file, "r") as f_module:
    module_paths = yaml.load(f_module)
    f_dbp  = os.path.join(root, module_paths["raw_dbpuris"])
    df_dbp = os.path.join(root, module_paths["df_dbpuris"])
    f_compounds  = os.path.join(root, module_paths["raw_compounds"])
    df_compounds = os.path.join(root, module_paths["df_compounds"])
    f_categories  = os.path.join(root, module_paths["raw_categories"])
    df_categories = os.path.join(root, module_paths["df_categories"])
    f_domains  = os.path.join(root, module_paths["raw_domains"])
    df_domains = os.path.join(root, module_paths["df_domains"])
    f_edges  = os.path.join(root, module_paths["raw_edges"])
    df_edges = os.path.join(root, module_paths["df_edges"])
    f_images  = os.path.join(root, module_paths["raw_images"])
    df_images = os.path.join(root, module_paths["df_images"])
    f_otherForms  = os.path.join(root, module_paths["raw_otherForms"])
    df_otherForms = os.path.join(root, module_paths["df_otherForms"])
    f_test  = os.path.join(root, module_paths["raw_test"])
    df_test = os.path.join(root, module_paths["df_test"])
    f_type  = os.path.join(root, module_paths["raw_type"])
    df_type = os.path.join(root, module_paths["df_type"])
    f_wnoffsets  = os.path.join(root, module_paths["raw_wnoffsets"])
    df_wnoffsets = os.path.join(root, module_paths["df_wnoffsets"])
    f_yago  = os.path.join(root, module_paths["raw_yagouris"])
    df_yago = os.path.join(root, module_paths["df_yagouris"])

def split_bn(df):
    df["POS"]   = df.bnid.apply(lambda x:x.split(":")[1][-1])
    df["BN"]    = df.bnid.apply(lambda x:x.split(":")[1][:-1])

def read_raw_frame(fin):
    """
        Works for type, domain
    """
    df = pd.read_table(fin, delimiter=",", skipinitialspace=True)
    split_bn(df)
    return df

def read_dbp():
    df = pd.read_table(f_dbp, names=["bnid", "dbp", "extra"], skiprows=1,delimiter=",", skipinitialspace=True)
    df.loc[x.index.get_values(), "dbp"]=x["dbp"]+","+x["extra"]
    df.drop("extra", axis=1)
    split_bn(df)
    return df

def read_compounds():
    df = pd.read_table(f_compounds, names=["bnid", "lang", "compound"], skiprows=1,delimiter=",", skipinitialspace=True)
    split_bn(df)
    return df



