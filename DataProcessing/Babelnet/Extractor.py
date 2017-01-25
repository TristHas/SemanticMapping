#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, yaml
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

root_paths_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "..",
                                "paths.yaml")
module_paths_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "paths.yaml")

with open(root_paths_file, "r") as f_root:
    root_path  = yaml.load(f_root)
    root       = root_path["babelnet_root"]
    store_path = root_path["babelnet_store"]
    categories = root_path["babelnet_categories"]
    graph      = root_path["babelnet_graph"]

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
    f_type  = os.path.join(root, module_paths["raw_type"])
    df_type = os.path.join(root, module_paths["df_type"])
    f_wnoffsets  = os.path.join(root, module_paths["raw_wnoffsets"])
    df_wnoffsets = os.path.join(root, module_paths["df_wnoffsets"])
    f_yago  = os.path.join(root, module_paths["raw_yagouris"])
    df_yago = os.path.join(root, module_paths["df_yagouris"])

def split_bn(df):
    df["POS"]   = df.bnid.apply(lambda x:x.split(":")[1][-1])
    df["BN"]    = df.bnid.apply(lambda x:x.split(":")[1][:-1])
    del df["bnid"]

def split_target(df):
    df["target_POS"]   = df.bnid.apply(lambda x:x.split(":")[1][-1])
    df["target_BN"]    = df.bnid.apply(lambda x:x.split(":")[1][:-1])
    del df["target"]

def split_categories(df):
    df["lang"]=df["category"].apply(lambda x:x.split(":")[1])
    df["cattype"]=df["category"].apply(lambda x:x.split(":")[0])
    df["category"]=df["category"].apply(lambda x:x.split(":")[2])

def extract_all():
    extract_type()
    extract_domains()
    extract_wnoffsets()
    extract_dbp()
    extract_yago()
    extract_compounds()
    extract_otherForms()
    extract_images()
    extract_graph()
    extract_categories()
    extract_synsets()

def extract_type():
    """
        Works for type, domain, wnoffset
    """
    df = pd.read_table(f_type, delimiter=",", skipinitialspace=True)
    split_bn(df)
    df.to_pickle(df_type)
    print "Type extracted from {} to {}".format(f_type, df_type)

def extract_domains():
    """
        Works for type, domain, wnoffset
    """
    df = pd.read_table(f_domains, delimiter=",", skipinitialspace=True)
    split_bn(df)
    df.to_pickle(df_domains)
    print "domains extracted from {} to {}".format(f_domains, df_domains)

def extract_wnoffsets():
    """
        Works for type, domain, wnoffset
    """
    df = pd.read_table(f_wnoffsets, delimiter=",", skipinitialspace=True)
    split_bn(df)
    df.to_pickle(df_wnoffsets)
    print "wnoffsets extracted from {} to {}".format(f_wnoffsets, df_wnoffsets)

def extract_synsets():
    f_synsets = "/home/tristan/data/Babelnet/java_extracted/synsets"
    df = pd.read_table(f_synsets, names=["bnid"], skiprows=1,delimiter=",", skipinitialspace=True)
    df = df.reset_index().rename(columns={"index":"bnID"})
    df.to_pickle("/media/tristan/b2e18d6c-6e39-4556-9ed7-032d4b0de1a5/Babelnet/processed_data/synsets")

def extract_dbp():
    """
        Handles extra separator (coma) in the target column
    """
    df = pd.read_table(f_dbp, names=["bnid", "dbp", "extra"], skiprows=1,delimiter=",", skipinitialspace=True)
    splitted = df[df.extra.notnull()]
    df.loc[splitted.index.get_values(), "dbp"]= splitted["dbp"]+"," + splitted["extra"]
    del df["extra"]
    split_bn(df)
    df.to_pickle(df_dbp)
    print "dbp extracted from {} to {}".format(f_dbp, df_dbp)

def extract_yago():
    """
        Handles extra separator (coma) in the target column
    """
    df = pd.read_table(f_yago, names=["bnid", "yago", "extra"], skiprows=1,delimiter=",", skipinitialspace=True)
    splitted = df[df.extra.notnull()]
    df.loc[splitted.index.get_values(), "yago"]= splitted["yago"]+"," + splitted["extra"]
    del df["extra"]
    split_bn(df)
    df.to_pickle(df_yago)
    print "yago extracted from {} to {}".format(f_yago, df_yago)

def extract_compounds():
    """
    """
    df = pd.read_table(f_compounds, names=["bnid", "lang", "compound"], skiprows=1,delimiter=",", skipinitialspace=True)
    split_bn(df)
    df.to_pickle(df_compounds)
    print "compounds extracted from {} to {}".format(f_compounds, df_compounds)

def extract_otherForms():
    """
    """
    df = pd.read_table(f_otherForms, names=["bnid", "lang", "otherForms"], skiprows=1,delimiter=",", skipinitialspace=True)
    split_bn(df)
    df.to_pickle(df_otherForms)
    print "otherForms extracted from {} to {}".format(f_otherForms, df_otherForms)

def extract_images():
    """
    """
    df = pd.read_table(f_images, names=["bnid", "html", "extra"], skiprows=1,delimiter=",", skipinitialspace=True)
    df["link"]=df.html.apply(lambda x:x.split('"')[1])
    df["cond"]=df["link"].apply(lambda x: x.lower().startswith("http"))
    df=df.drop(df[df.cond==False].index)
    split_bn(df)
    df.to_pickle(df_images)
    print "images extracted from {} to {}".format(f_images, df_images)

def extract_categories():
    with pd.get_store(store_path) as store:
        if categories in store:
            del store[categories]
        offset = 0
        chunk_iterator = pd.read_table( f_categories, delimiter=",", skipinitialspace=True,
                                        names=["bnid", "category", "extra"], skiprows=1,
                                        iterator = True, chunksize=500000)
        for df in chunk_iterator:
            new_offset      = offset + len(df)
            print new_offset
            df.index        = range(offset, new_offset)
            offset=new_offset
            splitted        = df[df.extra.notnull()]
            df.loc[splitted.index.get_values(), "category"]= splitted["category"]+"," + splitted["extra"]
            del df["extra"]
            split_categories(df)
            split_bn(df)
            store.append(categories, df, min_itemsize={"category":600}, data_columns = ["BN", "POS"])
    print "Categories extracted from {} to {}".format(f_categories, store_path)



