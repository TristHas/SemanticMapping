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
    extract_domain()
    extract_wnoffsets()
    extract_dbp()
    extract_yago()
    extract_compounds()
    extract_otherForms()
    extract_images()
    #extract_graph()
    #extract_categories()


def extract_type():
    """
        Works for type, domain, wnoffset
    """
    df = pd.read_table(f_type, delimiter=",", skipinitialspace=True)
    split_bn(df)
    df.to_pickle(df_type)

def extract_domain():
    """
        Works for type, domain, wnoffset
    """
    df = pd.read_table(f_domain, delimiter=",", skipinitialspace=True)
    split_bn(df)
    df.to_pickle(df_domain)

def extract_wnoffsets():
    """
        Works for type, domain, wnoffset
    """
    df = pd.read_table(f_wnoffsets, delimiter=",", skipinitialspace=True)
    split_bn(df)
    df.to_pickle(df_wnoffsets)

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

def extract_yago():
    """
        Handles extra separator (coma) in the target column
    """
    df = pd.read_table(f_dbp, names=["bnid", "yago", "extra"], skiprows=1,delimiter=",", skipinitialspace=True)
    splitted = df[df.extra.notnull()]
    df.loc[splitted.index.get_values(), "yago"]= splitted["yago"]+"," + splitted["extra"]
    del df["extra"]
    split_bn(df)
    df.to_pickle(df_yago)

def extract_compounds():
    """
    """
    df = pd.read_table(f_compounds, names=["bnid", "lang", "compound"], skiprows=1,delimiter=",", skipinitialspace=True)
    split_bn(df)
    df.to_pickle(df_compounds)

def extract_otherForms():
    """
    """
    df = pd.read_table(f_otherForms, names=["bnid", "lang", "otherForms"], skiprows=1,delimiter=",", skipinitialspace=True)
    split_bn(df)
    df.to_pickle(df_otherForms)

def extract_images():
    """
    """
    df = pd.read_table(f_images, names=["bnid", "html", "extra"], skiprows=1,delimiter=",", skipinitialspace=True)
    df["link"]=df.dbp.apply(lambda x:x.split('"')[1])
    df["cond"]=df["link"].apply(lambda x: x.lower().startswith("http"))
    df=df.drop(df[df.cond==False].index)
    split_bn(df)
    df.to_pickle(df_images)

def extract_categories():
    with pd.get_store(store_path) as store:
        # Delete hdfs files if already existing
        if categories in store:
            del store[categories]
            #log.warn("Deleting {} from the store".format(sense_vectors))
        # Offsets are needed for vectors to have unique index
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
            store.append(categories, df, min_itemsize={"category":500}, data_columns = ["BN", "POS"])

def extract_graph():
    """
    """
    with pd.get_store(store_path) as store:
        # Delete hdfs files if already existing
        if graph in store:
            del store[graph]
            #log.warn("Deleting {} from the store".format(sense_vectors))
        # Offsets are needed for vectors to have unique index
        offset = 0
        chunk_iterator = pd.read_table( f_edges, delimiter=",", skipinitialspace=True,
                                        iterator = True, chunksize=10000000)
        for df in chunk_iterator:
            new_offset      = offset + len(df)
            df.index        = range(offset, new_offset)
            offset          = new_offset
            split_target(df)
            split_bn(df)
            store.append(graph, df, min_itemsize={"pointer":100}, data_columns = ["BN", "POS"])


def extract_sparse_semantic_network_representation():
    with pd.get_store(store_path) as store:
        store.select_col


#print "Extracting Graph"
#extract_graph()
#print "Exctracting categories"
#extract_categories()
