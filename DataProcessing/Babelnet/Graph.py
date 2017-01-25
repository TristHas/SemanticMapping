#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, yaml
import pandas as pd
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

def get_all_synsets():
    return pd.read_pickle("/media/tristan/b2e18d6c-6e39-4556-9ed7-032d4b0de1a5/Babelnet/processed_data/synsets")

def get_all_targets():
    return pd.read_pickle("/media/tristan/b2e18d6c-6e39-4556-9ed7-032d4b0de1a5/Babelnet/processed_data/targets")

def get_all_langs():
    return pd.read_pickle("/media/tristan/b2e18d6c-6e39-4556-9ed7-032d4b0de1a5/Babelnet/processed_data/langs")

def get_all_pointers():
    return pd.read_pickle("/media/tristan/b2e18d6c-6e39-4556-9ed7-032d4b0de1a5/Babelnet/processed_data/pointers")

def get_dimension_from_edge(df):
    df=df.reset_index()
    df=df.merge(pointers, how="left").merge(langs, how="left").merge(synsets, how="left").merge(synsets.rename(columns={"bnID":"targetID", "bnid":"target"}), how="left")
    df["coord"] = df["langID"] + df["pointerID"] * len(langs) + df["targetID"] * len(langs) * len(pointers)
    df.set_index("index")
    df = df[["bnID","coord"]]
    return df

def get_edge_from_dimension(df):
    df = df.copy().reset_index()
    df["langID"] = df["coord"] % len(langs)
    df["pointerID"] = ((df["coord"] - df["langID"]) / len(langs)) % len(pointers)
    df["targetID"] = (df["coord"] - df["langID"] - df["pointerID"]*len(langs)) / (len(langs) * len(pointers))
    df=df.merge(pointers, how="left").merge(langs, how="left").merge(synsets, how="left").merge(synsets.rename(columns={"bnID":"targetID", "bnid":"target"}), how="left")
    assert all(df["coord"] == df["langID"] + df["pointerID"] * len(langs) + df["targetID"] * len(langs) * len(pointers))
    df.set_index("index")
    return df[["bnid","lang","target","pointer"]]

def extract_lang_pointers_targets():
    chunk_iterator = pd.read_table( f_edges, delimiter=",", skipinitialspace=True,
                                    iterator = True, chunksize=10000000)
    targets = set()
    pointers = set()
    lang = set()
    for df in chunk_iterator:
        pointers = set.union(pointers, df.pointer)
        lang = set.union(lang, df.lang)
        targets = set.union(lang, df.lang)
    langdf = pd.DataFrame(data={"pointer":list(poiters), "pointerID":range(len(poiters))})
    langdf.to_pickle("/media/tristan/b2e18d6c-6e39-4556-9ed7-032d4b0de1a5/Babelnet/processed_data/langs")
    pointerdf = pd.DataFrame(data={"lang":list(lang), "langID":range(len(lang))})
    pointerdf.to_pickle("/media/tristan/b2e18d6c-6e39-4556-9ed7-032d4b0de1a5/Babelnet/processed_data/pointers")
    pointerdf = pd.DataFrame(data={"target":list(targets), "targetID":range(len(targets))})
    pointerdf.to_pickle("/media/tristan/b2e18d6c-6e39-4556-9ed7-032d4b0de1a5/Babelnet/processed_data/targets")

def extract_graph_to_table():
    """
    """
    def get_dimension_from_edge(df):
        df=df.reset_index()
        df=df.merge(pointers, how="left").merge(langs, how="left").merge(synsets, how="left").merge(synsets.rename(columns={"bnID":"targetID", "bnid":"target"}), how="left")
        df["coord"] = df["langID"] + df["pointerID"] * len(langs) + df["targetID"] * len(langs) * len(pointers)
        df.set_index("index")
        df = df[["bnID","coord"]]
        return df
    pointers = get_all_pointers()
    langs = get_all_langs()
    synsets = get_all_synsets()
    with pd.get_store(store_path) as store:
        if graph in store:
            del store[graph]
        chunk_iterator = pd.read_table( f_edges, delimiter=",", skipinitialspace=True,
                                        iterator = True, chunksize=10000000)
        offset=0
        for chunk in chunk_iterator:
            new_offset      = offset + len(chunk)
            print new_offset
            coord = get_dimension_from_edge(chunk).astype("int64")
            coord.index = range(offset, new_offset)
            offset=new_offset
            store.append(graph, coord, data_columns = ["bnID"])

def get_graph_subset(bnID_list):
    """
        Selects the coordinates from the graph relative to the synsets in bnid_list.
        bnID must be given in the format bn:XXXXXXn
        Due to a weird behavior of select(where = "column == a=list_of_val"),
        we split the process in chunks of 30.
        Problem described in stack overflow
    """
    chunksize = 30
    df    = pd.DataFrame(data={"bnid":bnID_list})
    df    = df.merge(get_all_synsets())
    bnids = df.bnID.tolist()
    result = []
    with pd.get_store(store_path) as store:
        for i in range(len(bnids) / chunksize):
            chunk = bnids[chunksize * i: chunksize * (i+1)]
            result.append(store.select(graph, where=pd.Term('bnID','=',chunk)))
        chunk = bnids[chunksize * (len(bnids) / chunksize):]
        result.append(store.select(graph, where=pd.Term('bnID','=',chunk)))
    return pd.concat(result)

def unique_table(input, old, new):
    """
        This function takes as input a list of values and associate each
        different value a unique 0-based index. The result is returned as
        a two-columns dataframe
        Parameters:
            input = list of values to index
            old   = column name for the unique values of input
            new   = column name for the zero-based index
        Output:
            DataFrame with two columns old and new
    """
    old_list = list(set(input))
    new_list = range(len(old_list))
    return pd.DataFrame(data={old:old_list, new: new_list})

def graph_to_sparse(df):
    """
    """
    return csr_matrix((np.ones(len(df), dtype="uint8"), (df.bnindex, df.coordindex)))

def convert_coordinates(df):
    coord_table = unique_table(df.coord.tolist(), "coord", "coordindex")
    bnid_table  = unique_table(df.bnID.tolist(), "bnID", "bnindex")
    table = df.merge(coord_table).merge(bnid_table)[["bnindex","coordindex"]]
    return table, coord_table, bnid_table


