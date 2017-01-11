#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os, sys
import yaml
import numpy as np
import pandas as pd
pd.set_option('io.hdf.default_format','table')

from ..util.Helpers import Logger

path_file = 'paths.yaml'
log = Logger()


root_paths_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "..",
                                "paths.yaml")
module_paths_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "paths.yaml")


with open(root_paths_file, "r") as f_root:
    root_path = yaml.load(f_root)
    root = root_path["babelnet_root"]
    store_path = root_path["sensembed_store"]
    word_vectors = root_path["sensembed_words"]
    sense_vectors = root_path["sensembed_senses"]
    synset_vectors = root_path["sensembed_synsets"]


def extract_raw_vectors(chunksize = 500000):
    """
        DOC
    """
    with open(module_paths_file, 'r') as f:
        paths = yaml.load(f)
        raw_path = os.path.join(root, paths["sensembed_raw_file"])

    with pd.get_store(store_path) as store:
        # Delete hdfs files if already existing
        if sense_vectors in store:
            del store[sense_vectors]
            log.warn("Deleting {} from the store".format(sense_vectors))
        if word_vectors in store:
            del store[word_vectors]
            log.warn("Deleting {} from the store".format(word_vectors))
        # Offsets are needed for vectors to have unique index
        word_offset, sense_offset = 0,0
        chunk_iterator = pd.read_table( raw_path,
                                        names = ["vec_key"] + range(400),
                                        chunksize=chunksize,
                                        delim_whitespace=True,
                                        iterator = True)
        for chunk in chunk_iterator:
            gps = chunk.groupby(chunk["vec_key"].apply(lambda x: len(str(x).split(":"))))
            word_offset  = serialize_word_chunk(store, word_vectors,
                                                gps.get_group(1), word_offset)
            sense_offset =  serialize_sense_chunk(store, sense_vectors,
                                                 gps.get_group(2), sense_offset)

    log.info("Raw sensembed vector extraction complete. Extracted {} words and {} senses".format(word_offset, sense_offset))
    print "Raw sensembed vector extraction complete. Extracted {} words and {} senses".format(word_offset, sense_offset)


def serialize_word_chunk(store, table_name, chunk, offset):
    new_offset = offset + len(chunk)
    log.debug("Word chunk covers index from {} to {}".format(offset, new_offset -1))
    chunk.index     = range(offset, new_offset)
    chunk.columns   = ["word"] + range(400)
    store.append(table_name, chunk, min_itemsize={"word":100}, data_columns = ["word"])
    return new_offset

def serialize_sense_chunk(store, table_name, chunk, offset):
    new_offset = offset + len(chunk)
    log.debug("Sense chunk ({}) covers index from {} to {}".format(table_name, offset, new_offset - 1))
    chunk.index    = range(offset, new_offset)
    chunk["POS"]    = chunk.vec_key.apply(lambda x:x.split(":")[1][-1])
    chunk["BN"]    = chunk.vec_key.apply(lambda x:x.split(":")[1][:-1])
    chunk["word"]  = chunk.vec_key.apply(lambda x:x.split(":")[0][:-3])
    del chunk["vec_key"]
    store.append(table_name, chunk, min_itemsize={"word":100}, data_columns = ["word", "BN", "POS"])
    return new_offset

# Synset sensembed vector computation
def compute_synset_vectors(ngroup = 20):
    tmp_table = "/tmp/bns_gp"
    tmp_column = "gp"
    with pd.get_store(store_path) as store:
        try:
            if tmp_table in store:
                del store[tmp_table]
                log.warn("Deleting {} from the store".format(tmp_table))
            if synset_vectors in store:
                del store[synset_vectors]
                log.warn("Deleting {} from the store".format(word_vectors))
            log.info("Hashing synsets into {} groups".format(ngroup))
            bns_gp = store.select_column(sense_vectors, "BN").apply(lambda x: hash(x) % ngroup)
            bns_gp.name = tmp_column
            log.info("Creating temporary table gp")
            store.append(tmp_table, bns_gp)
            log.info("Computing synset mean group by group...")
            offset = 0
            for i in range(ngroup):
                log.debug("Group {}".format(i))
                chunk = store.select_as_multiple([sense_vectors, tmp_table], where = "{} = {}".format(tmp_column, i), selector = tmp_table)
                synset_gps = chunk.groupby(["BN", "POS"])[range(400)].agg(np.mean)
                synset_gps = synset_gps.reset_index()
                indices = range(offset, offset + len(synset_gps))
                offset = indices[-1] + 1
                synset_gps.index = indices
                store.append(synset_vectors, synset_gps, data_columns = ["BN", "POS"])
                log.debug("synset indices ranging from {} to {}".format(synset_gps.index[0], synset_gps.index[-1]))
            log.info("Synset mean computed")
        finally:
            log.info("Deleting temp table")
            del store[tmp_table]

def select_synset_vectors(df, ngroup = 20):
    tmp_table = "/tmp/bn_gp"
    tmp_column = "gp"
    selected_synsets = None
    assert "BN" in df.columns
    log.info("{} synsets to select".format(len(df)))
    with pd.get_store(store_path) as store:
        try:
            if tmp_table in store:
                del store[tmp_table]
                log.warn("Deleting {} from the store".format(tmp_table))
            log.info("Hashing synsets into {} groups".format(ngroup))
            bns_gp = store.select_column(synset_vectors, "BN").apply(lambda x: hash(x) % ngroup)
            bns_gp.name = tmp_column
            log.info("Creating temporary table gp")
            store.append(tmp_table, bns_gp)
            log.info("Selecting relevant synset vectors...")
            for i in range(ngroup):
                log.debug("Group {}".format(i))
                chunk = store.select_as_multiple([synset_vectors, tmp_table], where = "{} = {}".format(tmp_column, i), selector = tmp_table)
                synset = chunk[chunk.POS == "n"]
                synset = synset[synset.BN.isin(df.BN)]
                print synset.BN
                log.debug("Found {} synsets to append".format(len(synset)))
                if selected_synsets is None:
                    selected_synsets = synset
                    log.debug("Initializing selection")
                else:
                    selected_synsets = selected_synsets.append(synset, ignore_index = True)
            log.info("{} synset selected".format(len(selected_synsets)))
        finally:
            log.info("Deleting temp table")
            del store[tmp_table]
            return selected_synsets
