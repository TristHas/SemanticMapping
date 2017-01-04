#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
sys.path.append("..")

from scipy.spatial import cKDTree

import numpy as np
import pandas as pd
import hickle as hkl

from DataProcessing.DSAssemble.ILSVRC_Semantic import load_Sensembed_A_labelmap, load_Sensembed_B_labelmap
from DataProcessing.DSAssemble.ILSVRC_preprocessing import label_map_2012
from DataProcessing.Linking.Linker import merge_wn_sembed

# Really need some refactoring, there is no reason for those function to be here
# Plus, Need a scheme to share the labels between validator and featuremapper.
# Should merge the two? Need some UML drawing here.
# Feature Mapper only exists in the ThreadLoader but Validator accessed by Trainer
# Rethink architecture when need new functionalities

def get_Sensembed_A_labelmap():
    imnet_sembed   = merge_wn_sembed()
    imnet_labels   = load_Sensembed_A_labelmap()
    return imnet_labels.merge(imnet_sembed)

def make_Random_labelmap(n_class, sample_dim):
    data = np.random.randn(n_class, sample_dim + 5)
    df = pd.DataFrame(columns=["BN", "POS", "WNID", "gp", "LABEL"] + range(sample_dim), data = data)
    df["LABEL"] = range(n_class)
    pd.to_pickle(df, "/home/tristan/data/Imagenet/datasets/Sensembed_A/random_{}".format(sample_dim))

def get_Random_labelmap(sample_dim=400):
    return pd.load_pickle("/home/tristan/data/Imagenet/datasets/Sensembed_A/random_{}".format(sample_dim))

def normalize(vectors):
    """
        Normalize a set of vectors along the dimension 1 of their tensor representation
    """
    norms = np.expand_dims(np.linalg.norm(vectors,axis=1),1)
    return vectors / norms

class Validator(object):
    def __init__(self, labels = "smbd", distance = "cosine"):
        """
        """
        assert distance in ["cosine", "euclidean"]
        assert labels in ["smbd", "random"]
        self.distance = distance
        if labels == "smbd":
            index_map = get_Sensembed_A_labelmap()
        else:
            raise Exception("Not supposed to use random yet. Random labels must be shared by both Validator and FeatureMapper so need a better scheme")

        data = index_map.sort_values(by="LABEL").drop(["BN", "POS", "WNID", "gp", "LABEL"], axis=1)
        data = data.get_values().astype("float32")
        if distance == "cosine":
            data = normalize(data)
        self.tree = cKDTree(data)

    def smd_k_best_pred(self, predictions, k=5):
        """
            Returns the indices of the k nearest neighbours for each input predictions
        """
        if self.distance == "cosine":
            predictions = normalize(predictions)
        return self.tree.query(predictions, k)[1]

    def smbd_top_k_scores(self, predictions, labels, k=5):
        """
            Returns the mean top-k score for the input predictions.
        """
        k_nearest = self.smd_k_best_pred(predictions, k)
        matches = np.equal(k_nearest, np.tile(labels, (k,1)).swapaxes(0,1))
        return np.any(matches, axis=1).mean()

    def clas_k_best_pred(self, predictions, k):
        """
            DOC
        """
        return np.argpartition(-predictions, k, axis=1)[:,:k]

    def clas_top_k_scores(self, predictions, labels, k = 5):
        """
            DOC
        """
        k_nearest = self.clas_k_best_pred(predictions, k)
        matches = np.equal(k_nearest, np.tile(labels, (k,1)).swapaxes(0,1))
        return np.any(matches, axis=1).mean()


def test_myfunc(x,y,z):
    # Load dists and sensembed shared variables
    index_map = get_Sensembed_A_labelmap()
    data = index_map.sort_values(by="LABEL").drop(["BN", "POS", "WNID", "gp", "LABEL"], axis=1)
    data = data.get_values().astype("float32")
    smbds = theano.shared(np.tile(data,(256,1,1)).swapaxes(1,2))
    dist = np.zeros((907,907))
    for i in range(data.shape[0]):
        dist[i] = np.square(data - data[i]).mean(axis=1)
    dist = theano.shared(dist)

    pred        = T.matrix(dtype="float32")
    labels      = T.ivector()
    pred_dists  = T.square(pred.dimshuffle(0,1,'x') - smbds).mean(axis=1)
    batch_dists = dists[labels]
    loss        = T.mean(pred_dists - batch_dists)
    f           = theano.function(inputs=[pred, labels], outputs=[pred_dists, batch_dists, error])
