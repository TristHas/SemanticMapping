#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial import cKDTree
from Helpers import get_Sensembed_A_labelmap, normalize


class NNValidator(object):
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
