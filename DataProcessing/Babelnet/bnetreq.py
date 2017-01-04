#!/usr/bin/env python
# -*- coding: utf-8 -*-

from Linker import load_in_wn31_mapped
import pandas as pd

wn_bn_dbp_file = "/home/tristan/data/wnid_bnid_dbpres"
wn_bn_file      ="/home/tristan/data/wnid_bnid"

in_data = load_in_wn31_mapped()
wn_bn_data = pd.read_pickle(wn_bn_file)
wn_bn_dbp_data = pd.read_pickle(wn_bn_dbp_file)
