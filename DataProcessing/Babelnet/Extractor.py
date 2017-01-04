#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, yaml
import pandas as pd

root_paths_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "..",
                                "paths.yaml")
module_paths_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "paths.yaml")


with open(root_paths_file, "r") as f_root:
    root_path = yaml.load(f_root)
    root = root_path["babelnet_root"]

def extract_wn30_wn31_mappings():
    with open(module_paths_file, "r") as f_module:
        module_paths = yaml.load(f_module)
        f_dbp = module_paths["raw_dbpuris"]
        df_dbp = module_paths["df_dbpuris"]
