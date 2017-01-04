#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os
sys.path.append("..")

import numpy as np
import pandas as pd
import hickle as hkl
from util.Helpers import Logger

from Helpers import BatchIterator, Validator
from model_loaders import compile_model_proba

log = Logger()
log.info("All imports done. logging")

def validate_ResNet50_ILSVRC(ds_path="/home/tristan/data/Imagenet/datasets/ILSVRC2015/256_224/center_crop/"):
    mean, pred_func = compile_model_proba()
    bi = BatchIterator(ds_path, smbd=False, input_type="input", mean=mean, clas_mode = "caffe2012")
    v = Validator(smbd=False)
    i, score = 0,0
    for x,y in bi.epoch_clas_valid():
        pred = pred_func(x)
        score += v.clas_top_k_scores(pred, y)
        i += 1
        log.debug("{}th batch. Average score= {}%".format(i, 100 * score/i))

