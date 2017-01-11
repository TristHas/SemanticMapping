#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, time
from threading import Thread
import logging
import StringIO
import requests
import numpy as np
import pandas as pd

logfile = "/media/tristan/260726b1-f8fa-4740-8707-2dd2b1d197fd/default.log"

class Logger():
    def __init__(self):
        l = logging.Logger(__name__)
        fh = logging.FileHandler(logfile)
        fh.setLevel(logging.DEBUG)
        th = logging.StreamHandler()
        th.setLevel(logging.ERROR)
        formatter = logging.Formatter('%(created)f:%(levelname)s:%(filename)s:line %(lineno)s - %(message)s')
        fh.setFormatter(formatter)
        th.setFormatter(formatter)
        l.addHandler(fh)
        l.addHandler(th)

        self.error = l.error
        self.warn  = l.warning
        self.debug = l.debug
        self.info  = l.info
        self.exception = l.exception
        self.critical = l.critical
        self.log = l.log

def open_remote(url):
    """
        Returns an open file-like object from the url
    """
    data = requests.get(url)
    return StringIO.StringIO(data.content)

def download(url, path):
    response = requests.get(url)
    with open(path,"wb") as f:
        f.write(response.content)

log = Logger()
def check_file_path(path):
    """
        Creates directory hierarchy for a given file path if this hierarchy does not exist
        Logs a warning if the given file path is an existing file.
    """
    dir = os.path.dirname(path)
    if not os.path.isdir(dir):
        os.makedirs(dir)
        log.debug("Created dir {}".format(dir))
    if os.path.isfile(path):
        log.warn("CAREFUL! {} file already exists".format(path))
