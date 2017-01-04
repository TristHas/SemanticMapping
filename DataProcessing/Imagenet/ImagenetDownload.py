#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys, tarfile
from multiprocessing import Process, Manager
import yaml
from requests import ConnectionError
from ImagenetMetadata import load_syns_metadata
from ..util.Helpers import Logger, open_remote
log = Logger()


root_paths_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "..",
                                "paths.yaml")
with open(root_paths_file, "r") as f:
    photo_dir = yaml.load(f)["default_photo_dir"]


# Imagenet API
uid                 = 'tristan'
access_key          = 'c4fb5f6a0b2a85ef61cd205af6962af58bc8a096'
#wnid                = 'n02084071'

def wnid_tar_url(wnid,id=uid, key=access_key):
    """
        Returns the URL of the tar archive for the pictures of wnid
    """
    return 'http://www.image-net.org/download/synset?wnid={}&username={}&accesskey={}&release=latest&src=stanford'.format(wnid, id, key)

def dwn_wnid_photos(wnid, photo_dir = photo_dir):
    """
        Download the pictures of synset wnid folder photo_dir/wnid.
    """
    try:
        url  = wnid_tar_url(wnid)
        tar_obj = tarfile.open(fileobj=open_remote(url))
    except Exception as e:
        log.error("Errir downloading {}: {}:{}".format(wnid, e, e.message))
        return False
    try:
        wnid_photo_dir = os.path.join(photo_dir, wnid)
        if not os.path.isdir(os.path.dirname(wnid_photo_dir)):
            os.makedirs(wnid_photo_dir)
        tar_obj.extractall(wnid_photo_dir)
    except Exception as e:
        log.debug("Failed Tar extraction: {}".format(e))
        return False
    return True

def wnid_tar_files(wnid):
    """
        Returns a list of file-like objects containing the photos of synset wnid
    """
    url  = wnid_tar_url(wnid)
    tar_obj = tarfile.open(fileobj=open_remote(url))
    photos = []
    for member in tar_obj.getmembers():
         photos.append(tar_obj.extractfile(member))
    return photos

def downloaded_syns(photo_dir = photo_dir):
    return os.listdir(photo_dir)

def dwn_all_imnet(photo_dir = photo_dir):
    to_dwn = load_syns_metadata()
    to_dwn = to_dwn[to_dwn.numImage.notnull()]
    log.debug("{} synset with images".format(len(to_dwn)))
    downloaded = downloaded_syns(photo_dir)
    log.info("Download skipping existing {} folders. Laste five of them are {}".format(len(downloaded), downloaded[-5:]))
    to_dwn = to_dwn.drop(downloaded)
    to_dwn = to_dwn.sort(columns="numImage")
    log.debug("Remaine {} synset with images to download".format(len(to_dwn)))
    for wnid in to_dwn.index:
        log.info("Downloading {}".format(wnid))
        failed_attempt = 0
        while not dwn_wnid_photos(wnid, photo_dir):
            failed_attempt += 1
            log.debug("Attempt to download this synset failed for the {}th time".format(failed_attempt))
        log.debug("Downloaded {} photos into {}".format(
                                            len(os.listdir(os.path.join(photo_dir, wnid))),
                                            os.path.join(photo_dir, wnid)
                                            )
                 )


def dwn_all_imnet_multiprocess(photo_dir = photo_dir, nprocess = 2):
    to_dwn = load_syns_metadata()
    to_dwn = to_dwn[to_dwn.numImage.notnull()]
    log.debug("{} synset with images".format(len(to_dwn)))
    downloaded = downloaded_syns(photo_dir)
    log.info("Download skipping existing {} folders. Laste five of them are {}".format(len(downloaded), downloaded[-5:]))
    to_dwn = to_dwn.drop(downloaded)
    to_dwn = to_dwn.sort(columns="numImage")
    log.debug("Remain {} synset with images to download".format(len(to_dwn)))
    with Manager() as manager:
        dir   = manager.list([photo_dir])
        wnids = manager.list(to_dwn.index.get_values().tolist())
        processes = []
        for i in range(nprocess):
            processes.append(Process(target=download_process_method, args = (dir, wnids), name="p{}".format(i)))
        for p in processes:
            p.start()
        for p in processes:
            p.join()

def download_process_method(dir, wnids):
    photo_dir = dir[0]
    log.info("Process {} started. Downloading to {}".format(os.getpid(), photo_dir))
    while len(wnids) > 0:
        wnid = wnids.pop(0)
        log.info("Process {} downloading {}".format(os.getpid(), wnid))
        failed_attempt = 0
        while not dwn_wnid_photos(wnid, photo_dir):
            failed_attempt += 1
            log.debug("Attempt to download this synset failed for the {}th time".format(failed_attempt))
        log.debug("Process {} downloaded {} photos into {}".format(
                                            os.getpid(),
                                            len(os.listdir(os.path.join(photo_dir, wnid))),
                                            os.path.join(photo_dir, wnid)
                                            )
                    )


