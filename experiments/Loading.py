#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys, time
sys.path.append("..")
from threading import Thread


import hickle as hkl
import pandas as pd
import numpy as np

from Helpers import get_Sensembed_A_labelmap, get_Random_labelmap, normalize
from DataProcessing.util.Helpers import Logger

log = Logger()


class ThreadedLoader(object):
    def __init__(self,  ds_path = "/home/tristan/data/Imagenet/datasets/Sensembed_A/256_224/center_crop/train",
                        input_type = "features", input_mean = 0, input_unit_norm = True,
                        label_mapping = "Sensembed_A", label_unit_norm = True,
                        load_parallel = True, map_parallel = True,
                        queues_size=15, queues_timeout = 0.5):
        """
        """
        from Queue import Queue, Full, Empty
        self.memloader      = CPUMemLoader(ds_path=ds_path, input = input_type)
        self.nbatches       = self.memloader.n_batch
        self.mapper         = FeatureMapper(label_mapping = label_mapping, label_unit_norm = label_unit_norm,
                                            input_mean = input_mean, input_unit_norm = input_unit_norm)

        self.queues_timeout = queues_timeout
        self.load_queue = Queue(queues_size)
        self.load_running = False
        self.load_parallel = load_parallel
        self.map_queue = Queue(queues_size)
        self.map_running = False
        self.map_parallel = map_parallel
        self.run()

    def run(self):
        """
        """
        if self.load_parallel:
            self._run_loading()
        if self.map_parallel:
            self._run_mapping()

    def stop(self):
        """
        """
        if self.load_parallel:
            self._stop_loading()
        if self.map_parallel:
            self._stop_mapping()

    def next_batch(self):
        """
        """
        if self.load_parallel:
            if self.map_parallel:
                return self._get_mapped_data()
            else:
                x,z = self._get_loaded_data()
                if z is not None:
                    x = self.mapper.map_input(x)
                    y = self.mapper.map_label(z)
                    return x,y,z
                else:
                    return None, None, None
        else:
            x,z = self.memloader.next_batch()
            x   = self.mapper.map_input(x)
            y   = self.mapper.map_label(z)
            return x,y,z

    def epoch(self):
        """
        """
        return (self.next_batch() for i in range(self.memloader.n_batch))

    def _get_loaded_data(self):
        """
        """
        cond = True
        while cond and (self.load_running or (not self.load_queue.empty())):
            try:
                data = self.load_queue.get(timeout=self.queues_timeout)
                cond = False
                return data
            except Empty:
                pass
                #log.debug("Load queue get timeout")

        log.warn("get_loaded_data returns None")
        return None

    def _put_loaded_data(self, data):
        """
        """
        cond = True
        while cond and self.load_running:
            try:
                self.load_queue.put(data, timeout=self.queues_timeout)
                cond = False
            except Full:
                pass
                #log.debug("Load queue put timeout")

    def _get_mapped_data(self):
        """
        """
        cond = True
        while cond and (self.map_running or (not self.map_queue.empty())):
            try:
                data = self.map_queue.get(timeout=self.queues_timeout)
                cond = False
                return data
            except Empty:
                pass
                #log.debug("Map queue get timeout")
        log.warn("get_loaded_data returns None")
        return None

    def _put_mapped_data(self, data):
        """
        """
        cond = True
        while cond and self.map_running:
            try:
                self.map_queue.put(data, timeout=self.queues_timeout)
                cond = False
            except Full:
                pass
                #log.debug("Map queue put timeout")

    def _parallel_load_processing(self):
        """
        """
        while self.load_running:
            batch = self.memloader.next_batch()
            self._put_loaded_data(batch)

    def _parallel_map_processing(self):
        """
        """
        while self.map_running:
            x,z = self._get_loaded_data()
            x   = self.mapper.map_input(x)
            y   = self.mapper.map_label(z)
            self._put_mapped_data((x,y,z))

    def _run_loading(self):
        """
            Runs a parallel thread or process for loading batches from disk to memory
        """
        if not self.load_running:
            self.load_running = True
            self.loading_thread = Thread(target = self._parallel_load_processing, args=())
            self.loading_thread.start()
        else:
            log.warn("Parallel Loader asked to start loading while already loading")

    def _stop_loading(self):
        """
            Stops the parallel thread or process in charge of loading batches from disk to memory
        """

        if self.load_running:
            self.load_running = False
        else:
            log.warn("Parallel Loader asked to stop loading while not loading")

    def _run_mapping(self):
        """
            Runs a parallel thread or process for mapping labels to learning target
        """
        if not self.map_running:
            self.map_running = True
            self.mapping_thread = Thread(target = self._parallel_map_processing, args=())
            self.mapping_thread.start()
        else:
            log.warn("Parallel Mapper asked to start loading while already mapping")

    def _stop_mapping(self):
        """
            Stops the parallel thread or process in charge of mapping labels to learning target
        """
        if self.map_running:
            self.map_running = False
        else:
            log.warn("Parallel Loader asked to stop mapping while not mapping")


# Multiprocess Loader
from multiprocessing import Process, Value, Array, Queue
from Queue import Full, Empty
def get(queue, syncer, timeout = 0.5):
    """
    """
    while (syncer.value == 1) or (not queue.empty()):
        try:
            data = queue.get(timeout=timeout)
            return data
        except Empty:
            pass
            #log.info("get empty queue")
    log.warn("get returns None")
    return None

def put(queue, syncer, data, timeout = 0.5):
    """
    """
    cond = True
    while cond and (syncer.value == 1):
        try:
            batch = queue.put(data, timeout=timeout)
            cond = False
        except Full:
            #log.info("put full queue")
            pass

def parallel_load_processing(ds_path, load_queue, load_running, timeout = 0.1, input = "features", mean = 0):
    """
    """
    # Init memloader et load_running
    log = Logger()
    load_running.value = 1
    log.info("Parallel Load Process Starting")
    memloader = CPUMemLoader(ds_path, input, mean)
    while load_running.value == 1:
        batch = memloader.next_batch()
        #log.info("Load Putting")
        put(load_queue, load_running, batch, timeout)
    log.info("Parallel Load Process ending")

def parallel_map_processing(map_queue, load_queue, map_running, mapping_type, timeout = 0.1):
    """
    """
    log = Logger()
    map_running.value = 1
    log.info("Parallel Map Process Starting")
    mapper = LabelMapper(mapping_type)
    while map_running.value == 1:
        #log.info("Map getting")
        batch = get(load_queue, map_running, timeout)
        z   = mapper.map(batch[1])
        #log.info("Map putting")
        put(map_queue, map_running, (batch[0],batch[1],z), timeout)
    log.info("Parallel Map Process ending")


class MultiprocessedLoader(object):
    def __init__(self, ds_path, input = "features", mean = 0,
                mapping = "Sensembed_A",
                queues_size=15, queues_timeout = 0.5):
        """
        """
        self.load_queue = Queue(queues_size)
        self.load_running = Value('i', 0)
        self.map_queue = Queue(queues_size)
        self.map_running = Value('i', 0)
        self.mapping = mapping
        self.ds_path = ds_path
        self.input = input
        self.mean = mean
        self.timeout = queues_timeout
        self.run()
        self.n_batch = len(os.listdir(os.path.join(ds_path, "label")))

    def run(self):
        """
        """
        self._run_loading()
        self._run_mapping()
        time.sleep(1)

    def stop(self):
        """
        """
        self._stop_loading()
        self._stop_mapping()

    def next_batch(self):
        """
        """
        return get(self.map_queue, self.map_running)

    def epoch(self):
        """
        """
        return (self.next_batch() for i in range(self.n_batch))

    def _run_loading(self):
        """
            Runs a parallel thread or process for loading batches from disk to memory
        """
        if self.load_running.value == 0:
            self.loading_process = Process(target = parallel_load_processing, args=(self.ds_path, self.load_queue, self.load_running, self.timeout, self.input, self.mean))
            self.loading_process.start()
        else:
            log.warn("Parallel Loader asked to start loading while already loading")

    def _stop_loading(self):
        """
            Stops the parallel thread or process in charge of loading batches from disk to memory
        """

        if self.load_running.value == 1:
            self.load_running.value = 0
            time.sleep(2*self.timeout)
            self.loading_process.terminate()
            log.info("Parallel Load Process has terminated")
        else:
            log.warn("Parallel Loader asked to stop loading while not loading")

    def _run_mapping(self):
        """
            Runs a parallel thread or process for mapping labels to learning target
        """
        if self.map_running.value == 0:
            self.mapping_process = Process(target = parallel_map_processing, args=(self.map_queue, self.load_queue, self.map_running, self.mapping, self.timeout))
            self.mapping_process.start()
        else:
            log.warn("Parallel Mapper asked to start loading while already mapping")

    def _stop_mapping(self):
        """
            Stops the parallel thread or process in charge of mapping labels to learning target
        """
        if self.map_running.value == 1:
            self.map_running.value = 0
            time.sleep(2*self.timeout)
            self.mapping_process.terminate()
            log.info("Parallel Map Process has terminated")
        else:
            log.warn("Parallel Mapper asked to stop mapping while not mapping")


class CPUMemLoader(object):
    def __init__(self, ds_path, input = "features"):
        """
        """
        label_ids = os.listdir(os.path.join(ds_path, input))
        batch_ids = os.listdir(os.path.join(ds_path, "label"))
        assert set(batch_ids) == set(label_ids)
        self.ds_path    = ds_path
        self.input      = input
        self.batch_ids  = batch_ids
        self.batch_count= 0
        self.n_batch    = len(self.batch_ids)

    def get_batch(self, i):
        """
        """
        return (hkl.load(os.path.join(self.ds_path, self.input, i)).astype("float32"),
                hkl.load(os.path.join(self.ds_path, "label", i)).astype("uint"))


    def next_batch(self):
        """
        """
        x,y = self.get_batch(self.batch_ids[self.batch_count])
        self.batch_count = ( self.batch_count + 1 ) % self.n_batch
        return x,y

    def epoch(self):
        """
        """
        return (self.get_batch(i) for i in self.batch_ids)


class FeatureMapper(object):
    """
    """
    def __init__(self, label_mapping = "Sensembed_A", label_unit_norm = True,
                       input_mean = 0, input_unit_norm = False):
        """
        """
        self.label_mapping = label_mapping
        self.input_mean = input_mean
        self.input_unit_norm = input_unit_norm

        if self.label_mapping == "Sensembed_A":
            self.index_map = get_Sensembed_A_labelmap()
        elif self.label_mapping == "Random":
            self.index_map = get_Random_labelmap()
        else:
            log.error("Unknown mapping type {} for LabelMapper".format(mapping))
            raise Exception("Unknown dataset for LabelMapper")

        if label_unit_norm:
            vectors = self.index_map.drop(["BN", "POS", "WNID", "gp", "LABEL"], axis=1).get_values()
            self.index_map[range(400)] = normalize(vectors)

    def map_label(self, label):
        """
            Given a 1d array of labels, return a 2d array of sensembed vectors
        """
        if self.label_mapping == "Sensembed_A":
            smbd = pd.DataFrame(data={"LABEL":label})
            index = self.index_map[self.index_map.LABEL.isin(smbd.LABEL)]
            smbd = smbd.reset_index().merge(index, how="left").set_index('index')
            smbd = smbd.drop(["LABEL", "BN", "POS", "WNID", "gp"], axis=1).get_values().astype("float32")
            return smbd
        else:
            log.error("Unknown mapping type {} for LabelMapper".format(mapping))
            raise Exception("Unknown mapping type {} for LabelMapper".format(mapping))

    def map_input(self, input):
        """
            Given a 1d array of labels, return a 2d array of sensembed vectors
        """

        if self.input_mean:
            log.error("Unknown input mapping type: Non null input mean")
            raise Exception("Unknown input mapping type: Non null input mean")
        elif self.input_unit_norm:
            return input / np.expand_dims(np.linalg.norm(input,axis=1),1)
        else:
            return input
