#!/usr/bin/env python
# -*- coding: utf-8 -*-


def get_Sensembed_A_labelmap():
    imnet_sembed   = merge_wn_sembed()
    imnet_labels   = load_Sensembed_A_labelmap()
    return imnet_labels.merge(imnet_sembed)

def get_batch(i, ds_path):
    """
    """
    x = hkl.load(os.path.join(ds_path, "features", str(i)))
    z = hkl.load(os.path.join(ds_path, "label", str(i)))
    y = pd.DataFrame(data={"LABEL":z})
    index = index_map[index_map.LABEL.isin(y.LABEL)]
    y = y.reset_index().merge(index, how="left").set_index('index')
    y = y.drop(["LABEL", "BN", "POS", "WNID", "gp"], axis=1).get_values().astype("float32")
    return x,y,z

class BatchIterator(object):
    def __init__(self, ds_path, input_type = "features", mean=0,
                 train = True, valid = True, smbd = True, clas = True,
                 clas_mode = None):
        """
        """

        self.mean          = mean
        self.input_type    = input_type
        self.clas_mode     = clas_mode

        if smbd:
            self.index_map     = get_Sensembed_A_labelmap()
        if clas:
            self.caffe2012_map = label_map_2012()
        if train:
            self.train_path    = os.path.join(ds_path, "train")
            assert input_type in os.listdir(self.train_path) and "label" in os.listdir(self.train_path)
            self.train_batches = os.listdir(os.path.join(self.train_path, self.input_type))
        if valid:
            self.valid_path    = os.path.join(ds_path, "valid")
            assert self.input_type in os.listdir(self.valid_path) and "label" in os.listdir(self.valid_path)
            self.valid_batches = os.listdir(os.path.join(self.valid_path, self.input_type))
        self.smbd_valid_count = 0
        self.smbd_train_count = 0
        self.clas_valid_count = 0
        self.clas_train_count = 0

    def _label_to_smbd(self, label):
        """
            Given a 1d array of labels, return a 2d array of sensembed vectors
        """
        smbd = pd.DataFrame(data={"LABEL":label})
        index = self.index_map[self.index_map.LABEL.isin(smbd.LABEL)]
        smbd = smbd.reset_index().merge(index, how="left").set_index('index')
        smbd = smbd.drop(["LABEL", "BN", "POS", "WNID", "gp"], axis=1).get_values().astype("float32")
        return smbd

    def _label_to_class(self, label):
        """
            Takes care of the label alignment
        """
        if self.clas_mode == "caffe2012":
            return pd.DataFrame(data={"LABEL":label}).reset_index().merge(self.caffe2012_map,
                                              how="left").set_index('index').CaffeLabel.get_values()
        else:
            return label

    def _raw_batch(self, path, i):
        """
        """
        x,y = (hkl.load(os.path.join(path, self.input_type, str(i))).astype("float32"),
               hkl.load(os.path.join(path, "label", str(i))).astype("uint8"))
        if self.input_type == "input":
            return x - self.mean, y.astype()
        else:
            return x,y

    def get_smbd_valid(self, i):
        """
        """
        x,z = self._raw_batch(self.valid_path, i)
        y   = self._label_to_smbd(z)
        return x,y,z

    def get_smbd_train(self, i):
        """
        """
        x,z = self._raw_batch(self.train_path, i)
        y   = self._label_to_smbd(z)
        return x,y,z

    def get_clas_valid(self, i):
        """
        """
        x,z = self._raw_batch(self.valid_path, i)
        y = self._label_to_class(z)
        return x,y

    def get_clas_train(self, i):
        """
        """
        x,z = self._raw_batch(self.train_path, i)
        y = self._label_to_class(z)
        return x,y

    def next_smbd_valid(self):
        """
        """
        res = self.get_smbd_valid(self.valid_batches[self.smbd_valid_count])
        self.smbd_valid_count = (self.smbd_valid_count +1) % len(self.valid_batches)
        return res

    def next_smbd_train(self):
        """
        """
        res = self.get_smbd_train(self.train_batches[self.smbd_train_count])
        self.smbd_train_count = (self.smbd_train_count +1) % len(self.train_batches)
        return res

    def next_clas_valid(self):
        """
        """
        res = self.get_clas_valid(self.valid_batches[self.clas_valid_count])
        self.clas_valid_count = (self.clas_valid_count +1) % len(self.valid_batches)
        return res

    def next_clas_train(self):
        """
        """
        res = self.get_clas_train(self.train_batches[self.clas_train_count])
        self.clas_train_count = (self.clas_train_count +1) % len(self.train_batches)
        return res

    def epoch_clas_train(self):
        """
        """
        return (self.next_clas_train() for i in range(len(self.train_batches)))

    def epoch_clas_valid(self):
        """
        """
        return (self.next_clas_valid() for i in range(len(self.valid_batches)))

    def epoch_smbd_train(self):
        """
        """
        return (self.next_smbd_train() for i in range(len(self.train_batches)))

    def epoch_smbd_valid(self):
        """
        """
        return (self.next_smbd_valid() for i in range(len(self.valid_batches)))

