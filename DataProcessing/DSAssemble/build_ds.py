import sys, os

import yaml
import pandas as pd
import numpy as np
import hickle as hkl

from ..util.Helpers import Logger, check_file_path
from ..util.ImagePreprocess import get_image, resize, center_crop, test_image
log = Logger()


##### FILEPATH_LABEL association
# Those methods take a source directory as input and return a FILEPATH_LABEL dataframe
# FILEPATH_LABEL dataframes have a row for each dataset sample and two columns (FP, ID):
# FP is the full path of the sample
# ID is the label of the sample
def mapFL_subfolders(img_src_dir):
    """
        Returns a FILEPATH_LABEL DataFrame given the input source_dir.
        This function expects a source dir where each class samples are
        recorded in a corresponding subfolder. The ID associated is the
        name of the subfolder.
        __________________________________
        Parameters:
            img_sr_dir: File path of the root directory containing subfolders
        Outputs:
            FILEPATH_LABEL DataFrame with "ID" and "FP" colulmns
    """
    filepath_list = []
    for subfolder in os.listdir(img_src_dir):
        filepath_list.extend(os.listdir(os.path.join(img_src_dir, subfolder)))
    s = pd.Series(filepath_list)
    return pd.DataFrame(data = { "WNID":s.apply(lambda x: x.split("_")[0]),
                                 "FP":s.apply(lambda x: os.path.join(img_src_dir,x.split("_")[0],x))
                               }
                       )

def mapFL_name(img_src_dir): # USELESS
    """
        Returns a FILEPATH_LABEL DataFrame given the input source_dir.
        This function expects a flat source dir with no subfolders.
        and infers the class of the samples by their name following the naming convention:
        CLASSID_SAMPLEID(.jpg/.png...)
        __________________________________
        Parameters:
            img_sr_dir: File path of the root directory containing subfolders
        Outputs:
            FILEPATH_LABEL DataFrame with "ID" and "FP" colulmns
    """
    s = pd.Series(os.listdir(img_src_dir))
    return pd.DataFrame(data = { "WNID":s.apply(lambda x: x.split("_")[0]),
                                 "FP":s.apply(lambda x: os.path.join(img_src_dir,x))
                               }
                       )




#### FILEPATH_LABEL_DS methods
# These methods take a FILEPATH_LABEL dataframe and return a FILEPATH_LABEL_DS dataframe
# FILEPATH_LABEL_DS dataframes augment FILEPATH_LABEL dataframes with a DS column
# The DS value can take three values: "train", "valid" and "test"
def zero_shot_split_dataset(numimperclass_df, minim=500):
    """
        Splits a set of classes into test, valid and test depending on the number of available
        images within this class.
        __________________________________
        Parameters:
            numimperclass_df: DataFrame with column numImage
            minim: Minimum number of images to be qualified as training data
    """
    f = lambda x: "train" if x > minim else "valid" if x % 2 == 0 else "test"
    nphotos = numimperclass_df["numImage"].apply(f)
    # TODO


def split_dataset_ratio(filepath_label, ratio = (5,1,1)):
    """
        Randomly splits a DataFrame of Filepaths into train, valid and test datasets according
        to a given ratio.
        ________________________________
        Parameters:
            filepath_label = DataFrame as returned by map_wnid_imfilepath function
            ratio = triple of integer as (train_ratio, valid_ratio, test_ratio)
        Outputs:
            dataframe as given in input augmented with a DS column contatining
            values: "train", "valid", "test"
    """
    def split(x):
        y = hash(x) % sum(ratio)
        if y < ratio[0]:
            return "train"
        elif y < sum(ratio[:2]):
            return "valid"
        else:
            return "test"
    filepath_label["DS"] = filepath_label["FP"].apply(split)

#### Dataset saving functions
# The two methods are described in their docstrings
def make_minibatches(filepath_label, dest_dir,
                     mean_filename="mean", batch_size = 256,
                     img_size=256, shuffle = True, resizing = "resize"):
    """
        Serializes a filepath_label dataframe into batches using save_batch method.
        Additionally saves the mean and opitonnally shuffles the filepath_label.
    """
    img_mean    = np.zeros((3, img_size, img_size))
    batch_count = 0
    if shuffle:
        log.info("Shuffling...")
        filepath_label = filepath_label.reindex(np.random.permutation(filepath_label.index))
    for set, ds_frame in filepath_label.groupby(filepath_label["DS"]):
        log.info("Serializing batches for {} ds".format(set))
        for batch_id, batch in ds_frame.groupby(np.arange(len(ds_frame))// batch_size):
            img_mean += save_batch(batch_id, batch, set, dest_dir,  batch_size = batch_size,
                                   img_size=img_size, resizing = resizing)
            batch_count += 1
    img_mean /= batch_count
    mean_path = os.path.join(dest_dir, mean_filename)
    check_file_path(mean_path)
    hkl.dump(img_mean,mean_path, mode = "w")
    return img_mean

def save_batch( batch_id, batch, set, dest_dir, batch_size = 256, img_size=256,
                resizing = "resize"):
    """
        Given a set of file path, a batch id, a destination directory, image and batch sizes
        Serialize resized data as a 4D? tensor, labels as 1d tensor,
        and outputs the mean of the batch
        _ _ _ _ _ _ _ _ _ _ _ _ _
        Parameters:
            batch_id: identifier of the batch
            batch: Pandas DataFrame with a "FP" column
    """
    label_file  = os.path.join(dest_dir, set, "label", str(batch_id))
    input_file  = os.path.join(dest_dir, set, "input", str(batch_id))
    check_file_path(label_file), check_file_path(input_file)
    label_array = np.pad( batch["LABEL"].get_values(),
                          (0,256 - len(batch)),
                          'constant',
                          constant_values=(-1),
                         )
    input_array = np.zeros((batch_size, 3, img_size, img_size), np.uint8)
    for i, filepath in enumerate(batch["FP"].get_values()):
        if resizing == "resize":
            input_array[i,:,:,:] = get_image(filepath, img_size, resize, True, True)
        else:
            input_array[i,:,:,:] = get_image(filepath, img_size, center_crop, True, True)
    log.info("Serializing {} and {}".format(label_file, input_file))
    hkl.dump(label_array.astype("int32"), label_file, mode = "w")
    hkl.dump(input_array.astype("uint8"), input_file, mode = "w")
    return input_array.mean(axis=0)


def deal_with_corrupted_images(df, df_dir):
    df["non_corrupted"] = df.FP.apply(test_image)
    df_corrupted = df[- df.non_corrupted]
    df_correct   = df[df.non_corrupted]
    df_corrupted.to_pickle(os.path.join(df_dir, "corrupted"))
    df_correct.to_pickle(os.path.join(df_dir, "correct"))
    return df_correct
