#!/usr/bin/env python
# -*- coding: utf-8 -*-

def train_clas(ds_path="/home/tristan/data/Imagenet/datasets/Sensembed_A/256_224/center_crop/", nepoch=10, l_r=0.01, reg=0.01):
    bi = BatchIterator(ds_path, smbd=False, input_type="features")
    train_func, valid_func = compile_classification(l_r=l_r, reg=reg)
    v = Validator(smbd=False)
    tr, vl1, vl2= [], [], []

    val_sc1, val_sc2 = 0,0
    for x,y in bi.epoch_clas_valid():
        val,out = valid_func(x,y)
        val_sc1 += val
        val_sc2 += v.clas_top_k_scores(out,y)
    val_sc1 /= len(bi.valid_batches)
    vl1.append(val_sc1)
    val_sc2 /= len(bi.valid_batches)
    vl2.append(val_sc2)
    log.info("Without training: Average valid score= {}. Average top-5 error= {}%".format(val_sc1, 100 * val_sc2))

    for i in range(nepoch):
        start = time.time()
        log.info("{}th Epoch!".format(i))
        tr_sc, val_sc1, val_sc2 = 0,0,0
        for x,y in bi.epoch_clas_train():
            tr_sc += train_func(x,y)
        tr_sc /= len(bi.train_batches)
        tr.append(tr_sc)
        log.info("{}th epoch. Training average score= {}".format(i, tr_sc))
        for x,y in bi.epoch_clas_valid():
            val,out = valid_func(x,y)
            val_sc1 += val
            val_sc2 += v.clas_top_k_scores(out,y)
        val_sc1 /= len(bi.valid_batches)
        vl1.append(val_sc1)
        val_sc2 /= len(bi.valid_batches)
        vl2.append(val_sc2)
        log.info("{}th epoch. Average valid score= {}. Average top-5 error= {}%".format(i, val_sc1, 100 * val_sc2))
        log.info("Epoch execution time: {}s".format(time.time() - start))
    return tr, vl1, vl2

def train_smbd(ds_path="/home/tristan/data/Imagenet/datasets/Sensembed_A/256_224/center_crop/", nepoch=10, l_r=0.01, reg=0.01):
    bi = BatchIterator(ds_path, input_type="features")
    train_func, valid_func = compile_smbd_square(l_r=l_r, reg=reg)
    v = Validator()
    tr, vl1, vl2= [], [], []

    val_sc1, val_sc2 = 0,0
    for x,y,z in bi.epoch_smbd_valid():
        val,out = valid_func(x,y)
        val_sc1 += val
        val_sc2 += v.smbd_top_k_scores(out,z)
    val_sc1 /= len(bi.valid_batches)
    vl1.append(val_sc1)
    val_sc2 /= len(bi.valid_batches)
    vl2.append(val_sc2)
    log.info("Without training: Average valid score= {}. Average top-5 error= {}%".format(val_sc1, 100 * val_sc2))

    for i in range(nepoch):
        start = time.time()
        log.info("{}th Epoch!".format(i))
        tr_sc, val_sc1, val_sc2 = 0,0,0
        for x,y,z in bi.epoch_smbd_train():
            tr_sc += train_func(x,y)
        tr_sc /= len(bi.train_batches)
        tr.append(tr_sc)
        log.info("{}th epoch. Training average score= {}".format(i, tr_sc))
        for x,y,z in bi.epoch_smbd_valid():
            val,out = valid_func(x,y)
            val_sc1 += val
            val_sc2 += v.smbd_top_k_scores(out,z)
        val_sc1 /= len(bi.valid_batches)
        vl1.append(val_sc1)
        val_sc2 /= len(bi.valid_batches)
        vl2.append(val_sc2)
        log.info("{}th epoch. Average valid score= {}. Average top-5 error= {}%".format(i, val_sc1, 100 * val_sc2))
        log.info("Epoch execution time: {}s".format(time.time() - start))
    return tr, vl1, vl2


def run_test(nepoch = 30, func=train_clas, fname="try_clas"):
    f_name = os.path.join("/home/tristan/Desktop/", fname)
    regs = [0.1, 0.01, 0.001, 0.0001]
    lrs = [0.1, 0.01, 0.001]
    def results_to_df(reg, lr, tr,vl1,vl2):
        tr = pd.DataFrame(data={"reg":reg, "lr":lr, "score_type":"training", "scores": tr, "epoch":range(len(tr))})
        vl1 = pd.DataFrame(data={"reg":reg, "lr":lr, "score_type":"valid", "scores": vl1, "epoch":range(len(vl1))})
        vl2 = pd.DataFrame(data={"reg":reg, "lr":lr, "score_type":"topk", "scores": vl2, "epoch":range(len(vl2))})
        return pd.concat([tr,vl1,vl2])
    df = None
    for reg in regs:
        for lr in lrs:
            log.info("lr:{}   |  reg:{}".format(lr, reg))
            tr, vl1, vl2 = func(nepoch=nepoch, l_r=lr, reg=reg)
            if df is None:
                df = results_to_df(reg,lr,tr,vl1,vl2)
            else:
                df = pd.concat([df, results_to_df(reg,lr,tr,vl1,vl2)])
    df.to_pickle(f_name)


