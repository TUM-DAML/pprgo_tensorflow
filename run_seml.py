import os
import time
import logging
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sacred import Experiment
import seml

from pprgo import utils
from pprgo import ppr
from pprgo import pprgo

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ex = Experiment()
seml.setup_logger(ex)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))


@ex.automain
def run(data_dir, data_fname, split_seed, ntrain_div_classes, attr_normalization,
        alpha, eps, topk, ppr_normalization,
        hidden_size, nlayers, weight_decay, dropout,
        lr, max_epochs, batch_size, batch_mult_val,
        eval_step, run_val,
        early_stop, patience,
        nprop_inference, inf_fraction):
    '''
    Run training and inference.

    Parameters
    ----------
    data_dir:
        Directory containing .npz data files.
    data_fname:
        Name of .npz data file.
    split_seed:
        Seed for splitting the dataset into train/val/test.
    ntrain_div_classes:
        Number of training nodes divided by number of classes.
    attr_normalization:
        Attribute normalization. Not used in the paper.
    alpha:
        PPR teleport probability.
    eps:
        Stopping threshold for ACL's ApproximatePR.
    topk:
        Number of PPR neighbors for each node.
    ppr_normalization:
        Adjacency matrix normalization for weighting neighbors.
    hidden_size:
        Size of the MLP's hidden layer.
    nlayers:
        Number of MLP layers.
    weight_decay:
        Weight decay used for training the MLP.
    dropout:
        Dropout used for training.
    lr:
        Learning rate.
    max_epochs:
        Maximum number of epochs (exact number if no early stopping).
    batch_size:
        Batch size for training.
    batch_mult_val:
        Multiplier for validation batch size.
    eval_step:
        Accuracy is evaluated after every this number of steps.
    run_val:
        Evaluate accuracy on validation set during training.
    early_stop:
        Use early stopping.
    patience:
        Patience for early stopping.
    nprop_inference:
        Number of propagation steps during inference
    inf_fraction:
        Fraction of nodes for which local predictions are computed during inference.
    '''

    start = time.time()
    (adj_matrix, attr_matrix, labels,
     train_idx, val_idx, test_idx) = utils.get_data(
            f"{data_dir}/{data_fname}",
            seed=split_seed,
            ntrain_div_classes=ntrain_div_classes,
            normalize_attr=attr_normalization
    )
    time_loading = time.time() - start

    try:
        d = attr_matrix.n_columns
    except AttributeError:
        d = attr_matrix.shape[1]
    nc = labels.max() + 1

    # compute the ppr vectors for train/val nodes using Anderson
    start = time.time()
    topk_train = ppr.topk_ppr_matrix(adj_matrix, alpha, eps, train_idx, topk,
                                     normalization=ppr_normalization)
    if run_val:
        topk_val = ppr.topk_ppr_matrix(adj_matrix, alpha, eps, val_idx, topk,
                                       normalization=ppr_normalization)
    else:
        topk_val = None
    time_preprocessing = time.time() - start
    logging.info('Preprocessing done.')

    start = time.time()
    model = pprgo.PPRGo(d, nc, hidden_size, nlayers, lr, weight_decay, dropout,
                        sparse_features=type(attr_matrix) is not np.ndarray)

    sess = tf.compat.v1.Session()
    with sess.as_default():
        tf.compat.v1.global_variables_initializer().run()
        nepochs, loss_hist, acc_hist, f1_hist = pprgo.train(
                sess=sess, model=model, attr_matrix=attr_matrix,
                train_idx=train_idx, val_idx=val_idx,
                topk_train=topk_train, topk_val=topk_val,
                labels=labels,
                max_epochs=max_epochs, batch_size=batch_size, batch_mult_val=batch_mult_val,
                eval_step=eval_step, run_val=run_val,
                early_stop=early_stop, patience=patience,
                ex=ex)
    time_training = time.time() - start
    logging.info('Training done.')

    start = time.time()
    predictions, time_logits, time_propagation = model.predict(
            sess=sess, adj_matrix=adj_matrix, attr_matrix=attr_matrix, alpha=alpha,
            nprop=nprop_inference, inf_fraction=inf_fraction,
            ppr_normalization=ppr_normalization)
    time_inference = time.time() - start
    logging.info('Inference done.')

    results = {
            'accuracy_train': accuracy_score(labels[train_idx], predictions[train_idx]),
            'accuracy_val': accuracy_score(labels[val_idx], predictions[val_idx]),
            'accuracy_test': accuracy_score(labels[test_idx], predictions[test_idx]),
            'f1_train': f1_score(labels[train_idx], predictions[train_idx], average='macro'),
            'f1_val': f1_score(labels[val_idx], predictions[val_idx], average='macro'),
            'f1_test': f1_score(labels[test_idx], predictions[test_idx], average='macro'),
    }

    gpu_max_bytes = tf.contrib.memory_stats.MaxBytesInUse()
    results.update({
            'time_loading': time_loading,
            'time_preprocessing': time_preprocessing,
            'time_training': time_training,
            'time_inference': time_inference,
            'time_logits': time_logits,
            'time_propagation': time_propagation,
            'gpu_memory': sess.run(gpu_max_bytes),
            'memory': utils.get_max_memory_bytes(),
            'nepochs': nepochs,
    })
    return results
