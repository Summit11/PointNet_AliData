import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
from sklearn.model_selection import train_test_split
import pandas as pd
import time

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import tf_util
from model import *
from hParams import hParams


is_training = True
global_step = 0

LOG_DIR = './log'
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp model.py %s' % (LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
        hParams.BASE_LEARNING_RATE,  # Base learning rate.
        batch * hParams.BATCH_SIZE,  # Current index into the dataset.
        hParams.DECAY_STEP,  # Decay step.
        hParams.DECAY_RATE,  # Decay rate.
        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!!
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
        hParams.BN_INIT_DECAY,
        batch * hParams.BATCH_SIZE,
        hParams.BN_DECAY_DECAY_STEP,
        hParams.BN_DECAY_DECAY_RATE,
        staircase=True)
    bn_decay = tf.minimum(hParams.BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def main():
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(hParams.GPU_INDEX)):
            pointclouds_pl, labels_pl = placeholder_inputs(hParams.BATCH_SIZE, hParams.NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss
            pred = get_model(pointclouds_pl, is_training=is_training_pl, bn_decay=bn_decay)
            loss = get_loss(pred, labels_pl)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(hParams.BATCH_SIZE * hParams.NUM_POINT)
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if hParams.OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=hParams.MOMENTUM)
            elif hParams.OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        #config.log_device_placement = True
        sess = tf.Session(config=config)

        ckpt = tf.train.get_checkpoint_state(LOG_DIR)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            log_string('loading model {}'.format(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            # Init variables
            init = tf.global_variables_initializer()
            sess.run(init, {is_training_pl: is_training})

        if is_training:
            # Add summary writers
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                                 sess.graph)
            test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

            ops = {'pointclouds_pl': pointclouds_pl,
                   'labels_pl': labels_pl,
                   'is_training_pl': is_training_pl,
                   'pred': pred,
                   'loss': loss,
                   'train_op': train_op,
                   'merged': merged,
                   'step': batch}

            for epoch in range(hParams.MAX_EPOCH):
                log_string('**** EPOCH %03d ****' % (epoch))
                sys.stdout.flush()

                train_one_epoch(sess, ops, train_writer)
                eval_one_epoch(sess, ops, test_writer)

                # Save the variables to disk.
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), global_step)
                log_string("Model saved in file: %s" % save_path)

        else:
            ops = {'pointclouds_pl': pointclouds_pl,
                   'is_training_pl': is_training_pl,
                   'pred': pred}
            inference(sess, ops)


def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    log_string('----')
    global global_step
    for file_idx in range(hParams.TRAIN_FILE_NUM):
        #read npy file
        file_name = '{}_batch_{}.npy'.format('train', file_idx)
        data = np.fromfile(hParams.TRAIN_FILE_PATH + file_name, np.float32)
        data = data.reshape(-1, 6)
        #get flame_idxs
        flame_idxs = np.unique(data[:, 0])
        np.random.shuffle(flame_idxs)
        num_batches = len(flame_idxs) // hParams.BATCH_SIZE
        loss_sum = 0

        for batch_idx in range(num_batches):
            start_idx = batch_idx * hParams.BATCH_SIZE
            end_idx = min((batch_idx + 1) * hParams.BATCH_SIZE, len(flame_idxs))
            batch_flame_idxs = flame_idxs[start_idx : end_idx]
            point_clouds, Y = provider.genBatchData(data, batch_flame_idxs, hParams.NUM_POINT)

            feed_dict = {ops['pointclouds_pl']: point_clouds,
                         ops['labels_pl']: Y,
                         ops['is_training_pl']: True}
            summary, step, _, loss_val, pred_val = sess.run(
                [ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred']],
                feed_dict = feed_dict
            )
            global_step = step
            train_writer.add_summary(summary, step)
            pred_val = np.argmax(pred_val, 2)
            loss_sum += loss_val

            if batch_idx % 10 == 0:
                log_string('Current train batch/total batch num: %d/%d' % (batch_idx, num_batches))
                if batch_idx != 0 and batch_idx % 20 == 0:
                    iou = provider.get_iou(provider.label2hot(Y, hParams.NUM_CLASSES),
                                           provider.label2hot(pred_val, hParams.NUM_CLASSES))
                    log_string('train mean iou: %f, cyclist: %f, tricycle: %f, sm allMot: %f, bigMot: %f, pedestrian: %f, crowds: %f, unknown: %f' %
                        (iou[0], iou[1], iou[2], iou[3], iou[4], iou[5], iou[6],iou[7]))

        log_string('train mean loss: %f' % (loss_sum / float(num_batches)))

def eval_one_epoch(sess, ops, test_writer):
    log_string('======================== valid ========================')
    for file_idx in range(hParams.VALID_FILE_NUM):
        file_name = '{}_batch_{}.npy'.format('train', hParams.TRAIN_FILE_NUM + file_idx)
        data = np.fromfile(hParams.VALID_FILE_PATH + file_name, np.float32)
        data = data.reshape(-1, 6)

        flame_idxs = np.unique(data[:, 0])
        np.random.shuffle(flame_idxs)
        num_batches = len(flame_idxs) // hParams.BATCH_SIZE
        loss_sum = 0
        # valid_iou_sum = 0
        for batch_idx in range(num_batches):
            start_idx = batch_idx * hParams.BATCH_SIZE
            end_idx = min((batch_idx + 1) * hParams.BATCH_SIZE, len(flame_idxs))
            batch_flame_idxs = flame_idxs[start_idx: end_idx]
            point_clouds, Y = provider.genBatchData(
                data, batch_flame_idxs, hParams.NUM_POINT)

            feed_dict = {ops['pointclouds_pl']: point_clouds,
                         ops['labels_pl']: Y,
                         ops['is_training_pl']: False}
            summary, step, loss_val, pred_val = sess.run(
                [ops['merged'], ops['step'], ops['loss'], ops['pred']],
                feed_dict = feed_dict
            )
            test_writer.add_summary(summary, step)
            pred_val = np.argmax(pred_val, 2)
            loss_sum += loss_val

            if batch_idx % 10 == 0:
                log_string('**Current train batch/total batch num: %d/%d' % (batch_idx, num_batches))
                if batch_idx != 0 and batch_idx % 20 == 0:
                    iou = provider.get_iou(provider.label2hot(Y, hParams.NUM_CLASSES),
                                           provider.label2hot(pred_val, hParams.NUM_CLASSES))
                    log_string(
                        '**eval mean iou: %f, cyclist: %f, tricycle: %f, sm allMot: %f, bigMot: %f, pedestrian: %f, crowds: %f, unknown: %f' %
                        (iou[0], iou[1], iou[2], iou[3], iou[4], iou[5], iou[6], iou[7]))
        log_string('**eval mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('======================== valid ========================')


def inference(sess, ops):
    if not os.path.exists(hParams.RESULT_PATH):
        os.mkdir(hParams.RESULT_PATH)
    for file_data, name_index in provider.npy_read(hParams.TEST_FILE_PATH, is_training=False):
        num_batches = len(name_index) // hParams.BATCH_SIZE
        flame_idxs = np.unique(file_data[:, 0])
        for batch_idx in range(num_batches):
            start_idx = batch_idx * hParams.BATCH_SIZE
            end_idx = min((batch_idx + 1) * hParams.BATCH_SIZE, len(name_index))
            batch_flame_idxs = flame_idxs[start_idx: end_idx]
            point_clouds, fix_head, fix_tail = provider.genBatchDataTest(file_data, batch_flame_idxs, hParams.NUM_POINT)
            feed_dict = {ops['pointclouds_pl']: point_clouds,
                         ops['is_training_pl']: False}
            start = time.time()
            pred_val = sess.run(ops['pred'], feed_dict=feed_dict)
            end = time.time()
            print('time used : %f' % (end - start))
            pred_val = np.argmax(pred_val, axis = 2)
            for i in range(pred_val.shape[0]):
                single_result = pred_val[i, ...].reshape(-1)
                pd.DataFrame(single_result[fix_head: -fix_tail]).to_csv(hParams.RESULT_PATH + name_index.iloc[:, 0][batch_flame_idxs[i]],
                                                                   header=None, index=False)

if __name__ == '__main__':
    main()
    LOG_FOUT.close()
