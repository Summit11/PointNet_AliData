import tensorflow as tf
import math
import time
import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import tf_util
import provider
from hParams import hParams

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32,
                                    shape=(batch_size, num_point, 4))
    labels_pl = tf.placeholder(tf.int32,
                                shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl

def get_model(point_cloud, is_training, bn_decay=None):
    """ ConvNet baseline, input is BxNx3 gray image """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    is_bn = True
    ''' global feature extract Net '''
    input_image = tf.expand_dims(point_cloud, -1)
    # CONV
    net = tf_util.conv2d(input_image, 64, [1, 4], padding='VALID', stride=[1, 1],
                         bn=is_bn, is_training=is_training, scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1, 1], padding='VALID', stride=[1, 1],
                         bn=is_bn, is_training=is_training, scope='conv2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1, 1], padding='VALID', stride=[1, 1],
                         bn=is_bn, is_training=is_training, scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1, 1], padding='VALID', stride=[1, 1],
                         bn=is_bn, is_training=is_training, scope='conv4', bn_decay=bn_decay)
    points_feat1 = tf_util.conv2d(net, 1024, [1, 1], padding='VALID', stride=[1, 1],
                                  bn=is_bn, is_training=is_training, scope='conv5', bn_decay=bn_decay)
    # MAX
    pc_feat1 = tf_util.max_pool2d(points_feat1, [num_point, 1], padding='VALID', scope='maxpool1')
    # FC
    pc_feat1 = tf.reshape(pc_feat1, [batch_size, -1])
    pc_feat1 = tf_util.fully_connected(pc_feat1, 256, bn=is_bn, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    pc_feat1 = tf_util.fully_connected(pc_feat1, 128, bn=is_bn, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    #print(pc_feat1)

    ''' concat with pointwise feature '''
    # CONCAT
    pc_feat1_expand = tf.tile(tf.reshape(pc_feat1, [batch_size, 1, 1, -1]), [1, num_point, 1, 1])
    points_feat1_concat = tf.concat(axis=3, values=[points_feat1, pc_feat1_expand])


    # CONV
    net = tf_util.conv2d(points_feat1_concat, 512, [1, 1], padding='VALID', stride=[1, 1],
                         bn=is_bn, is_training=is_training, scope='conv6')
    net = tf_util.conv2d(net, 256, [1, 1], padding='VALID', stride=[1, 1],
                         bn=is_bn, is_training=is_training, scope='conv7')
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training, scope='dp1')
    net = tf_util.conv2d(net, 8, [1, 1], padding='VALID', stride=[1, 1],
                         activation_fn=None, scope='conv8')
    net = tf.squeeze(net, [2])

    return net

def get_loss(pred, label):
    """ pred: B,N,13
        label: B,N """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    return tf.reduce_mean(loss)

if __name__ == "__main__":
    with tf.Graph().as_default():
        pointcloud_pl, feature_pl, voxel_index, voxel_num_pl, labels_pl = placeholder_inputs(2, 45, 58386)
        is_training_pl = tf.placeholder(tf.bool, shape=())
        file_name = '{}_batch_{}.npy'.format('train', 0)
        data = np.fromfile(hParams.TRAIN_FILE_PATH + file_name, np.float32)
        data = data.reshape(-1, 6)
        net = get_model(pointcloud_pl, feature_pl, voxel_index, voxel_num_pl, tf.constant(True))
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            start = time.time()
            point_clouds, voxel_info, pointvoxel_indexs, voxel_num, Y = provider.genBatchData(data, [0,1], 58386)
            for i in range(10):
                print(i)
                sess.run(net, feed_dict={pointcloud_pl: point_clouds,
                                         feature_pl: voxel_info,
                                         voxel_num_pl: voxel_num,
                                         voxel_index: pointvoxel_indexs,
                                         labels_pl: Y,
                                         is_training_pl: True})
            print(time.time() - start)
