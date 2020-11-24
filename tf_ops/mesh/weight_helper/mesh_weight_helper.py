import tensorflow as tf
import sys, os

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)
weight_module = tf.load_op_library(os.path.join(base_dir, 'tf_metric2weight_so.so'))


def mesh_metric2weight(originalData, decimateData, vtReplace, vtMap, is_normal=False):
    '''
    if normal is False: compute raw weight based on inverse of euclidean distance
                        between xyz of original and decimated vertices ;
    if normal is True: compute raw weight based on cosin similary between normals
                       of original and decimated vertices.
    '''
    sequence = tf.range(tf.shape(vtReplace)[0], dtype=tf.int32)
    index = tf.where(vtReplace < 0, -vtReplace, sequence)  # cluster index in input
    idx = tf.gather(vtMap, index)
    ID = tf.maximum(idx, 0)
    ID = tf.expand_dims(ID, axis=1)
    nvIn, nvOut = tf.shape(originalData)[0], tf.shape(decimateData)[0]
    decimateData = tf.concat([decimateData, tf.zeros((nvIn - nvOut, 3))], axis=0)

    if not tf.constant(is_normal):
        delta = originalData - tf.gather_nd(decimateData, ID)
        eucli_dist = tf.sqrt(tf.reduce_sum(tf.square(delta), axis=1))
        weight = tf.divide(1.0, eucli_dist+2e-16)
        weight = tf.where(idx>=0, weight, tf.zeros((nvIn)))
    else:
        cosin_similarity = tf.reduce_sum(tf.multiply(originalData,  # cos<normal_In, normal_Out>
                                         tf.gather_nd(decimateData, ID)), axis=-1)
        weight = tf.where(idx>=0, cosin_similarity, tf.zeros((nvIn)))

    normalized_weight = weight_module.normalize_weight(weight, vtReplace, vtMap)
    return normalized_weight
tf.no_gradient('NormalizeWeight')
