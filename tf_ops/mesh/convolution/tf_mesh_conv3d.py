import tensorflow as tf
import sys, os

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)
mesh_modules = tf.load_op_library(os.path.join(base_dir, 'tf_mesh_conv3d_so.so')) # 2D surfaces
# print(dir(mesh_modules))


def edge_conv3d(input, filter, vertex, face, vtMap, fuzzy=True):
    '''
    Input:
        input:  (concat_NvIn, in_channels) float32 array, input point features
        filter: (binsize, in_channels, multiplier) float32 array, convolution filter
        vertex: (concat_NvIn, 3) float32 array, batch-sample concatenated xyz array
        face:   (concat_NfIn, 3) int32 array, batch-sample concatenated facet-vertex-indices array
        vtMap:  (concat_NvIn,) int32 array, compute output features for input vertex mapping to
                                            non-negative indices in output
        fuzzy:  bool value, if true, use fuzzy SPH3D, if false, use 0/1 SPH3D
    Output:
        output: (concat_NvOut, out_channels) float32 array, output point features
                                             output_channels=in_channels*channel_multiplier
    '''
    return mesh_modules.edge_conv3d(input, filter, vertex, face, vtMap, fuzzy)
@tf.RegisterGradient("EdgeConv3d")
def _edge_conv3d_grad(op, grad_output):
    input = op.inputs[0]
    filter = op.inputs[1]
    vertex = op.inputs[2]
    face = op.inputs[3]
    vtMap = op.inputs[4]
    fuzzy = op.inputs[5]
    grad_input, grad_filter = mesh_modules.edge_conv3d_grad(input, filter, grad_output, vertex,
                                                                face, vtMap, fuzzy)
    return [grad_input, grad_filter, None, None, None, None, None]


def facet2facet_conv3d(input, filter, intrplWgts, numInterior):
    '''
    Compute the feature of each triangular face based on the provided interpolated interior features.
    The common input feature for this function is facet textures.
    Input:
        input:       (concat_NfIn*maxK, in_channels) float32 array, input facet interpolated features
        filter:      (3, in_channels, multiplier) float32 array, convolution filter
        intrplWgts:  (concat_NfIn*maxK, 3) float32 array, face interior interpolation weights,
                                                           Barycentric Coordinates
        numInterior: (concat_NfIn) int32 vector, number of interpolated interior points in each facet
    Output:
        output: (concat_NfIn, out_channels) float32 array, output facet features
                                            out_channels = in_channels * multiplier
    '''
    return mesh_modules.facet2facet_conv3d(input, filter, intrplWgts, numInterior)
@tf.RegisterGradient('Facet2facetConv3d')
def _facet2facet_conv3d_grad(op, grad_output):
    input  = op.inputs[0]
    filter = op.inputs[1]
    intrplWgts  = op.inputs[2]
    numInterior = op.inputs[3]
    grad_input, grad_filter = mesh_modules.facet2facet_conv3d_grad(input, filter, grad_output,
                                                                   intrplWgts, numInterior)
    return [grad_input, grad_filter, None, None]


def vertex2facet_conv3d(input, filter, face, numInterval):
    '''
    Compute the feature of each triangular face based on its vertices' features by interpolation
    Input:
        input:       (concat_NvIn, in_channels) float32 array, input vertex/point features
        filter:      (3, in_channels, multiplier) float32 array, convolution filter
        face:        (concat_NfIn, 3) int32 array, vertex list of each facet
        numInterval: (concat_NfIn) int32 vector, number of intervals to divide during the
                                   interior interpolation process of each facet
    Output:
        output:      (concat_NfIn, out_channels) float32 array, output facet features
                                                 out_channels = in_channels * multiplier
    '''
    return mesh_modules.vertex2facet_conv3d(input, filter, face, numInterval)
@tf.RegisterGradient('Vertex2facetConv3d')
def _vertex2facet_conv3d_grad(op, grad_output):
    input  = op.inputs[0]
    filter = op.inputs[1]
    face = op.inputs[2]
    numInterval = op.inputs[3]
    grad_input, grad_filter = mesh_modules.vertex2facet_conv3d_grad(input, filter, grad_output, face, numInterval)
    return [grad_input, grad_filter, None, None]


def facet2vertex_conv3d(input, filter, coeff, face, nfCount, vtMap):
    '''
        Input:
            input:   (concat_NfIn, in_channels) float32 array, input facet features
            filter:  (modelSize, in_channels, multiplier) float32 array, convolution filter
            coeff:   (concat_NfIn, modelSize) float32 array, coefficients for each model
            face:    (concat_NfIn, 3) int32 array, vertex list of each facet
            nfCount: (concat_NvIn) int32 vector, number of adjacent faces of each vertex
            vtMap:   (concat_NvIn) int32 vector, input to output vertex index mapping
        Output:
            output:  (concat_NvOut, out_channels) float32 array, output vertex/point features,
                                                  out_channels = in_channels * multiplier
        '''
    return mesh_modules.facet2vertex_conv3d(input, filter, coeff, face, nfCount, vtMap)
@tf.RegisterGradient('Facet2vertexConv3d')
def _facet2vertex_conv3d_grad(op, grad_output):
    input = op.inputs[0]
    filter = op.inputs[1]
    coeff = op.inputs[2]
    face = op.inputs[3]
    nfCount = op.inputs[4]
    vtMap = op.inputs[5]
    grad_input, grad_filter, \
    grad_coeff = mesh_modules.facet2vertex_conv3d_grad(input, filter, coeff, grad_output,
                                                       face, nfCount, vtMap)
    return [grad_input, grad_filter, grad_coeff, None, None, None]


def compute_gmm_diagonal_coeffs(data, model_mu, model_lambda):
    '''
    Input:
         data:         (N, 3), N is the size of the data, face normals used in our experiment
         model_mu:     (T, 3), T is the number of templates/clusters in the gaussian mixture model
         model_lambda: (T), reciprocal of standard deviations of angle between each face normals with
                            each gmm model. We choose this for the reason that dividing by ~zero will
                            be problematic, so replace it with multiplication with their reciprocals
    Output:
         coeff:       (N, T), coefficients specifying percentage of the data in each component of
                              the mixture model
    '''
    # normalize length of clustering center normals
    epsilon = 1e-15
    length = tf.sqrt(tf.reduce_sum(tf.square(model_mu), axis=-1, keepdims=True))
    model_mu = tf.math.divide(model_mu, length+epsilon)  # unit length normal vectors

    Nf = tf.shape(data)[0]
    Nmodel = tf.shape(model_mu)[0]
    expand_data = tf.expand_dims(data, axis=1)
    expand_mu = tf.expand_dims(model_mu, axis=0)
    delta = expand_data - expand_mu
    square_dist = tf.reduce_sum(tf.square(delta),axis=-1)
    square_lambda = tf.square(model_lambda)
    broadcast_square_lambda = tf.broadcast_to(square_lambda,[Nf,Nmodel])
    # Note: reciprocal value leads to tf.multiply
    zeta = tf.multiply(square_dist, broadcast_square_lambda)

    # clip exponential tails to zero
    coeff = tf.exp(-zeta/2)
    coeff = tf.divide(coeff, tf.reduce_sum(coeff, axis=-1, keepdims=True)+epsilon)
    coeff = tf.where(coeff<0.01, 0.0, coeff)
    return coeff
