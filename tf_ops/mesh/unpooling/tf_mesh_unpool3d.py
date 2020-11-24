import tensorflow as tf
import sys, os

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)
unpool3d_module = tf.load_op_library(os.path.join(base_dir, 'tf_unpool3d_so.so'))


def mesh_interpolate(input, vt_replace, vt_map):
    '''
        During the decimation, we record the vertex clusters in vt_map.
        In unpooling within decoders, we interpolate features for vertices in the
        before-decimation mesh from vertex features of its successive decimated mesh.
        Here we perform average interpolation to get feature, which means that
        the unpooled feature of each output vertex is 1/Nc of that from its input vertex.
        Nc is the related vertex cluster size in decimation.
    '''
    return unpool3d_module.mesh_interpolate(input, vt_replace, vt_map)
@tf.RegisterGradient("MeshInterpolate")
def _mesh_interpolate_grad(op, grad_output):
    input = op.inputs[0]
    vt_replace = op.inputs[1]
    vt_map = op.inputs[2]
    grad_input = unpool3d_module.mesh_interpolate_grad(input, grad_output, vt_replace, vt_map)
    return [grad_input, None, None]


