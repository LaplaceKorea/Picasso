import tensorflow as tf
import sys, os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
decimation_module = tf.load_op_library(os.path.join(BASE_DIR, 'tf_decimate_so.so'))
# print(dir(decimation_module))


def decimate_mesh_basic(vertexIn, faceIn, geometryIn, nvIn, mfIn, nvOut,
                        useArea=True, wgtBnd=5, shuffleBins=500, wgtConsist=0):
    '''
    input:
        vertexIn:   (batch_points, 3) float32 array, concatenated points with/without features
        faceIn:     (batch_faces, 3) int32 array, concatenated triangular faces
        geometryIn: (batch_faces, 5) float32 array, geometrics of each face which is
                                     composed of [normal=[nx,ny,nz],intercept=d,area]
        nvIn:       (batch,) int32 vector, point/vertex number of each sample in the batch
        mfIn:       (batch,) int32 vector, face number of each sample in the batch
        nvOut:      (batch,) int32 vector, expected number of points/vertices to output of each sample
        wgtBnd:      float scalar, weight boundary quadric error, (>1) preserves boundary edges
        shuffleBins: integer scalar, the partition bin number for per-sample
    returns:
        vertexOut:  (batch_points, 3) float32 array, concatenated points with/without features
        faceOut:    (batch_faces, 3) int32 array, concatenated triangular faces
        nvOut:     (batch,) int32 vector, point/vertex number of each sample in the batch
        mfOut:     (batch,) int32 vector, face number of each sample in the batch
    '''
    nv2Remove = nvIn - nvOut
    nvIn = tf.cumsum(nvIn, exclusive=False)
    mfIn = tf.cumsum(mfIn, exclusive=False)
    vertexOut, faceOut, isDegenerate, vtReplace, vtMap, \
    nvOut, mfOut = decimation_module.mesh_decimation(vertexIn[:,0:3], faceIn, geometryIn, nvIn, mfIn,
                                                     nv2Remove, shuffleBins, useArea, wgtBnd, wgtConsist)

    faceOut = tf.gather_nd(faceOut, tf.where(tf.logical_not(isDegenerate)))
    vertexOut = tf.concat([vertexOut, vertexIn[:,3:]], axis=1)
    vertexOut = tf.gather_nd(vertexOut, tf.where(vtMap>=0))

    return vertexOut, faceOut, nvOut, mfOut


def mesh_decimation(vertexIn, faceIn, geometryIn, nvIn, mfIn, nvOut,
                    useArea=True, wgtBnd=5, shuffleBins=500, wgtConsist=0):
    '''
    input:
        vertexIn:   (batch_npoints, 3+) float32 array, concatenated points with/without features
        faceIn:     (batch_nfaces, 3) int32 array, concatenated triangular faces
        geometryIn: (batch_nfaces, 5) float32 array, geometrics of each face which is
                                     composed of [normal=[nx,ny,nz],intercept=d,area]
        nvIn:       (batch,) int32 vector, point/vertex number of each sample in the batch
        mfIn:       (batch,) int32 vector, face number of each sample in the batch
        nvOut:      (batch,) int32 vector, expected number of points/vertices to output of each sample
        wgtBnd:      float scalar, weight boundary quadric error, (>1) preserves boundary edges
        shuffleBins: integer scalar, the partition bin number for per-sample
    returns:
        vertexOut:   (batch_mpoints, 3) float32 array, concatenated points with/without features
        faceOut:     (batch_mfaces, 3) int32 array, concatenated triangular faces
        geometryOut: (batch_mfaces, 5) float32 array, geometrics of each output face which is
                                     composed of [normal=[nx,ny,nz],intercept=d,area]
        vtReplace:  (batch_npoints,) int32 array, negative values (remove minus '-') for vertex to be contracted in each cluster
                                                 zero for vertex no change because it forms a cluster by itself
                                                 positive values recording the cluster size excluding the vertex itself
        vtMap:     (batch_npoints,) int32 array, contracted/degenerated vertices got mapping to -1,
                                                the valid vertices got mapping start from 0
        nvOut:     (batch,) int32 vector, point/vertex number of each sample in the batch
        mfOut:     (batch,) int32 vector, face number of each sample in the batch
    '''
    nv2Remove = nvIn - nvOut
    repIn = tf.zeros(shape=tf.shape(vertexIn[:,0]), dtype=tf.int32)
    mapIn = tf.range(tf.shape(vertexIn)[0], dtype=tf.int32)

    loop_vars = [nv2Remove, vertexIn, faceIn, geometryIn, repIn, mapIn, nvIn, mfIn]
    Bshape, Vshape = nv2Remove.get_shape(), repIn.get_shape()
    shape_invariants = [Bshape, tf.TensorShape([None,6]), tf.TensorShape([None,3]),
                        tf.TensorShape([None,5]), Vshape, Vshape, Bshape, Bshape]

    def condition(nv2Remove, vertexIn, faceIn, geometryIn, repIn, mapIn, nvIn, mfIn):
        return tf.reduce_any(tf.greater(nv2Remove, 0))

    def body(nv2Remove, vertexIn, faceIn, geometryIn, repIn, mapIn, nvIn, mfIn):
        nvIn_cumsum = tf.cumsum(nvIn, exclusive=False)
        mfIn_cumsum = tf.cumsum(mfIn, exclusive=False)
        vertexOut, faceOut, isDegenerate, repOut, mapOut, \
        nvOut, mfOut = decimation_module.mesh_decimation(vertexIn[:,0:3], faceIn, geometryIn, \
                                                         nvIn_cumsum, mfIn_cumsum, nv2Remove, \
                                                         shuffleBins, useArea, wgtBnd, wgtConsist)
        faceIn = tf.gather_nd(faceOut, tf.where(tf.logical_not(isDegenerate)))
        vertexIn = tf.concat([vertexOut, vertexIn[:,3:]], axis=1) # added additional features other than xyz
        vertexIn = tf.gather_nd(vertexIn, tf.where(mapOut>=0))
        geometryIn = compute_triangle_geometry(vertexIn[:,:3], faceIn)
        nv2Remove = nv2Remove - (nvIn - nvOut)
        repIn, mapIn = combine_clusters(repIn, mapIn, repOut, mapOut)
        nvIn, mfIn = nvOut, mfOut
        return [nv2Remove, vertexIn, faceIn, geometryIn, repIn, mapIn, nvIn, mfIn]

    final = tf.while_loop(condition, body, loop_vars, shape_invariants)
    vertexOut, faceOut, geometryOut, repOut, mapOut, nvOut, mfOut = final[1:]
    return vertexOut, faceOut, geometryOut, nvOut, mfOut, repOut, mapOut
tf.no_gradient('MeshDecimation')


def combine_clusters(repA, mapA, repB, mapB):
    '''
       input:
            repA: (batch_points,) int32 array, vertex clustering information of LARGE input
            mapA: (batch_points,) int32 array, vertex mappinging information of LARGE input
            repB: (batch_points,) int32 array, vertex clustering information of SMALL/decimated input
            mapB: (batch_points,) int32 array, vertex mappinging information of SMALL/decimated input
       returns:
            repComb: (batch_points,) int32 array, vertex clustering information after merging LARGE/SMALL input
            mapComb: (batch_points,) int32 array, vertex mappinging information after merging LARGE/SMALL input
    '''
    repComb, mapComb = decimation_module.combine_clusters(repA, mapA, repB, mapB)
    return repComb, mapComb
tf.no_gradient('CombineClusters')


def compute_triangle_geometry(vertex, face):
    epsilon = 1e-15
    vec10 = tf.gather_nd(vertex,tf.expand_dims(face[:,1],axis=1)) - tf.gather_nd(vertex,tf.expand_dims(face[:,0],axis=1))
    vec20 = tf.gather_nd(vertex,tf.expand_dims(face[:,2],axis=1)) - tf.gather_nd(vertex,tf.expand_dims(face[:,0],axis=1))
    raw_normal = tf.linalg.cross(vec10, vec20)
    length = tf.sqrt(tf.reduce_sum(tf.multiply(raw_normal,raw_normal),axis=1,keepdims=True))
    area = tf.divide(length,2)
    normal = tf.math.divide(raw_normal,length+epsilon) # unit length normal vectors
    v1 = tf.gather_nd(vertex,tf.expand_dims(face[:,0],axis=1))
    d = -tf.reduce_sum(tf.multiply(normal,v1),axis=1,keepdims=True)  # intercept of the triangle plane
    geometry = tf.concat([normal,d,area],axis=1)
    return geometry
tf.no_gradient('ComputeTriangleGeometry')


def compute_vertex_geometry(vertex, face, face_geometry):
    vertex_geometry = decimation_module.compute_vertex_geometry(vertex, face, face_geometry)
    raw_vertex_normal = vertex_geometry[:,:3]
    vertex_area = vertex_geometry[:,3:]
    length = tf.sqrt(tf.reduce_sum(tf.multiply(raw_vertex_normal,raw_vertex_normal),axis=1,keepdims=True))
    vertex_normal = tf.math.divide_no_nan(raw_vertex_normal,length) # unit length normal vectors
    geometry = tf.concat([vertex_normal,vertex_area],axis=1)
    return geometry
tf.no_gradient('ComputeVertexGeometry')


def count_vertex_adjface(face, vtMap, vertexOut):
    '''
        Count the number of adjacent faces for output vertices, i.e. {vtMap[i] | vtMap[i]>=0}
    '''
    nfCount = decimation_module.count_vertex_adjface(face, vtMap, vertexOut[:,:3])
    return nfCount
tf.no_gradient('CountVertexAdjface')


