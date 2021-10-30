import math
import numpy as np
from numba import cuda,types, from_dtype
from raytracer.cudaOptions import cudaOptions

# Not privatizing the quaternion functions as cuda fails inside a class.

# http://graphics.stanford.edu/courses/cs348a-17-winter/Papers/pdf -- eq#3
# also https://github.com/Unity-Technologies/Unity.Mathematics/blob/master/src/Unity.Mathematics/quaternion.cs
#-------------------------------------------------------- CUDA DEVICE FUNCTION ------------- (not user callable) ------
@cuda.jit(device=True)
def _inner(qx,qy,qz,q2x,q2y,q2z):
    return qx*q2x + qy*q2y + qz*q2z
@cuda.jit(device=True)
def _innerQ(qw,qx,qy,qz,q2w,q2x,q2y,q2z):
    return qw*q2w + qx*q2x + qy*q2y + qz*q2z
@cuda.jit(device=True)
def _outer(qx,qy,qz,q2x,q2y,q2z):
    return ((qy*q2z) - (qz*q2y), (qz*q2x) - (qx*q2z), (qx*q2y) - (qy*q2x))
@cuda.jit(device=True)
def outer(v1,v2):
    qx,qy,qz = v1
    q2x,q2y,q2z = v2
    return _outer(qx,qy,qz,q2x,q2y,q2z)
@cuda.jit(device=True)
def _norm(vx,vy,vz):
    return math.sqrt(_inner(vx,vy,vz,vx,vy,vz))
@cuda.jit(device=True)
def _norm2(vx,vy,vz):
    return _inner(vx,vy,vz,vx,vy,vz)
@cuda.jit(device=True)
def _normQ(qw,qx,qy,qz):
    return math.sqrt(_innerQ(qw,qx,qy,qz,qw,qx,qy,qz))
#-------------------------------------
@cuda.jit(device=True)
def _normalizeV(vx,vy,vz):
    v = _norm(vx,vy,vz)
    return vx/v,vy/v,vz/v
@cuda.jit(device=True)
def _normalizeQ(qw,qx,qy,qz):
    q = _normQ(qw,qx,qy,qz)
    return qw/q,qx/q,qy/q,qz/q
@cuda.jit(device=True)
def normalizeV(vector):
    vx,vy,vz = vector
    return _normalizeV(vx,vy,vz)
@cuda.jit(device=True)
def normalizeQ(q):
    qw,qx,qy,qz = q
    return _normalizeQ(qw,qx,qy,qz)
#-------------------------------------
@cuda.jit(device=True)
def _addQ(qw,qx,qy,qz,q2w,q2x,q2y,q2z):
    return (qw+q2w,qx+q2x,qy+q2y,qz+q2z)
@cuda.jit(device=True)
def _multQ(qw,qx,qy,qz,q2w,q2x,q2y,q2z):
    A = qw*q2w - _inner(qx,qy,qz,q2x,q2y,q2z)
    cx,cy,cz = _outer(qx,qy,qz,q2x,q2y,q2z)
    X = qw*q2x + q2w*qx + cx
    Y = qw*q2y + q2w*qy + cy
    Z = qw*q2z + q2w*qz + cz
    return (A,X,Y,Z)
@cuda.jit(device=True)
def addQ(q,q2):
    qw,qx,qy,qz = q
    q2w,q2x,q2y,q2z = q2
    return _addQ(qw,qx,qy,qz,q2w,q2x,q2y,q2z)
@cuda.jit(device=True)
def multQ(q,q2):
    qw,qx,qy,qz = q
    q2w,q2x,q2y,q2z = q2
    return _multQ(qw,qx,qy,qz,q2w,q2x,q2y,q2z)
@cuda.jit(device=True)
def dirV(vert1,vert2):
    vx1,vy1,vz1 = vert1
    vx2,vy2,vz2 = vert2
    return _normalizeV(vx2-vx1,vy2-vy1,vz2-vz1)
#-------------------------------------
@cuda.jit(device=True)
def conjugate(qw,qx,qy,qz):
    return (qw,-qx,-qy,-qz)
@cuda.jit(device=True)
def _q(angle,vector):
    vx,vy,vz = normalizeV(vector)
    halfangle = angle/2
    cos = math.cos(halfangle)
    sin = math.sin(halfangle)
    return (cos,vx*sin,vy*sin,vz*sin)    
#-------------------------- Rotation Functions ---------------------\
@cuda.jit(device=True)    
def _rotq(vx,vy,vz,qw,qx,qy,qz):
    A = (qw*qw) - _norm2(qx,qy,qz)
    B = 2*_inner(qx,qy,qz,vx,vy,vz)
    cx,cy,cz = _outer(qx,qy,qz,vx,vy,vz)
    C = 2*qw
    X = A*vx + B*qx + C*cx
    Y = A*vy + B*qy + C*cy
    Z = A*vz + B*qz + C*cz
    return X,Y,Z

@cuda.jit(device=True)
def _screenspaceq(vx,vy,vz,qw,qx,qy,qz):
    A = (qw*qw) - _norm2(qx,qy,qz)
    B = 2*_inner(qx,qy,qz,vx,vy,vz)
    _,cy,cz = _outer(qx,qy,qz,vx,vy,vz)
    C = 2*qw
    Y = A*vy + B*qy + C*cy
    Z = A*vz + B*qz + C*cz
    return Y,Z

@cuda.jit(device=True)
def rotq(vertex,q):
    vx,vy,vz = vertex
    #qw,qx,qy,qz = normalizeQ(q)
    qw,qx,qy,qz = normalizeQ(q)
    x,y,z = _rotq(vx,vy,vz,qw,qx,qy,qz)
    return x,y,z

@cuda.jit(device=True)
def screenspaceq(vx,vy,vz,q):
    qw,qx,qy,qz = q
    return _screenspaceq(vx,vy,vz,qw,qx,qy,qz)

# @cuda.jit(device=True)
# def rotate(vertex,angle,axis):
#     vx,vy,vz = normalizeV(vertex)
#     qw,qx,qy,qz = _q(angle,axis)
#     rotq(vx,vy,vz,qw,qx,qy,qz)

@cuda.jit(device=True)
def genq(phi,alpha):
    #alpha = (np.pi/2) - theta
    qA = _q(phi,(0,0,1)) # Z-rotation
    uvector = rotq((0,1,0),qA) # Y -> Y'
    qB = _q(-alpha,uvector) # Y'-rotation
    return multQ(qB,qA)

@cuda.jit(device=True)
def genq_LookRotation(direction):
    # needs normalized input
    dx,dy,dz = direction
    phi = math.atan2(dy,dx)
    dxy = math.sqrt(dx*dx + dy*dy)
    alpha = math.acos(dxy) #dxy/1
    return genq(phi,alpha)  
#-------------------------------------------------------- CUDA KERNELS ------------- (user callable) -------------------
@cuda.jit
def rotateq(vertex,q):
    x,y,z = rotq(vertex,q)
    vertex[0] = x
    vertex[1] = y
    vertex[2] = z
@cuda.jit
def _rotateQ(verts,q):
    i = cuda.grid(1) # thread number
    if i<len(verts):
        x,y,z = rotq(verts[i],q)
        verts[i,0] = x
        verts[i,1] = y
        verts[i,2] = z
        
@cuda.jit
def generateq(q,phi,alpha):
    q[0:4] = genq(phi,alpha)

    
@cuda.jit
def generateVQ_LOR(q,vertex1,vertex2):
    direction = dirV(vertex1,vertex2)
    q[0:4] = genq_LookRotation(direction)
#--------------------------------------------------------
class quaternion:
    @staticmethod
    def rotate(verts,phi=0.0,alpha=0.0):
        workers = len(verts)
        blocks = math.ceil(workers / cudaOptions.maxthreadsperblock)
        #print(blocks)
        q = np.zeros(shape=(4,))
        generateq[1,1](q,phi,alpha)
        #print(q)
        _rotateQ[blocks,cudaOptions.maxthreadsperblock](verts,q)

        
### Some tests:
# vertex = np.array([1.0,1.0,0.0])
# q = np.array([0.966,0.0,0.259,0.0])
# rotateq[1,1](vertex,q)
# print(vertex)

# x = np.array([0.0,0.0,0.0,0.0])
# generateq[1,1](x,1.0,0.0)
# print(x)