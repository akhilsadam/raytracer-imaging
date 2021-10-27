import math
import numpy as np
from numba import cuda,types, from_dtype
from raytracer.cudaOptions import rayOptions
from raytracer.cudaOptions import cudaOptions
# Not privatizing the quaternion functions as cuda fails inside a class.

@cuda.jit(device=True)
def calculate_unorm_weight(vertex,ray):
    # note vertex is in screenspace! Only if ray intersects vertex!
    dy = ray.y - vertex.y
    dz = ray.z - vertex.z
    R2 = vertex.radius*vertex.radius
    D2 = dy*dy + dz*dz
    weight2 = R2-D2
    if weight2 > 0:
        return math.sqrt(weight2)
    else:
        return 0
    
@cuda.jit
def unorm_weight(weight,vertex,ray):
    weight[0]=calculate_unorm_weight(vertex,ray)
    
@cuda.jit
def unorm_singleweight(rayNHit,rayHits,rayWeights,voxels,rays,nvoxels,nrays):
    i, j = cuda.grid(2)
    if i < nvoxels and j < nrays:
        weight=calculate_unorm_weight(voxels[i],rays[j])
        if weight > 0:
            pos = rayNHit[j]
            rayWeights[j,pos] = weight
            rayHits[j,pos] = i
            rayNHit[j] += 1
        
#@cuda.jit
def unorm_rayweight(voxels,rays,maximal_length):
    nvoxels = len(voxels)
    nrays = len(rays)
    nweight = int(maximal_length)
    
    threadsperblock = (cudaOptions.maxthreadsper2Dblock,cudaOptions.maxthreadsper2Dblock)
    blockV = math.ceil(nvoxels / cudaOptions.maxthreadsper2Dblock)
    blockRay = math.ceil(nrays / cudaOptions.maxthreadsper2Dblock)
    blockspergrid = (blockV,blockRay)

    rayNHit = np.zeros(nrays,dtype=int)
    rayHits = np.empty((nweight,nrays),dtype=np.int_)
    rayWeights = np.empty((nweight,nrays),dtype=rayOptions.weight_dtype)
    
    print(blockspergrid,threadsperblock)
    unorm_singleweight[blockspergrid, threadsperblock](rayNHit,rayHits,rayWeights,voxels,rays,nvoxels,nrays)
    # weights = weights * (1/np.sum(weights))
    return rayNHit,rayHits,rayWeights