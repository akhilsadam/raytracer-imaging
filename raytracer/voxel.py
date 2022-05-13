import math
import numpy as np
from numba import cuda,jit
from raytracer.cudaOptions import cudaOptions

# Not privatizing the cuda functions as cuda fails inside a class.

#Ray and Voxel:
SphereVoxel = np.dtype([
    # value
    ('r', 'f8'),
    # sphere (x, y, z) coordinates
    ('x', 'f8'),  ('y', 'f8'), ('z', 'f8'),], align=True) 
Ray = np.dtype([
    # screenspace (y, z) coordinates (assuming direction along -x)
    ('y', 'f8'), ('z', 'f8'),], align=True)

# rays = np.empty(shape=(1,),dtype=Ray)
# unorm_rayweight[1,1](weights,voxels[0],rays[0])
# weights
    
# workers = len(verts)
# blocks = math.ceil(workers / cudaOptions.maxthreadsperblock)
# setVoxelPosition[blocks,cudaOptions.maxthreadsperblock](voxels,verts,voxel_radius

@cuda.jit
def setVoxelPosition(voxels,verts,nvoxels):
    i = cuda.grid(1) # thread number
    if i<nvoxels:
        x,y,z = verts[i]
        voxels[i].x = x
        voxels[i].y = y
        voxels[i].z = z
        
@cuda.jit
def setRayPosition(rays,rayverts,camera_nrays):
    i = cuda.grid(1) # thread number
    if i<camera_nrays:
        y,z = rayverts[i]
        rays[i].y = y
        rays[i].z = z

        
class voxel:
    # prepare voxel coordinates
    def generateEmptyVoxels(voxel_size,fill=0):
        x, y, z = np.indices(voxel_size)
        x = x - int(voxel_size[0]/2)
        y = y - int(voxel_size[1]/2)
        z = z - int(voxel_size[2]/2)
        nvoxels = voxel_size[0]*voxel_size[1]*voxel_size[2]
        
        voxels = np.full(nvoxels,fill_value=fill,dtype=SphereVoxel)
        verts = np.array([x.flatten(),y.flatten(),z.flatten()])
        vertsF = verts.T.astype(float)

        blocks = math.ceil(nvoxels / cudaOptions.maxthreadsperblock)
        setVoxelPosition[blocks,cudaOptions.maxthreadsperblock](voxels,vertsF,nvoxels)
        return voxels,verts,nvoxels

class pixel:   
    def generateCamera(f_Ny,voxel_size):
        maximal_length = int(math.sqrt(np.sum(np.power(voxel_size,2)))) 
        range_length = 0.5*maximal_length
        camera_size = np.array([f_Ny*maximal_length,f_Ny*maximal_length])
        camera_nrays = camera_size[0]*camera_size[1]
        camera_range = [[-range_length,range_length],[-range_length,range_length]]
        return maximal_length,range_length,camera_size,camera_nrays,camera_range
    
    def generateEmptyRays(camera_size,maximal_length,camera_nrays):
        ry,rz = np.indices(camera_size)
        ys = maximal_length*((ry+0.5)/camera_size[0] - 0.5)
        zs = maximal_length*((rz+0.5)/camera_size[1] - 0.5)
        
        rays = np.empty(shape=(camera_nrays,),dtype=Ray)
        rayverts = np.array([ys.flatten(),zs.flatten()]).T.astype(float)
        
        blocks = math.ceil(camera_nrays / cudaOptions.maxthreadsperblock)
        setRayPosition[blocks,cudaOptions.maxthreadsperblock](rays,rayverts,camera_nrays)
        return rays,rayverts