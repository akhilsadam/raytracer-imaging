import math
import numpy as np
from numba import cuda,types, from_dtype, jit
from raytracer.cudaOptions import rayOptions
from raytracer.cudaOptions import cudaOptions
from raytracer.rotation.quaternion import *
from raytracer.raytracer.voxel import *
from raytracer.algorithms.algorithms import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
# Not privatizing the cuda functions as cuda fails inside a class.
# Ought to remove any remaining for-loops, but with the caveat: numba does not support kernel creation within kernels...
##################################
#------ CUDA functions ----------#
##################################
@cuda.jit(device=True)
def sumV(v):
    s = 0.0
    for i in range(len(v)):
        s = s + v[i]
    return s

@cuda.jit(device=True)
def divV(v,d):
    for i in range(len(v)):
        v[i] = v[i]/d

@cuda.jit(device=True)
def normL1(vector):
    mag = sumV(vector)
    divV(vector,mag)
    
#obsolete
@cuda.jit(device=True)
def calculate_unorm_weight(vertex,ray,voxel_radius,precision):
    # note vertex is in screenspace! Only if ray intersects vertex!
    dy = ray.y - vertex.y
    dz = ray.z - vertex.z
    R2 = vertex.radius*vertex.radius
    D2 = dy*dy + dz*dz
    weight2 = R2-D2
    if weight2 > precision:
        return math.sqrt(weight2)/voxel_radius
    else:
        return 0
    
@cuda.jit(device=True)
def calculate_rotated_unorm_weight(vertex,ray,voxel_radius,precision,q):
    # screenspace conversion
    y,z = screenspaceq(vertex.x,vertex.y,vertex.z,q)
    # continue with regular calculation
    dy = ray.y - y
    dz = ray.z - z
    R2 = vertex.radius*vertex.radius
    D2 = dy*dy + dz*dz
    weight2 = R2-D2
    if weight2 > precision:
        return math.sqrt(weight2)/voxel_radius
    else:
        return 0
    
@cuda.jit(device=True)
def calculate_singleray_rotated_unorm_weight(vertex,ray_y,ray_z,voxel_radius,precision,q):
    # screenspace conversion
    y,z = screenspaceq(vertex.x,vertex.y,vertex.z,q)
    # continue with regular calculation
    dy = ray_y - y
    dz = ray_z - z
    R2 = vertex.radius*vertex.radius
    D2 = dy*dy + dz*dz
    weight2 = R2-D2
    if weight2 > precision:
        return math.sqrt(weight2)/voxel_radius
    else:
        return 0
    
@cuda.jit(device=True)
def calculate_singleproject(voxels,hits,nhits):
    single_projection = 0
    for u in range(nhits):
        i = hits[u]
        single_projection = single_projection + voxels[i].r
    return single_projection   
##################################
#------ CUDA kernels ------------#  
##################################
@cuda.jit
def unorm_weight(weight,vertex,ray,voxel_radius):
    weight[0]=calculate_unorm_weight(vertex,ray,voxel_radius)

# obsolete
@cuda.jit
def unorm_singleweight(rayNHit,rayHits,rayWeights,voxels,rays,nvoxels,nrays,voxel_radius,precision):
    i, j = cuda.grid(2)
    if i < nvoxels and j < nrays:
        weight=calculate_unorm_weight(voxels[i],rays[j],precision,voxel_radius)
        if weight > 0:
            pos = rayNHit[j]
            rayWeights[j,pos] = weight
            rayHits[j,pos] = i
            rayNHit[j] = pos + 1
            
@cuda.jit
def unorm_rotated_singleweight(rayNHit,rayHits,rayWeights,voxels,rays,nvoxels,nrays,voxel_radius,precision,q):
    i, j = cuda.grid(2)
    if i < nvoxels and j < nrays:
        weight=calculate_rotated_unorm_weight(voxels[i],rays[j],precision,voxel_radius,q)
        if weight > 0:
            pos = rayNHit[j]
            rayWeights[j,pos] = weight
            rayHits[j,pos] = i
            rayNHit[j] = pos + 1
            
@cuda.jit
def raynorm_singleweight(weights,rayNHit,nrays):
    j = cuda.grid(1)
    if j < nrays:
        normL1(weights[j,0:rayNHit[j]])

@cuda.jit
def singleproject(projection,voxels,rayNHit,rayHits,nrays):
    j = cuda.grid(1)
    if j < nrays:
        projection[j] = calculate_singleproject(voxels,rayHits[j,0:rayNHit[j]],rayNHit[j])
        
@cuda.jit
def singlerayproject(projection,voxels,rayNHit,rayHit):
    projection[0] = calculate_singleproject(voxels,rayHit[0:rayNHit[0]],rayNHit[0])
        
# int, int, int[maxlength], .....
@cuda.jit
def singleray_unorm_rotated_weight_kernel(rayNHit,rayHit,rayWeight,voxels,ray_y,ray_z,nvoxels,voxel_radius,precision,q):
    i = cuda.grid(1)
    if i < nvoxels:
        weight=calculate_singleray_rotated_unorm_weight(voxels[i],ray_y,ray_z,precision,voxel_radius,q)
        if weight > 0:
            pos = rayNHit[0]
            rayWeight[pos] = weight
            rayHit[pos] = i
            rayNHit[0] = pos + 1

##################################
#------ functions ---------------#
##################################
class raytracer:
    
    @staticmethod
    def raynorm_weights(rayWeights,rayNHit):
        nrays=len(rayWeights)
        blockRay = math.ceil(nrays / cudaOptions.maxthreadsperblock)
        raynorm_singleweight[blockRay, cudaOptions.maxthreadsperblock](rayWeights,rayNHit,nrays)
        
    @staticmethod
    def unorm_rayweight(voxels,rays,maximal_length,voxel_radius=math.sqrt(2),phi=0.0,alpha=0.0):
        nvoxels = len(voxels)
        nrays = len(rays)
        nweight = int(maximal_length)

        threadsperblock = (cudaOptions.maxthreadsper2Dblock,cudaOptions.maxthreadsper2Dblock)
        blockV = math.ceil(nvoxels / cudaOptions.maxthreadsper2Dblock)
        blockRay = math.ceil(nrays / cudaOptions.maxthreadsper2Dblock)
        blockspergrid = (blockV,blockRay)
        precision = cudaOptions.precision

        rayNHit = np.zeros(nrays,dtype=int)
        rayHits = np.empty((nrays,nweight),dtype=np.int_)
        rayWeights = np.empty((nrays,nweight),dtype=rayOptions.weight_dtype)
        
        # generate quaternion rotation
        q = np.zeros(shape=(4,))
        generateq[1,1](q,phi,alpha)
        print(q)
        
        # print(blockspergrid,threadsperblock)
        unorm_rotated_singleweight[blockspergrid, threadsperblock](rayNHit,rayHits,rayWeights,voxels,rays,nvoxels,nrays,voxel_radius,precision,q)
        return rayNHit,rayHits,rayWeights
    
    def __init__(self,voxel_size,method="OSEM",f_Nyquist=2,voxel_radius=math.sqrt(2)):
        self.method = algorithms.getMethodID(method)
        # prepare voxel coordinates
        self.voxels,_,self.nvoxels = voxel.generateEmptyVoxels(voxel_size,algorithms.getFillValue(method),voxel_radius)
        self.voxel_radius = voxel_radius
        # prepare camera
        self.f_Nyquist = f_Nyquist
        self.maximal_length,self.range_length,self.camera_size,self.camera_nrays,self.camera_range = pixel.generateCamera(f_Nyquist,voxel_size)
        self.camera_rays,self.rayverts = pixel.generateEmptyRays(self.camera_size,self.maximal_length,self.camera_nrays)
        # other
        self.precision=cudaOptions.precision
        self.maxthreadsperblock=cudaOptions.maxthreadsperblock

    def norm_raytrace(self,phi=0.0,alpha=0.0):
        rayNHit,rayHits,rayWeights=raytracer.unorm_rayweight(self.voxels,self.camera_rays,self.maximal_length,self.voxel_radius,phi,alpha)
        raytracer.raynorm_weights(rayWeights,rayNHit)
        return rayNHit,rayHits,rayWeights

    def rayproject(self,rayNHit,rayHits):
        nrays=len(rayNHit)
        blockRay = math.ceil(nrays / cudaOptions.maxthreadsperblock)
        projection = np.zeros(nrays,dtype=rayOptions.project_dtype)
        singleproject[blockRay, cudaOptions.maxthreadsperblock](projection,self.voxels,rayNHit,rayHits,nrays)
        return projection
    
    def reconstruct(self,rays,iterations=1,autosave=True,make_plot=True):
        nrays = len(rays)
        nweight = int(self.maximal_length)

        rayNHit = np.zeros(shape=(1,),dtype=np.int_)
        rayHit = np.empty(shape=(nweight,),dtype=np.int_)
        rayWeight = np.empty(shape=(nweight,),dtype=rayOptions.weight_dtype)
        calculated_projection = np.zeros(shape=(1,),dtype=rayOptions.project_dtype)
        projection_error = np.zeros(shape=(nrays,),dtype=rayOptions.error_dtype)
        
        blockV = math.ceil(self.nvoxels / self.maxthreadsperblock)
                
        for i in tqdm(range(nrays*iterations)):
            rayIndex = i%nrays
            rayNHit[0] = 0
            calculated_projection[0] = 0

            # get vertex from LOR
            rayvertex1 = np.ascontiguousarray(rays[rayIndex,0:3])
            rayvertex2 = np.ascontiguousarray(rays[rayIndex,4:7])

            # generate quaternion rotation
            q = np.zeros(shape=(4,))
            generateVQ_LOR[1,1](q,rayvertex1,rayvertex2)
            # print(q)

            # ray coordinates in screenspace
            rotateq[1,1](rayvertex1,q)
            ray_y,ray_z = rayvertex1[1],rayvertex1[2]

            singleray_unorm_rotated_weight_kernel[blockV,self.maxthreadsperblock](rayNHit,rayHit,rayWeight,self.voxels,ray_y,ray_z,self.nvoxels,self.voxel_radius,self.precision,q)
            nhits = rayNHit[0]
            hits = rayHit[0:nhits]
            weights = rayWeight[0:nhits]

            # normalize weights
            weights = weights/np.sum(weights)

            # print(hits)
            # print(weights)

            # calculate estimate
            singlerayproject[1,1](calculated_projection,self.voxels,rayNHit,rayHit) 

            if nhits > 0:
                projection_error[rayIndex] = algorithms.backpropagate(self.method,self.voxels,calculated_projection[0],weights,hits,nhits)
        
        self.save_voxels()
        if make_plot:
            plt.plot(projection_error)
        return projection_error
    
    def make_projection(self,phi=0.0,alpha=0.0,make_plot=True):
        rayNHit,rayHits,rayWeights = self.norm_raytrace(phi,alpha)
        projection = self.rayproject(rayNHit,rayHits)
        if make_plot:
            plt.figure(figsize=(12,12))
            plt.imshow(projection.reshape(self.camera_size),extent=np.array(self.camera_range).flatten())
            plt.colorbar()
        return rayNHit,rayHits,rayWeights,projection
    
    def load_voxels(self,path="voxelImage.npy"):
        self.voxels = np.load(path)
        
    def save_voxels(self,path="voxelImage.npy"):
        np.save(path,self.voxels)
            
        
        