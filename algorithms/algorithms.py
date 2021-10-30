import math
import numpy as np
from numba import cuda,types, from_dtype,jit
from raytracer.cudaOptions import rayOptions
from raytracer.cudaOptions import cudaOptions
from raytracer.rotation.quaternion import *
from tqdm import tqdm

# note if you are using Python version 3.10 or greater, use pattern matching instead of the if statements... 

class algorithms:
    @staticmethod
    def getMethodID(method):
        fillValues = {
        "ART": 0,
        "OSEM": 1,
        }
        return fillValues.get(method,0)
    
    # voxel fill value
    @staticmethod
    def getFillValue(methodID):
        fillValues = {
        0: 0,
        1: 1,
        }
        return fillValues.get(methodID,0)
    
    @staticmethod
    def backpropagate(methodID,voxels,calculated,weights,hits,nhits):
        # backpropagate error
        blockW = math.ceil(nhits/cudaOptions.maxthreadsperblock)
        
        if methodID==0:
            err = algorithms.error_ART(calculated)
            backpropagate_ART[blockW,cudaOptions.maxthreadsperblock](voxels,err,weights,hits,nhits)
        elif methodID==1:
            err = algorithms.error_OSEM(calculated)
            backpropagate_ART[blockW,cudaOptions.maxthreadsperblock](voxels,err,weights,hits,nhits)
        else: 
            err = algorithms.error_ART(calculated)
            backpropagate_ART[blockW,cudaOptions.maxthreadsperblock](voxels,err,weights,hits,nhits)
            
        return err
    
    ########################
    ## Error Functions #####
    ########################
    @staticmethod
    @jit(nopython=True)
    def error_ART(calculated):
        return 1-calculated
    @staticmethod
    @jit(nopython=True)
    def error_OSEM(calculated):
        if calculated != 0.0:
            return 1/calculated
        else:
            return 1

########################
## CUDA KERNELS ########
########################

@cuda.jit
def backpropagate_ART(voxels,err,weights,hits,nhits):
    i = cuda.grid(1)
    if i < nhits:
        index = hits[i]
        value = voxels[index].r
        weight = weights[i]
        voxels[index].r = value + err*weight
        
@cuda.jit
def backpropagate_OSEM(voxels,err,weights,hits,nhits):
    i = cuda.grid(1)
    if i < nhits:
        index = hits[i]
        value = voxels[index].r
        # weight = weights[i]
        voxels[index].r = value*err
    