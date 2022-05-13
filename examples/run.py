import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import math
import numpy as np
from numba import cuda,types, from_dtype
import raytracer.cudaOptions
from raytracer.rotation.quaternion import *
from raytracer.raytracer.voxel import *
from raytracer.raytracer.raytrace import *

#world_size = np.array([400,400,1000])
voxel_scale = 10
voxel_size = np.array([40,40,100]) #odd so we have center point
raytracerA = raytracer(voxel_size,voxel_scale,method="ART")
#raytracerA.load_voxels()

raytracerA.load_data()

projection_error = raytracerA.reconstruct(nrays=100,iterations=1)

raytracerA.make_projection(phi=0.0*np.pi/180.0,alpha=0.0*np.pi/180.0)