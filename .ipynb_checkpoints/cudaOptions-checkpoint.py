import numpy as np
class rayOptions:
        weight_dtype = np.float64
class cudaOptions:
    maxthreadsperblock = 256 #set to recommended limit from https://forums.developer.nvidia.com/t/how-to-choose-how-many-threads-blocks-to-have/55529
    maxthreadsper2Dblock = 16