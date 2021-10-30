import math
import numpy as np
class rayOptions:
        weight_dtype = np.float64
        project_dtype = np.float64
        error_dtype = np.float64
        data_directory = "pointdata.txt" # need to change - set to data directory!
class cudaOptions:
    maxthreadsperblock = 256 #set to recommended limit from https://forums.developer.nvidia.com/t/how-to-choose-how-many-threads-blocks-to-have/55529
    maxthreadsper2Dblock = 16
    precision = math.pow(10,-15)