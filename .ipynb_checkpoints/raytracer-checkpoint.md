# raytracer-imaging
For PET image reconstruction
##
Directory:
    cudaOptions.py
    rayOptions.py
    raytracer.py
    rotation/
        quaternion.py
        radon.py
    raytrace/
        voxel.py
        raytrace.py
    algorithms/
        art/art.py
        fbp/fbp.py
        mlem/mlem.py
        osem/osem.py
##
Dependencies:
matplotlib
tqdm
numba=54.1
CUDA=11.4.2