from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='pareconv',
    version='1.0.0',
    ext_modules=[
        CUDAExtension(
            name='pareconv.ext',
            sources=[
                'pareconv/extensions/extra/cloud/cloud.cpp',
                'pareconv/extensions/cpu/grid_subsampling/grid_subsampling.cpp',
                'pareconv/extensions/cpu/grid_subsampling/grid_subsampling_cpu.cpp',
                'pareconv/extensions/cpu/radius_neighbors/radius_neighbors.cpp',
                'pareconv/extensions/cpu/radius_neighbors/radius_neighbors_cpu.cpp',
                'pareconv/extensions/pybind.cpp',
            ],
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
