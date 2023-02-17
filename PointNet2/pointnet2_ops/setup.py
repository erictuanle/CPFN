import os
import glob
import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

sources = glob.glob("cuda_ops/src/*.cpp") + glob.glob("cuda_ops/src/*.cu")
headers = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cuda_ops/include')

setuptools.setup(
    name="pointnet2_ops", 
    version="1.0",
    description="PointNet++ modules",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    ext_modules=[
        CUDAExtension(
            name='cuda_ops',
            sources=sources,
            extra_compile_args={
                "cxx": ["-O2", "-I{}".format(headers)],
                "nvcc": ["-O2", "-I{}".format(headers)],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)