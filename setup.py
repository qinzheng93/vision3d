import os.path as osp
from glob import glob

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


extension_dir = "vision3d/csrc"


def get_sources():
    sources = []
    sources += glob(osp.join(extension_dir, "cpu", "*", "*.cpp"))
    sources += glob(osp.join(extension_dir, "cuda", "*", "*.cpp"))
    sources += glob(osp.join(extension_dir, "cuda", "*", "*.cu"))
    sources += glob(osp.join(extension_dir, "pybind.cpp"))
    return sources


def get_include_dirs():
    include_dirs = []
    include_dirs.append(osp.abspath(osp.join(extension_dir, "external", "eigen3")))
    include_dirs.append(osp.abspath(osp.join(extension_dir, "external", "nanoflann")))
    include_dirs.append(osp.abspath(osp.join(extension_dir, "common")))
    return include_dirs


def get_requirements():
    with open("requirements.txt", "r") as f:
        lines = f.readlines()
    requirements = [line.strip() for line in lines]
    return requirements


setup(
    name="vision3d",
    version="2.0.1",
    install_requires=get_requirements(),
    ext_modules=[
        CUDAExtension(
            name="vision3d.ext",
            sources=get_sources(),
            include_dirs=get_include_dirs(),
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
