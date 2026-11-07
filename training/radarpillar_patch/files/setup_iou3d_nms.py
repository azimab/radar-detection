"""Build only the rotated IoU/NMS extension needed by PointPillar."""

import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


MODULE = "pcdet.ops.iou3d_nms"
SOURCES = [
    "src/iou3d_cpu.cpp",
    "src/iou3d_nms_api.cpp",
    "src/iou3d_nms.cpp",
    "src/iou3d_nms_kernel.cu",
]


setup(
    name="radarpillar_iou3d_nms",
    ext_modules=[
        CUDAExtension(
            name=f"{MODULE}.iou3d_nms_cuda",
            sources=[os.path.join(*MODULE.split("."), source) for source in SOURCES],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
