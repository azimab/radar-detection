# vendor/pylib

Unpacked PyPI wheels, so the validator and visualizer can use the official nuScenes
devkit on machines where `pip` is not available (the portal nodes have no pip):

| package | version | why |
|---|---|---|
| nuscenes-devkit | 1.2.0 | the reference loader the dataset is validated against |
| pyquaternion | 0.9.9 | devkit dependency |
| cachetools | 7.2.0 | devkit dependency |
| shapely | 2.1.2 (cp310 manylinux x86_64) | devkit dependency |
| descartes | 1.1.0 | devkit dependency |

Only these scripts put it on `sys.path`; nothing is installed. Delete the directory and
`pip install nuscenes-devkit` instead wherever pip exists.
