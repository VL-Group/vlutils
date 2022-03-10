import os
from setuptools import setup
from distutils.util import strtobool


INSTALL_REQUIRES = [
    "torch>=1.9",
    "tqdm",
    "rich",
    "nvidia-ml-py3"
] if not strtobool(os.getenv("CONDA_BUILD", "false")) else []

setup(install_requires=INSTALL_REQUIRES)
