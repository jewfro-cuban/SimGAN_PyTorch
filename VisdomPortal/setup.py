import os

from setuptools import find_packages
from setuptools import setup


def read(file_name):
    with open(os.path.join(os.path.dirname(__file__), file_name), 'r') as f:
        return f.readlines()

setup(
    name='visdom-portal',
    version='0.1.0',
    description='A simple tool for using PyTorch and Visdom.',
    # long_description=read('README.md'),
    url='https://github.com/automan000/VisdomPortal',
    author='automan',
    keywords='',
    packages=find_packages(exclude=[]))