from setuptools import find_packages
from setuptools import setup

_VERSION = '1.0.0'

setup(
    name='ptutil',
    version=_VERSION,
    author='Jake Tuero',
    author_email='tuero@ualberta.ca',
    include_package_data=True,
    packages=find_packages(),
    install_requires=[
        'torch>=1.3.0',
        'gin-config'
    ],
    description='ptutil: A personal pytorch util library for training models',
    url='https://github.com/tuero/ptutil'
)
