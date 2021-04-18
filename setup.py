from setuptools import find_packages
from setuptools import setup

_VERSION = '0.1.0'

setup(
    name='pt-util',
    version=_VERSION,
    author='Jake Tuero',
    author_email='tuero@ualberta.ca',
    include_package_data=True,
    packages=find_packages(),
    install_requires=[
        'torch>=1.3.0',
        'gin-config'
    ],
    description='pt-util: A personal pytorch util library for training models',
    url='https://github.com/tuero/pt-util'
)
