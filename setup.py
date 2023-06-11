from setuptools import setup, find_packages
from pkg_resources import parse_requirements

with open('requirements.txt') as f:
    requirements = [str(req) for req in parse_requirements(f)]


setup(
    name='three_d_data_manager',
    version='0.1.0',
    packages=find_packages(),
    install_requires=requirements,
    author='Shahar Zuler',
    author_email='shahar.zuler@gmail.com',
    description='A package for managing 3D data',
    url='https://github.com/shaharzuler/three_d_data_manager',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
    ],
)