import os
import setuptools
from setuptools import setup


here = os.path.abspath(os.path.dirname(__file__))
requires_list = []
with open(os.path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    for line in f:
        requires_list.append(str(line))

setup(
    name="pli",
    version="0.1",
    description="Implementation of `Pseudo-Likelihood Inference` and baseline SBI methods",
    author="Theo Gruner",
    author_email="theo.gruner@gmail.com",
    packages=setuptools.find_packages(),
    install_requires=requires_list,
)
