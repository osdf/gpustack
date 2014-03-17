#! /usr/bin/env python
# -*- coding: utf-8 -*-


__author__ = 'Christian Osendorfer, osendorf@in.tum.de'


from setuptools import setup, find_packages


setup(
    name="gpustack",
    keywords="python, cuda, numpy, deep neural networks",
    packages=find_packages(exclude=['examples', 'docs']),
    include_package_data=True,
)

