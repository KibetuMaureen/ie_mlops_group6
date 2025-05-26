from setuptools import setup, find_packages

setup(
    name="ie_mlops_group6",
    version="0.1.0",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.6",
)

