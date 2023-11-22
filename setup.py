from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='mistify',
    version='0.0.1',
    packages=[
        'mistify', 'mistify.fuzzify', 
        'mistify.infer', 'mistify.wrap', 
        'mistify.utils', 'mistify.process', 
        'mistify.functional'
    ]
)
