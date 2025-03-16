from setuptools import setup, Extension
from Cython.Build import cythonize

setup(
    name='chan_fs_dll',
    ext_modules=cythonize([
        Extension("chan_fs_dll", ["wrapper.py"])
    ]),
)