from distutils.core import setup
from Cython.Build import cythonize

setup(
    name='jps_c',
    ext_modules = cythonize("jps_c.pyx")
)