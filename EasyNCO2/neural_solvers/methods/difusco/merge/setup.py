# python setup.py build_ext --inplace

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy
"""
Installation

To install the cython_merge module, run the following command:

python setup.py build_ext --inplace
"""

ext = Extension("cython_merge", ["cython_merge.pyx"],
                include_dirs=[numpy.get_include()])

setup(ext_modules=[ext], cmdclass={'build_ext': build_ext})
