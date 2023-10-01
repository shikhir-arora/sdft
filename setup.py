from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# Define Cython extension
cython_extension = Extension(
    name="cython_sdft_functions",
    sources=["cython_sdft_functions.pyx"],
    include_dirs=[numpy.get_include()],
    extra_compile_args=['-O3', '-g', '-march=native', '-funroll-loops']
)

setup(
    name='SDFT',
    version='0.1',
    description='SDFT',
    author='Shikhir Arora <shikhir@thesarogroup.com>',
    url='https://github.com/shikhir-arora/sdft',
    license='MIT',
    ext_modules=cythonize([cython_extension], annotate=True),
    setup_requires=[
        'numpy>=1.26.0',
        'cython>=3.0.2'
    ],
    install_requires=[
        'scipy>=1.11.3',
        'matplotlib>=3.8.0',
        'sounddevice>=0.4.6',
        'rich>=13.6.0'
    ]
)

