import platform
import sys

import numpy
import cupy
import torch
import dask 

print( "OS: ", platform.system(), platform.release() )
print( "Python version: ", sys.version.replace( "\n", "" ))
print()

print( "Numpy version: ", numpy.__version__ )
print( "Cupy version: ", cupy.__version__ )
print( "Torch version: ", torch.__version__ )
print( "Dask version: ", dask.__version__ ) 