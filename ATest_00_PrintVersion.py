import platform, sys

import numpy
import torch 

print( "OS: ", platform.system(), platform.release() )
print( "Python version: ", sys.version.replace( "\n", "" ))
print( "Numpy version: ", numpy.__version__ )
print( "Torch version: ", torch.__version__ )

