# pytorch imaginary number test

import numpy
import torch

Ks = torch.arange( 1,  6 + 1, 1 )
print( "is_scalar = ", num( Ks[0] ) )

print( [""] * 4 )

import torch

use_gpu = 1
device_name = "cuda" if use_gpu else "cpu"
device = torch.device( device_name)

print( "device type = ", type( device ) )

a = torch.rand( (2, 2), dtype=torch.cdouble, device=device )
b = a.to( device )
c = a.cpu()

print( a )
print( a.data_ptr() == b.data_ptr() )
print( a.data_ptr() == c.data_ptr() )

import scipy 
print( "scipy version = ", scipy.__version__ )
from scipy.special import factorial

print( "factorial(0) = ", factorial(0) )

import os

src_dir = os.path.dirname( os.path.abspath(__file__) )

print( "os.path.basename = ", os.path.basename( __file__ ) )
print( "os.path.basename = ", os.path.basename( __file__ ) )


from pathlib import Path
print( "Path( __file__ ).stem = ", Path( __file__ ).stem) 
print( "Path( __file__ ).name = ", Path( __file__ ).name)

