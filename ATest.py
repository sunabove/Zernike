# pytorch imaginary number test


print( [""] * 4 )

import torch

use_gpu = 1
device_name = "cuda" if use_gpu else "cpu"
device = torch.device( device_name)

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
