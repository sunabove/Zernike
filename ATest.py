# pytorch imaginary number test

import torch

use_gpu = 1
device_name = "cuda" if use_gpu else "cpu"
device = torch.device( device_name)

a = torch.rand( (2, 2), dtype=torch.cdouble, device=device )
print( a )

import scipy 
print( "scipy version = ", scipy.__version__ )
from scipy.special import factorial

print( "factorial(0) = ", factorial(0) )
