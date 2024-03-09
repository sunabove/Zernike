import os
import cv2
import pylab as pl
import scipy

from matplotlib import cm
from math import atan2

#import numpy as np
#from numpy import cos, sin, conjugate, sqrt

import cupy as np
from cupy import cos, sin, conjugate, sqrt
import numpy, cupy

def _polar(r, theta):
    x = r * cos(theta)
    y = r * sin(theta)

    return 1.*x + 1.j*y
pass

def _factorial( n ):    
    fact = scipy.special.factorial( cupy.asnumpy( n ) )

    fact = cupy.array( fact )

    return fact

    #return n * _factorial(n - 1)
pass

def _zernike_poly(Y, X, n, l):

    y, x = Y[0], X[0]

    vxy = np.zeros(Y.size, dtype=complex)

    m = np.arange( 0, int( (n-l)//2 ) + 1 )

    fact = (-1.)**m*_factorial(n-m)/( _factorial(m)*_factorial((n - 2*m + l) // 2)*_factorial((n - 2*m - l) // 2) )

    for index, [x, y] in enumerate( zip(X, Y) ) :        
        Vnl =  fact * ( sqrt(x*x + y*y)**(n - 2*m) * _polar(1.0, l*atan2(y, x)) )
        
        Vnl = np.sum( Vnl )
    
        vxy[index] = Vnl
    pass

    return vxy
pass # _slow_zernike_poly

def zernike_reconstruct(img, radius, D, cof):
    idx = np.ones(img.shape)

    cofy, cofx = cof
    cofy = float(cofy)
    cofx = float(cofx)
    radius = float(radius)    

    Y, X = np.where(idx > 0)
    P = img[Y, X].ravel()
    Yn = ( (Y -cofy)/radius).ravel()
    Xn = ( (X -cofx)/radius).ravel()

    k = (np.sqrt(Xn**2 + Yn**2) <= 1.)
    frac_center = np.array(P[k], np.double)

    Yn = Yn[k]
    Xn = Xn[k]
    frac_center = frac_center.ravel()

    # in the discrete case, the normalization factor is not pi 
    # but the number of pixels within the unit disk
    npix = float(frac_center.size)

    reconstr = np.zeros(img.size, dtype=complex)
    accum = np.zeros(Yn.size, dtype=complex)

    for n in range( D + 1 ):
        for l in range( n + 1 ):
            if (n - l)%2 == 0:
                # get the zernike polynomial
                vxy = _zernike_poly(Yn, Xn, float(n), float(l))
                # project the image onto the polynomial and calculate the moment
                a = sum(frac_center * conjugate(vxy)) * (n + 1)/npix
                # reconstruct
                accum += a * vxy
            pass
        pass
    pass

    reconstr[k] = accum

    return reconstr
pass # zernike_reconstruct

def test_zernike_reconstruct( img, D=12 ) :

    import cupy

    img = np.array( img + 0 )

    rows, cols = img.shape
    radius = cols//2 if rows > cols else rows//2

    reconst = zernike_reconstruct(img, radius, D, (rows/2., cols/2.))

    reconst = ( reconst + 0 ).reshape( img.shape )

    pl.figure(1)
    pl.imshow( cupy.asnumpy( img ), cmap="gray", origin = 'upper')
    pl.figure(2)    
    pl.imshow( cupy.asnumpy( reconst.real ), cmap="gray", origin = 'upper')

pass # test_zernike_reconstruct

if __name__ == '__main__':
    src_dir = os.path.dirname( os.path.abspath(__file__) )

    img = cv2.imread( f'{src_dir}/image/f.png', 0 )
    
    test_zernike_reconstruct( img )
pass