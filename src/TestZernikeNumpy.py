# -*- coding: utf-8 -*-

import numpy as np
from time import time, sleep
from math import atan2, log10
from numpy import cos, sin, conjugate, sqrt

import sys, logging as log
log.basicConfig(stream=sys.stdout, format='%(levelname)s %(filename)s:%(lineno)04d %(message)s', datefmt='%Y-%m-%d:%H:%M:%S', level=log.INFO )

import cv2
import pylab as pl
from pylab import gray 
from matplotlib import cm
import matplotlib.pyplot as plt

from skimage import data
from skimage import color
from skimage.transform import rescale 
import mahotas

# function define

facts = { } 

def factorial(n) :
    if n <= 1 :
        return 1 ; 
    pass

    if n in facts :
        return facts[n]
    pass

    f = n*factorial(n -1) 
    facts[n] = f;

    return f
pass # -- factorial

def polar(r, theta):
    x = r * cos(theta)
    y = r * sin(theta)

    return 1.*x+1.j*y
pass # -- polar

def _slow_zernike_poly(Y, X, n, l):
    
    y, x = Y[0], X[0]
    vxy = np.zeros(Y.size, dtype=complex)
    
    for index, (x, y) in enumerate( zip(X,Y) ):
        Vnl = 0.
        
        for m in range( int( (n-l)//2 ) + 1 ):
            Vnl += (-1.)**m *factorial(n-m) /  \
                ( factorial(m) * factorial((n - 2*m + l) // 2) * factorial((n - 2*m - l) // 2) ) * \
                ( sqrt(x*x + y*y)**(n - 2*m) * polar(1.0, l*atan2(y,x)) )
        pass
    
        vxy[index] = Vnl
    pass

    return vxy
pass # -- _slow_zernike_poly

def zernike_reconstruct(img, d):
    shape = img.shape
    
    radius = max( img.shape[0]/2.0, img.shape[1]/2.0 )
    
    row = shape[0]
    col = shape[1]
    
    idx = np.ones(shape)
    
    cofy = row/2.0
    cofx = col/2.0
    
    print( f"The image reconstruction order = { d }, radius = {radius}")

    y, x = np.where(idx > 0)
    p = img[y, x].ravel()
    
    yn = ( (y -cofy)/radius ).ravel()
    xn = ( (x -cofx)/radius ).ravel()

    k = (np.sqrt(xn**2 + yn**2) <= 1.)
    frac_center = np.array(p[k], np.double)
    
    yn = yn[k]
    xn = xn[k]
    
    frac_center = frac_center.ravel()

    # in the discrete case, the normalization factor is not pi but the number of pixels 
    # within the unit disk
    
    npix = float(frac_center.size)

    reconstr = np.zeros(img.size, dtype=complex)
    accum = np.zeros(yn.size, dtype=complex)

    for n in range(d + 1):
        for l in range(n+1):
            if (n-l)%2 == 0:
                # get the zernike polynomial
                vxy = _slow_zernike_poly(yn, xn, float(n), float(l))
                # project the image onto the polynomial and calculate the moment
                a = sum(frac_center * conjugate(vxy)) * (n + 1)/npix
                # reconstruct
                accum += a * vxy
            pass
        pass
    pass

    reconstr[k] = accum
    
    return reconstr
pass # -- zernike_reconstruct

def test_zernike_numpy( is_jupyter = 1 ) :
    print( "Calculating moments ...." )
    
    from TestZernike import save_image
    
    img_name = "lena"
    img = mahotas.demos.load( img_name )
    
    save_image(img, f"{img_name}_org.png")
    
    img = color.rgb2gray( img )
    
    log.info( f"image shape = {img.shape}" )
    
    rescale_width = 50 
    
    if rescale_width :
        scale = rescale_width/img.shape[1]
        img = rescale(img, scale, anti_aliasing=True)
        save_image(img, f"{img_name}_size_{rescale_width}.png")
        
        log.info( f"image shape = {img.shape}" )
    pass
    
    if is_jupyter : 
        plt.imshow( img, cmap="gray" )
        plt.show()
    pass
    
    row_cnt = 1
    col_cnt = 5
    cnt = row_cnt*col_cnt
    
    gmax = np.max( img )
    
    print( f"gmax = {gmax}" )
    
    for idx, d in enumerate( [10, 20, 40 ] ) : 
        then = time()
        
        reconst = zernike_reconstruct(img, d)
    
        reconst = reconst.reshape( img.shape )
        
        now = time()
        
        elapsed = now - then
        print( f"Time elapsed = {elapsed:.3f}" )
        
        #img_reconst_abs = np.absolute( reconst )
        img_reconst = reconst.real
        
        img_diff = img - img_reconst
        
        gmax = np.max( img_reconst ) # 복원된 이미지의 회색조 최대값 
        
        mse = np.sum( np.square( img_diff ) )/(img_diff.shape[0]*img_diff.shape[1])
        
        psnr = 10*log10(gmax*gmax/mse)
        
        title = f"ord = {d}, gmax = {gmax}, mse = {mse}, psnr = {psnr:.2f}"
        fileName = f"{title}.png"
        
        print( title )
        
        save_image(img_reconst, fileName)
        
        if is_jupyter : 
            plt.imshow( img_reconst, cmap="gray", origin = 'upper')
            plt.xlabel( f'ord = {d}, psnr = {psnr}', fontsize=8)
            plt.show()
        pass
    pass
pass

if __name__ == '__main__' : 
    test_zernike_numpy( is_jupyter = 0)    
    print( "Good bye!" )
pass
