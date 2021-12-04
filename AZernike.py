# -*- coding: utf-8 -*-

print( f"Hello... Good morning!" )

use_gpu = 0
print( f"use_gpu = {use_gpu}" )

if use_gpu :
    print( f"import cupy as np" )

    import cupy as np
else :
    print( f"import numpy as np" )
    
    import numpy as np 
pass

import cv2 as cv, math
from time import *
from scipy.special import factorial
from matplotlib import pyplot as plt
from datetime import datetime
from tqdm.notebook import tqdm
from IPython.display import clear_output
from Profiler import *

complex_type = np.clongdouble
complex_type = np.cdouble
pi = np.pi

line = line1 = "*"*60 
line2 = "\n" + line + ""
line3 = line2 + "\n"

print( f"Importing python packages was done." )
print( f"time = {perf_counter_ns()}" )

def _rps( r_ps, rho, p_2s, hash, debug = 0 ) :
    rho_id = id( rho )
    
    p_2s = int( p_2s )
    
    key_all = f"rps:{p_2s}:{r_ps}"
    
    rho_power = None
    
    if key_all in hash :
        rho_power = hash[ key_all ] 
        
        return rho_power;
    pass

    if p_2s in hash :
        rho_power = hash[ p_2s ]
    else : 
        if p_2s in [ -2, -1, 0, 1, 2 ] :
            rho_power = np.power( rho, p_2s )
        else :
            rho_power = _rps( 1, rho, p_2s//2, hash=hash, debug = debug)
            
            if p_2s % 2 == 1 : 
                rho_power = rho_power*rho_power*rho
            else :
                rho_power = rho_power*rho_power
            pass
        pass
    
        hash[ p_2s ] = rho_power
    pass

    if r_ps not in [ 1, 1.0 ] :
        rho_power = r_ps*rho_power
    pass
    
    hash[ key_all ] = rho_power
    
    return rho_power
pass

@profile
def Rpq(p, q, rho, hash={}, debug = 0 ) :
    q = abs( q )
    
    if abs(q) > p : 
        print( f"Invalid argument, abs(q = {q}) < p(={p}) is not satisfied")
        return 
    pass

    if int(p - abs(q))%2 == 1 : 
        print( f"Invalid argument, p({p}) - q({q}) should be an even number.")
        return 
    pass

    key = f"rpq:{p}:{q}"
    
    if key in hash :
        return hash[ key ] 
    pass

    r_pq_rho = None 
    
    if p == 0 and q == 0 :
        r_pq_rho = np.ones_like( rho )
    elif p == 1 and q == 1 :
        r_pq_rho = rho
    elif p == 2 and q == 2 :
        r_pq_rho = rho*rho
    else :
        t = max( (p - q)/2, 0 )
        s = np.arange( 0, t + 1 )

        R_ps = np.power( -1, s )*factorial(p - s)/factorial(s)/factorial( (p + q)/2 - s)/factorial( (p - q)/2 - s )
        #R_ps = R_ps.astype( np.int_ )

        rho_power = []

        for r_ps, p_2s in zip( R_ps, p - 2*s ) :
            rho_power.append( _rps( r_ps, rho, p_2s, hash, debug=debug ) )
        pass

        rho_power = np.array( rho_power )

        r_pq_rho = np.sum( rho_power, axis=0 )
    pass
    
    hash[ key ] = r_pq_rho
        
    if debug : 
        print( line2 )
        print( f"p = {p}, q={q}, (p - |q|)/2 = {t}" )
        print( "s = ", s )
        print( "R_ps = ", R_ps )
        print( "rho_power shape = ", rho_power.shape )
        print( "rho_power.T shape = ", rho_power.T.shape )
        print( "rho_power = ", rho_power )
        print( "rho_power.T = ", rho_power.T )
        print( "R_pq_rho = ", r_pq_rho )    
        #print( "R_sum = ", R_sum )
        print( line2 )
    pass
    
    return r_pq_rho
pass

@profile
def Vpq( p, q, rho, theta, hash={}, debug = 0 ) :    
    q = int(q)
    
    use_hash = False
    
    key = f"v:{p}:{q}"
    
    if use_hash and key in hash :
        return hash[ key ]
    pass
    
    v_pq = None 
    
    if q < 0 :
        v_pq = Vpq( p, -q, rho, theta, hash=hash, debug=debug)
        
        v_pq = v_pq.real - 1j*v_pq.imag
    else :
        r_pq = Rpq( p, q, rho, hash=hash, debug = 0 )
    
        #v_pq = r_pq*np.exp( 1j*q*theta )
        
        v_pq = r_pq
    
        if q : 
            v_pq = v_pq*np.exp( (1j*q)*theta )
        pass
    pass

    if use_hash :
        hash[ key ] = v_pq
    
    if debug : 
        print( f"Vpq({p}, {q}) = ", v_pq )
    pass

    return v_pq
pass

@profile
def rho_theta( img, debug = 0 ) :
    h = img.shape[0]
    w = img.shape[1]
    
    mwh = max( h - 1, w -1 )
    radius = math.sqrt( 2*mwh*mwh )
    
    debug and print( f"H = {h}, W = {w}, r = {radius}" )
    
    y, x = np.indices( img.shape )

    if not use_gpu: 
        np.set_printoptions(suppress=1)

    if debug : 
        print( "x = ", x )
        print( "y = ", y )
    pass

    y = (y/mwh*2 - 1.0).flatten()
    x = (x/mwh*2 - 1.0).flatten()
    
    dx = 2.0/max(h, w)
    dy = dx
    
    #y = y*(1 - dy/2.0)
    #x = x*(1 - dx/2.0)
    
    if debug : 
        print( "x = ", x )
        print( "y = ", y )
    pass
    
    rho_square = np.sqrt( x**2 + y**2 )
    
    k = np.where( rho_square <= 1.0 )
    
    y = y[k]
    x = x[k]
    
    rho_square = rho_square[k]
    
    if debug : 
        print( "x[k] = ", x )
        print( "y[k] = ", y )
    pass

    rho = np.sqrt( rho_square )
    theta = np.arctan2( y, x )
    
    return rho, theta, x, y, dx, dy
pass

def print_curr_time() :
    # 현재 시각 출력 
    print("Current Time =", datetime.now().strftime("%H:%M:%S") )
pass

print( "Zernike functions are defined.")
print_curr_time()
