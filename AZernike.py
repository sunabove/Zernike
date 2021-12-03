# -*- coding: utf-8 -*-

import numpy as np, cv2 as cv, math
from time import *
from scipy.special import factorial
from matplotlib import pyplot as plt
from datetime import datetime
from tqdm.notebook import tqdm
from IPython.display import clear_output
from Profiler import *

line = line1 = "*"*60 
line2 = "\n" + line + ""
line3 = line2 + "\n"

print( f"Hello... Good morning!" )
print( f"Importing python packages was done." )
print( f"time = {perf_counter_ns()}" )

def _rps( r_ps, rho, p_2s, hash, debug = 0 ) :
    rho_id = id( rho )
    
    key_all = f"{p_2s}:{r_ps}"
    
    rho_power = None
    
    if key_all in hash :
        rho_power = hash[key_all] 
        
        return rho_power;
    pass

    if p_2s in hash :
        rho_power = hash[ p_2s ]
    else : 
        if p_2s in [ 0, 1, 2 ] :
            rho_power = np.power( rho, p_2s )
        else :
            rho_power = _rps( 1, rho, p_2s//2, hash, debug = debug)
            
            if p_2s % 2 == 1 : 
                rho_power = rho_power*rho_power*rho
            else :
                rho_power = rho_power*rho_power
            pass
        pass
    pass

    if r_ps not in [ 1, 1.0 ] :
        rho_power = r_ps*rho_power
    pass
    
    hash[ key_all ] = rho_power
    
    return rho_power
pass

@profile
def Rpq(p, q, rho, hash={}, debug = 0 ) :
    if abs(q) > p : 
        print( f"Invalid argument, abs(q = {q}) < p(={p}) is not satisfied")
        return 
    pass

    if int(p - abs(q))%2 == 1 : 
        print( f"Invalid argument, p({p}) - q({q}) should be an even number.")
        return 
    pass

    q = abs( q )

    t = max( (p - q)/2, 0 )
    s = np.arange( 0, t + 1 )
    
    R_ps = np.power( -1, s )*factorial(p - s)/factorial(s)/factorial( (p + q)/2 - s)/factorial( (p - q)/2 - s )
    R_ps = R_ps.astype( np.int_ )

    rho_power = []
    
    for r_ps, p_2s in zip( R_ps, p - 2*s ) :
        rho_power.append( _rps( r_ps, rho, p_2s, hash, debug=debug ) )
    pass

    rho_power = np.array( rho_power )
    
    R_pq_rho = np.sum( rho_power, axis=0 )
    
    #R_sum = np.sum( R_pq_rho )
        
    if debug : 
        print( line2 )
        print( f"p = {p}, q={q}, (p - |q|)/2 = {t}" )
        print( "s = ", s )
        print( "R_ps = ", R_ps )
        print( "rho_power shape = ", rho_power.shape )
        print( "rho_power.T shape = ", rho_power.T.shape )
        print( "rho_power = ", rho_power )
        print( "rho_power.T = ", rho_power.T )
        print( "R_pq_rho = ", R_pq_rho )    
        #print( "R_sum = ", R_sum )
        print( line2 )
    pass
    
    return R_pq_rho
pass

@profile
def Vpq( p, q, rho, theta, hash={}, debug = 0 ) :
    R_pq = Rpq( p, q, rho, hash=hash, debug = 0 )
    
    V_pq = R_pq 
    
    if q : 
        V_pq = R_pq*np.exp( 1j*q*theta )
    pass    
    
    #print( "rho = ", rho )
    if debug : 
        print( f"V_pq({p}, {q}) = ", V_pq )
    pass

    return V_pq
pass

@profile
def rho_theta( img, debug = 0 ) :
    h = img.shape[0]
    w = img.shape[1]
    
    mwh = max( h, w )
    radius = math.sqrt( 2*mwh*mwh )
    
    debug and print( f"H = {h}, W = {w}, r = {radius}" )
    
    x, y = np.where( img >= 0 )

    np.set_printoptions(suppress=1)

    if debug : 
        print( "x = ", x )
        print( "y = ", y )
    pass

    x = x/(mwh/math.sqrt(2)) - 1.0/math.sqrt(2)
    y = y/(mwh/math.sqrt(2)) - 1.0/math.sqrt(2)

    if debug : 
        print( "x = ", x )
        print( "y = ", y )
    pass

    rho = np.sqrt( x**2 + y**2 )
    theta = np.arctan2( y, x )
    
    return rho, theta, x, y
pass

def print_curr_time() :
    # 현재 시각 출력 
    print("Current Time =", datetime.now().strftime("%H:%M:%S") )
pass

print( "Zernike functions are defined.")
print_curr_time()
