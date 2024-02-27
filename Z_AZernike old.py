# -*- coding: utf-8 -*-

import numpy as np, cv2 as cv, math
from scipy.special import factorial
from matplotlib import pyplot as plt
from datetime import datetime
from tqdm.notebook import tqdm
from IPython.display import clear_output
from AProfiler import *

line = line1 = "*"*60 
line2 = "\n" + line + ""
line3 = line2 + "\n"

print( f"Hello... Good morning!" )
print( f"Importing python packages was done." )

@profile
def Rpq(p, q, rho, debug = 0 ) :
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
    rho_power = []
    
    for p_2s in p - 2*s :
        rho_power.append( np.power(rho, p_2s ) )
    pass

    rho_power = np.array( rho_power )
    
    R_pq_rho = np.inner( rho_power.T, R_ps )
    
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
def Vpq( p, q, x, y, debug = 0 ) :
    rho = np.sqrt( x**2 + y**2 )
    
    R_pq = Rpq( p, q, rho, debug = 0 )
    
    V_pq = R_pq 
    
    if q : 
        V_pq = R_pq*np.exp( 1j*q*np.arctan2( y, x ) );
    pass    
    
    #print( "rho = ", rho )
    if debug : 
        print( f"V_pq({p}, {q}) = ", V_pq )
    pass

    return V_pq
pass

print( "Zernike functions are defined.")

current_time = datetime.now().strftime("%H:%M:%S")
print("Current Time =", current_time)

