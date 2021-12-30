# -*- coding: utf-8 -*-

print( f"Hello... Good morning!" )

use_gpu = 1
use_numpy = not use_gpu
use_cupy = use_gpu

print( f"use_gpu = {use_gpu}" )

if use_gpu :
    print( f"import cupy as np" )

    import cupy as np
else :
    print( f"import numpy as np" )

    import numpy as np 
pass

import numpy, cupy, igpu

import cv2 as cv, math
from time import *
from scipy.special import factorial
from matplotlib import pyplot as plt
from datetime import datetime
from tqdm.notebook import tqdm
from IPython.display import clear_output
from Profiler import *

if use_numpy :
    complex_type = np.clongdouble
    complex_type = np.cdouble
pass

pi = numpy.pi

line = line1 = "*"*60 
line2 = "\n" + line + ""
line3 = line2 + "\n"

print( f"Importing python packages was done." )
print( f"time = {perf_counter_ns()}" )

def _rps( r_ps, rho, p_2s, hash, use_hash = 0, debug = 0 ) :
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
    
        if use_hash : 
            hash[ p_2s ] = rho_power
        pass
    pass

    if r_ps not in [ 1, 1.0 ] :
        rho_power = r_ps*rho_power
    pass
    
    if use_hash : 
        hash[ key_all ] = rho_power
    pass
    
    return rho_power
pass

@profile
def _pqs_facotrial(p, q, s):
    if use_gpu :
        p = cupy.asnumpy( p )
        s = cupy.asnumpy( s )
        
        R_ps = np.power( -1, np.array(s) )*np.array( factorial(p - s)/factorial(s)/factorial( (p + q)/2 - s)/factorial( (p - q)/2 - s ) )
        
        return R_ps
    else :
        R_ps = np.power( -1, s )*factorial(p - s)/factorial(s)/factorial( (p + q)/2 - s)/factorial( (p - q)/2 - s )
        
        return R_ps
    pass
pass

@profile
def Rpq(p, q, rho, hash={}, use_hash=1, debug = 0 ) :
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
    
    if use_hash and key in hash :
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

        R_ps = _pqs_facotrial( p, q, s )

        rho_power = []

        for r_ps, p_2s in zip( R_ps, p - 2*s ) :
            rho_power.append( _rps( r_ps, rho, p_2s, hash, debug=debug ) )
        pass

        rho_power = np.array( rho_power )

        r_pq_rho = np.sum( rho_power, axis=0 )
    pass
    
    if use_hash : 
        hash[ key ] = r_pq_rho
    pass
        
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
def Vpq( p, q, rho, theta, hash={}, use_hash=0, debug = 0 ) :    
    q = int(q)
    
    key = f"v:{p}:{q}"
    
    if use_hash and key in hash :
        return hash[ key ]
    pass
    
    v_pq = None 
    
    if use_hash and q < 0 :
        v_pq = Vpq( p, -q, rho, theta, hash=hash, debug=debug)
        
        v_pq = v_pq.real - 1j*v_pq.imag
    else :
        r_pq = Rpq( p, q, rho, hash=hash, debug = 0 )

        v_pq = r_pq
    
        if q : 
            q_theta = None

            if not use_hash :
                q_theta = np.exp( (1j*q)*theta )
            else :
                q_theta_key = f"theta:{q}"
                
                if q_theta_key in hash :
                    q_theta = hash[ q_theta_key ]
                else :
                    q_theta = np.exp( (1j*q)*theta )
                    hash[ q_theta_key ] = q_theta
                pass
            pass
            
            v_pq = v_pq*q_theta

            if not use_hash :
                del q_theta 
            pass
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
    
    y, x = np.where( img >= 0 ) 

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

def print_cpu_info() :
    import platform, psutil
    
    print("="*40, "CPU Details", "="*40)

    print(f"Processor type: {platform.processor()}")
    #Operating system release
    print(f"Operating system release: {platform.release()}")
    #Operating system version
    print(f"Operating system version: {platform.version()}")

    print()
    #Physical cores
    print(f"Number of physical cores: {psutil.cpu_count(logical=False)}")
    #Logical cores
    print(f"Number of logical cores: {psutil.cpu_count(logical=True)}")
    #Current frequency
    print(f"Current CPU frequency: {psutil.cpu_freq().current/1000:.2f} GHz")
    #Min frequency
    print(f"Min CPU frequency: {psutil.cpu_freq().min/1000} GHz")
    #Max frequency
    print(f"Max CPU frequency: {psutil.cpu_freq().max/1000} GHz")

    print()
    print("="*40, "Memory Details", "="*40)
    #Total RAM
    print(f"Total RAM installed: {round(psutil.virtual_memory().total/10**9, 2)} GB")
    #Available RAM
    print(f"Available RAM: {round(psutil.virtual_memory().available/10**9, 2)} GB")
    #Used RAM
    print(f"Used RAM: {round(psutil.virtual_memory().used/10**9, 2)} GB")
    #RAM usage
    print(f"RAM usage: {psutil.virtual_memory().percent}%")
pass # -- print_cpu_info

def print_gpu_info() :
    import GPUtil
    from tabulate import tabulate
    print("="*40, "GPU Details", "="*40)
    gpus = GPUtil.getGPUs()
    list_gpus = []
    for gpu in gpus:
        # get the GPU id
        gpu_id = gpu.id
        # name of GPU
        gpu_name = gpu.name
        # get % percentage of GPU usage of that GPU
        gpu_load = f"{gpu.load*100}%"
        # get free memory in MB format
        gpu_free_memory = f"{gpu.memoryFree/1_000:_} GB"
        # get used memory
        gpu_used_memory = f"{gpu.memoryUsed/1_000:_} GB"
        # get total memory
        gpu_total_memory = f"{gpu.memoryTotal/1_000:_} GB"
        # get GPU temperature in Celsius
        gpu_temperature = f"{gpu.temperature} °C"
        gpu_uuid = gpu.uuid
        list_gpus.append((
            gpu_id, gpu_name, gpu_load, gpu_free_memory, gpu_used_memory,
            gpu_total_memory, gpu_temperature 
        ))
    print(tabulate(list_gpus, headers=("id", "name", "load", "free memory", "used memory", "total memory", "temperature" )))
pass # -- print_gpu_info 

print( "Zernike functions are defined.")
print_curr_time()
