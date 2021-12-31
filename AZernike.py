# -*- coding: utf-8 -*-

print( f"Hello... Good morning!" )

import numpy, cupy
#import numpy as np, cupy as cp
import igpu, math, cv2 as cv 

from time import *
from scipy.special import factorial
from matplotlib import pyplot as plt
from datetime import datetime
from tqdm.notebook import tqdm
from IPython.display import clear_output
from Profiler import *

pi = numpy.pi

line = line1 = "*"*60 
line2 = "\n" + line + ""
line3 = line2 + "\n"

numpy.set_printoptions(suppress=1)

print( f"Importing python packages was done." )
print( f"time = {perf_counter_ns()}" )

#@profile
def _pqs_facotrial( p, q, t, use_gpu ) :
    s = numpy.arange( 0, t + 1 ) 
    
    R_ps = numpy.power( -1, s )*factorial(p - s)/factorial(s)/factorial( (p + q)/2 - s)/factorial( (p - q)/2 - s )
    
    if use_gpu :
        s = cupy.asarray( s )
        R_ps = cupy.asarray( R_ps )
    pass

    return R_ps, s 
pass # _pqs_facotrial

def _rps( r_ps, rho, p_2s, hash, use_gpu, use_hash = 0, debug = 0 ) :
    rho_id = id( rho )
    
    p_2s = int( p_2s )
    
    key_all = f"rps:{p_2s}:{r_ps}"
    
    rho_power = None
    
    if key_all in hash :
        rho_power = hash[ key_all ] 
        
        if use_gpu :
            rho_power = cupy.asarray( rho_power )
        pass
        
        return rho_power;
    pass

    if p_2s in hash :
        rho_power = hash[ p_2s ]
        
        if use_gpu :
            rho_power = cupy.asarray( rho_power )
        pass
    else : 
        np = cupy if use_gpu else numpy 
        
        if p_2s in [ -2, -1, 0, 1, 2 ] :
            rho_power = np.power( rho, p_2s )
        else :
            rho_power = _rps( 1, rho, p_2s//2, hash, use_gpu, use_hash=use_hash, debug = debug)
            
            if p_2s % 2 == 1 : 
                rho_power = rho_power*rho_power*rho
            else :
                rho_power = rho_power*rho_power
            pass
        pass
    
        if use_hash : 
            hash[ p_2s ] = cupy.asnumpy( rho_power ) if use_gpu else rho_power
        pass
    pass

    if r_ps not in [ 1, 1.0 ] :
        rho_power = r_ps*rho_power
    pass
    
    if use_hash : 
        hash[ key_all ] = cupy.asnumpy( rho_power ) if use_gpu else rho_power 
    pass

    #print( f"rho_power type = {rho_power.dtype} " )
    
    return rho_power
pass # _rps

#@profile
# radial function
def Rpq(p, q, rho, use_gpu, hash={}, use_hash=1, debug = 0 ) :
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
    
    r_pq_rho = None 
    
    if use_hash and key in hash :
        r_pq_rho = hash[ key ]
        
        if use_gpu :
            r_pq_rho = cupy.asarray( r_pq_rho )
        pass 
    
        return r_pq_rho 
    pass

    np = cupy if use_gpu else numpy 
    
    if p == 1 and q == 1 :
        r_pq_rho = rho
    elif p == 2 and q == 2 :
        r_pq_rho = rho*rho
    else :
        t = max( (p - q)/2, 0 ) 

        R_ps, s = _pqs_facotrial( p, q, t, use_gpu )

        rho_power = []

        for r_ps, p_2s in zip( R_ps, p - 2*s ) :
            rho_power.append( _rps( r_ps, rho, p_2s, hash, use_gpu, use_hash=use_hash, debug=debug ) )
        pass

        rho_power = np.array( rho_power ) 

        r_pq_rho = np.sum( rho_power, axis=0 )
    pass
    
    if use_hash : 
        hash[ key ] = cupy.asnumpy( r_pq_rho ) if use_gpu else r_pq_rho 
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
pass # radial function

#@profile
def Vpq( p, q, rho, theta, use_gpu, hash={}, use_hash=0, debug = 0 ) :    
    q = int(q)
    
    key = f"v:{p}:{q}" 
    
    if use_hash and key in hash :
        v_pq = hash[ key ]
        
        return cupy.asarray( v_pq ) if use_gpu else v_pq  
    pass
    
    np = cupy if use_gpu else numpy
    
    v_pq = None 
    
    r_pq = Rpq( p, q, rho, use_gpu, hash=hash, debug = 0)

    v_pq = r_pq*np.exp( (1j*q)*theta ) 

    if use_hash :
        hash[ key ] = cupy.asnumpy( v_pq ) if use_gpu else v_pq 
    pass
    
    if debug :
        print( f"Vpq({p}, {q}) = ", v_pq )
    pass

    return v_pq
pass

#@profile
def rho_theta( img, circle_type, use_gpu, debug = 0 ) :
    h = img.shape[0]
    w = img.shape[1]
    
    mwh = max( h - 1, w -1 )
    radius = math.sqrt( 2*mwh*mwh )
    
    debug and print( f"H = {h}, W = {w}, r = {radius}" )
    
    np = cupy if use_gpu else numpy
    
    y, x = np.where( img >= 0 ) 

    if debug : 
        print( "x = ", x )
        print( "y = ", y )
    pass

    dx = 2.0/max(h, w)
    dy = dx
    
    if "inner" in circle_type : 
        y = (y/mwh*2 - 1.0).flatten()
        x = (x/mwh*2 - 1.0).flatten()
        
        dx = 2.0/max(h, w)
        dy = dx
    else :
        sqrt_2 = math.sqrt(2)
        
        y = (y/mwh*sqrt_2 - (1.0/sqrt_2) ).flatten()
        x = (x/mwh*sqrt_2 - (1.0/sqrt_2) ).flatten()
        
        dx = sqrt_2/max(h, w)
        dy = dx
    pass
    
    dx = 2.0/max(h, w)
    dy = dx
    
    if debug : 
        print( "x = ", x )
        print( "y = ", y )
    pass
    
    rho_square = x**2 + y**2
    
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
    
    return rho, theta, x, y, dx, dy, k
pass # rho_theta

# 저니크 피라미드 생성 
def create_zernike_pyramid( row_cnt, col_cnt, circle_type, img_type, use_gpu, use_hash=0, debug = 0 ) : 
    print( f"use_gpu = {use_gpu}, use_hash = {use_hash}" )
    print_curr_time()

    print( "\nZernike Pyramid Creation Validation" )
    
    K = 2
    h = 1_000*K
    w = h 
    img = numpy.ones( (h, w), numpy.uint8 )
    
    if use_gpu :
        img = cupy.asarray( img )
    pass
    
    rho, theta, x, y, dx, dy, k = rho_theta( img, circle_type, use_gpu=use_gpu, debug=debug )
    
    hash = {}
    
    np = cupy if use_gpu else numpy
    
    imgs = []
    titles = []
    
    row_cnt -= 1
    total_cnt = row_cnt*col_cnt
    
    p = 0 
    idx = 0 

    while idx < total_cnt : 
        q = - p 
        while idx < total_cnt and q <= p : 
            if (p - q)%2 ==  0 :         
                title = f"\nZ({p}, {q})"
                titles.append( title )
            
                v_pl = Vpq( p, q, rho, theta, use_gpu, hash=hash, use_hash=use_hash, debug=0)
                
                z_img = None 
                
                if "im" in img_type : 
                    z_img = v_pl.imag
                elif "abs" in img_type : 
                    z_img = np.absolute( v_pl )
                else :
                    z_img = v_pl.real
                pass 
                
                img = np.zeros( (h, w), numpy.float_ )
                
                img = img.flatten()
                
                img[k] = z_img
                
                img = img.reshape( h, w ) 
                
                imgs.append( img )
                
                idx += 1
            pass
        
            q += 1  
        pass
    
        p += 1
    pass

    n = len( imgs ) 
    
    fig, charts = plt.subplots( row_cnt, col_cnt, figsize=( 3*col_cnt, 3*row_cnt) )
    charts = charts.flatten() if row_cnt*col_cnt > 1 else [charts]
    chart_idx = 0
    
    for idx, img in enumerate( imgs ) : 
        chart = charts[ idx ]
                
        if use_gpu :
            img = cupy.asnumpy( img )
        pass

        pos = chart.imshow( img, cmap="Spectral" )
        if idx % col_cnt == col_cnt - 1  : 
            fig.colorbar(pos, ax=chart)
        pass
        
        chart.set_xlabel( f"{titles[idx]}\n" ) 
    pass

    plt.tight_layout();
    plt.savefig( f"./pyramid/zernike_pyramid.png" )
    plt.show()
    
pass #create_zernike_pyramid

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
    pass
    
    from tabulate import tabulate
    
    print(tabulate(list_gpus, headers=("id", "name", "load", "free memory", "used memory", "total memory", "temperature" )))
pass # -- print_gpu_info 

print( "Zernike functions are defined.")
print_curr_time()
