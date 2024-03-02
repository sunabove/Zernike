# -*- coding: utf-8 -*-

print( f"Hello... Good morning!\n" )

import os, time, math, logging as log, cv2 as cv
import psutil, GPUtil, pandas as pd
import torch
import numpy, scipy
import ray # ray for parallel computing

from skimage import data
from skimage import io
from skimage.color import rgb2gray

from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import cm

from datetime import datetime
from datetime import timedelta

from tabulate import tabulate
from AProfiler import *

from tqdm import tqdm
from pathlib import Path

log.basicConfig(level=log.DEBUG) 

pi = torch.pi

line = line1 = "*"*90 
line2 = "\n" + line + ""
line3 = line2 + "\n"

print( f"Packages imported." )

class Grid :

    def __init__(self):
        self.rho = None
        self.theta = None

        self.x = None
        self.y = None 

        self.dx = None 
        self.dy = None

        self.kidx = None 
        self.area = None

        self.resolution = None
        self.circle_type = None
    pass

pass # Grid

ray_inited = False 

def ray_init() :
    global ray_inited 

    if not ray_inited : 
        ray_inited = True 
        then = time.time()
        
        print( "Initializaing ray ..." )
        
        ray.init()
        
        elapsed = time.time() - then
        
        print( f"Initializing ray done. Elapsed time = {elapsed:.3f} (sec.)")
    pass

pass # ray_init

def factorial( n ) :
    if torch.is_tensor( n ) :
        v = torch.exp( torch.lgamma(n + 1) )

        v[ torch.where( v.isinf() ) ] = 0 

        if torch.isnan( v ).any() : print( "torch factorial() : nan encountered" ); print( f"n = {n}")
        if torch.isinf( v ).any() : print( "torch factorial() : inf encountered" ); print( f"n = {n}")
    else :
        v = scipy.special.factorial( n )

        if numpy.isnan( v ).any() : print( "numpy factorial() : nan encountered" ); print( f"n = {n}")
        if numpy.isinf( v ).any() : print( "numpy factorial() : inf encountered" ); print( f"n = {n}")
    pass

    return v
pass

#@profile
def _pqs_facotrial( p, q, device ) :
    use_torch = 1

    if use_torch : 
        return _pqs_facotrial_torch( p, q, device )
    else :
        return _pqs_facotrial_numpy( p, q, device )
    pass
pass # _pqs_facotrial

def _pqs_facotrial_torch( p, q, device ) :
    kmax = max( (p - q)/2, 0 ) 

    k = torch.arange( 0, kmax + 1, device=device ) 

    #fact = factorial( p - s )/factorial( s )/factorial( (p + q)/2 - s)/factorial( (p - q)/2 - s )
    fact = factorial( p - k )
    fact1 = fact/factorial( k )
    fact2 = fact1/factorial( (p + q)/2 - k )
    fact3 = fact2/factorial( (p - q)/2 - k )
    
    R_ps = torch.pow( -1, k )*( fact3 )

    if R_ps.isnan().any() :
        print( "_pqs_factorial( ....) : Nan encountered." )
    pass

    return R_ps, k 
pass # _pqs_facotrial_torch

def _pqs_facotrial_numpy( p, q, device ) :
    kmax = max( (p - q)/2, 0 ) 

    k = numpy.arange( 0, kmax + 1 ) 

    #fact = factorial( p - s )/factorial( s )/factorial( (p + q)/2 - s)/factorial( (p - q)/2 - s )
    fact = factorial( p - k )
    fact1 = fact/factorial( k )
    fact2 = fact1/factorial( (p + q)/2 - k )
    fact3 = fact2/factorial( (p - q)/2 - k )
    fact4 = torch.tensor( fact3 ).to( device )
    
    R_ps = torch.pow( -1, torch.tensor( k, device=device ) )*( fact4 )    

    if R_ps.isnan().any() :
        print( "_pqs_factorial( ....) : Nan encountered." )
    pass

    return R_ps, k 
pass # _pqs_facotrial_numpy

def _rps( r_ps, rho, p_2s, debug=0 ) :
    p_2s = int( p_2s )
    
    rho_power = None
    
    if p_2s in [ -2, -1, 0, 1, 2 ] :
        rho_power = torch.pow( rho, p_2s )
    else :
        rho_power = _rps( 1, rho, p_2s//2 )
        
        if p_2s % 2 == 1 : 
            rho_power = rho_power*rho_power*rho
        else :
            rho_power = rho_power*rho_power
        pass
    pass
    
    if r_ps not in [ 1, 1.0 ] :
        rho_power = r_ps*rho_power
    pass

    return rho_power
pass # _rps

#@profile
# radial function
def Rpq( p, q, rho, device, debug=0 ) :
    q = abs( q )
    
    if abs(q) > p : 
        log.info( f"Invalid argument, abs(q = {q}) < p(={p}) is not satisfied")
        return 
    pass

    if int(p - abs(q))%2 == 1 : 
        log.info( f"Invalid argument, p({p}) - q({q}) should be an even number.")
        return 
    pass

    r_pq_rho = None
    
    if p == 1 and q == 1 :
        r_pq_rho = rho
    elif p == 2 and q == 2 :
        r_pq_rho = rho*rho
    else :
        R_ps, k = _pqs_facotrial( p, q, device=device )

        for r_ps, p_2s in zip( R_ps, p - 2*k ) :
            rps = _rps( r_ps, rho, p_2s, debug=debug ) 
            
            if r_pq_rho is None :
                r_pq_rho = rps
            else :
                r_pq_rho = r_pq_rho + rps
            pass
        pass 
    pass
    
    if 0 and debug : 
        print( line2 )
        print( f"p = {p}, q={q}" )
        print( "R_ps = ", R_ps ) 
        print( "R_pq_rho = ", r_pq_rho )    
        #print( "R_sum = ", R_sum )
        print( line2 )
    pass
    
    return r_pq_rho
pass # radial function

def vpq_key( p, q ) :
    return f"v:{p}:{q}"
pass # vpq_key

def get_cache_device( curr_device, resolution ) : 
    # GPU를 사용하는 경우에는 마지막 GPU를 캐시 장치로 사용
    # CPU인 경우에는 CPU를 캐시 장치로 사용한다.

    if "cuda" in f"{curr_device}" :
        gpus = GPUtil.getGPUs()

        gpu_cnt = len( gpus )

        device_no = gpu_cnt - 1

        target_mem = resolution*resolution*8

        for idx in range( gpu_cnt -1, 0, -1 ) :
            free_mem, total_mem = torch.cuda.mem_get_info( idx )

            if idx == 0 and free_mem > target_mem*4 :
                device_no = idx
                break
            if free_mem > target_mem : 
                device_no = idx
                break
            pass
        pass

        return torch.device( f"cuda:{device_no}" )
    else :
        return torch.device( "cpu" )
    pass
pass

def Vpq_file_path( p, q, resolution, circle_type ) :
    src_dir = os.path.dirname( os.path.abspath(__file__) )
    cache_file = f"{src_dir}/pyramid/v_{circle_type}_R{int(resolution):04d}_P{p:02d}_Q{q:02d}.pth"

    return cache_file
pass # Vpq_file_path

def vpq_load_from_cache( p, q, grid, device, cache, debug=0 ) :
    resolution = grid.resolution
    circle_type = grid.circle_type

    v_pq = None

    if q < 0 :
        v_pq = vpq_load_from_cache( p, abs(q), grid, device, cache, debug=debug )

        if v_pq is not None:
            v_pq = torch.conj( v_pq )

            return v_pq
        pass
    pass

    if cache is None :
        return v_pq
    elif cache is not None and p in cache and q in cache[p] : 
        v_pq = cache[p][q]

        return v_pq.to( device )
    pass
    
    cache_file = Vpq_file_path( p, q, resolution, circle_type )
    
    if cache is not None and os.path.exists( cache_file ) :
        if debug :
            print( f"--- zernike cache file load = {cache_file}" )
        pass

        v_pq = torch.load( cache_file, map_location=device, weights_only=1 )
    pass

    return v_pq
pass # Vpq_load_cache

def vpq_save_to_cache( v_pq, p, q, grid, device, cache, debug=0 ) :

    resolution = grid.resolution
    circle_type = grid.circle_type

    if cache is not None and q < 0 : 
        v_pq = torch.conj( v_pq )
        
        vpq_save_to_cache( v_pq, p, abs(q), grid, device, cache, debug=debug )
    elif cache is not None :

        if not p in cache :
            cache[p] = { }
        pass

        cache_device = get_cache_device( device, resolution )

        if debug :
            print( f"V p = {p:3d}, q = {q:3d}, resolution = {resolution}, cache_device = {cache_device}" )
        pass

        cache[p][q] = v_pq.to( cache_device )

        # save to file

        cache_file = Vpq_file_path( p, q, resolution, circle_type )

        if os.path.exists( cache_file ) == False :
            torch.save( v_pq, cache_file )

            if debug: 
                print( f"--- zernike cache file save = {cache_file}" )
            pass
        pass
    pass
pass # Vpq_save_cache

#@profile
def Vpq( p, q, grid, device=None, cache=None, debug=0) :

    rho = grid.rho
    theta = grid.theta
    resolution = grid.resolution
    circle_type = grid.circle_type

    if 0 and debug :
        print( f"V p = {p}, q = {q}, circle_type = {circle_type}, K = {resolution/1000}, cache = {cache != None}" )

    v_pq = None

    if cache is not None :
        v_pq = vpq_load_from_cache( p, q, grid, device, cache, debug=debug) 
    pass

    if v_pq is not None :
        return v_pq
    elif v_pq is None : 
        q = int(q)
        
        if q < 0 : 
            v_pq = Vpq( p, abs(q), grid, device=device, cache=cache, debug=debug )
            
            v_pq = torch.conj( v_pq )
        else : 
            r_pq = Rpq( p, q, rho, device=device, debug=0 )

            if q != 0 :
                v_pq = r_pq*torch.exp( (1j*q)*theta )
            else :
                v_pq = r_pq + 0j
            pass
        pass
    pass

    if cache is not None :
        vpq_save_to_cache( v_pq, p, q, grid, device, cache, debug=debug) 
    pass

    if 0 and debug :
        print( f"Vpq({p}, {q}) = ", v_pq )
    pass

    return v_pq
pass

#@profile
def rho_theta( resolution, circle_type, device, debug=0 ) :
    img = torch.ones( ( int(resolution), int( resolution) ), dtype=torch.float64, device=device ) 
    
    h = img.shape[0]
    w = img.shape[1]
    
    radius = math.sqrt( w*w + h*h )
    
    debug and print( f"H = {h}, W = {w}, r = {radius}" )
    
    # 영상 인덱스를 직교 좌표계 값으로 변환
    y, x = torch.where( img >= 0 ) 

    if debug : 
        print( f"x size = { x.size()}" )
        print( f"y size = { y.size()}" )
    pass

    dy = dx = 2.0/w    
    area = pi
    
    if "inner" in circle_type : 
        y = (y/h*2 - 1.0).flatten()
        x = (x/w*2 - 1.0).flatten()
        
        dy = dx = 2.0/max(h, w)
        
        area = pi # 3.14 area of the whole circle
    else : # outer cirlce
        sqrt_2 = math.sqrt(2)
        
        y = (y/h*sqrt_2 - (1.0/sqrt_2) ).flatten()
        x = (x/w*sqrt_2 - (1.0/sqrt_2) ).flatten()
        
        dy = dx = sqrt_2/max(h, w)
        
        area = 2  # area of the rectangle inside
    pass 

    if debug : 
        print( f"x size = {x.size()}" )
        print( f"y size = {y.size()}" )
    pass
    
    rho_square = x**2 + y**2
    
    kidx = None
    
    if "inner" in circle_type : 
        kidx = torch.where( rho_square <= 1.0 )
    else :
        # all index of outer circle
        kidx = torch.where( rho_square <= 2.0 )
    pass
    
    y = y[ kidx ]
    x = x[ kidx ]    
    rho_square = rho_square[ kidx ]
    
    if debug : 
        print( "x[k] = ", x )
        print( "y[k] = ", y )
    pass

    rho = torch.sqrt( rho_square )
    theta = torch.arctan2( y, x )
    
    grid = Grid()

    grid.rho = rho
    grid.theta = theta
    grid.x = x
    grid.y = y
    grid.dx = dx
    grid.dy = dy
    grid.kidx = kidx
    grid.area = area

    grid.resolution = resolution
    grid.circle_type = circle_type
    
    #return rho, theta, x, y, dx, dy, kidx, area
    return grid
pass # rho_theta

def print_curr_time() :
    # 현재 시각 출력 
    print("Current Time =", datetime.now().strftime("%H:%M:%S") )
pass

def print_cpu_info() :
    import platform, psutil
    
    print( " CPU Details ".center( len(line), "*") ) 

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
    print( " Memory Details ".center( len(line), "*") ) 
    #Total RAM
    print(f"Total RAM installed: {round(psutil.virtual_memory().total/10**9, 2)} GB")
    #Available RAM
    print(f"Available RAM: {round(psutil.virtual_memory().available/10**9, 2)} GB")
    #Used RAM
    print(f"Used RAM: {round(psutil.virtual_memory().used/10**9, 2)} GB")
    #RAM usage
    print(f"RAM usage: {psutil.virtual_memory().percent}%")
    
    max_memory = psutil.virtual_memory().total/10**9 # GB
    
    return max_memory
pass # -- print_cpu_info

def max_cpu_memory() :
    import platform, psutil 
            
    max_memory = psutil.virtual_memory().available/10**9 # GB
    
    #log.info(f"CPU Available RAM: { max_memory } GB")
    
    return max_memory
pass # -- max_cpu_memory

def max_gpu_memory() :
    import GPUtil
    
    max_memory = 0 
    
    gpus = GPUtil.getGPUs()
    
    for gpu in gpus:
        # get the GPU id
        gpu_free_memory = gpu.memoryFree/1_000  
        
        if gpu_free_memory > max_memory : 
            max_memory = gpu_free_memory
        pass
    pass

    log.info(f"GPU Available RAM: { max_memory } GB")
    
    return max_memory
pass # -- max_gpu_memory 

# 배열 여부 체크 
def is_array( data ) :
    return isinstance( data, (list, tuple, numpy.ndarray, torch.Tensor) )
pass # is_array

# 스칼라 여부 체크 
def is_scalar( data ) :
    return not is_array( data, )
pass # is_scalar

def print_gpu_info() :
    print( " GPU Details ".center( len(line), "*") ) 

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
        
        list_gpus.append( ( gpu_id, gpu_name, gpu_load, gpu_free_memory, gpu_used_memory, gpu_total_memory, gpu_temperature ) )
    pass
    
    print( tabulate( list_gpus, headers=("id", "name", "load", "free memory", "used memory", "total memory", "temperature" )) )
pass # -- print_gpu_info 

def pq_list( T ) :
    pqs = []

    for p in range( 0, T + 1 ) : 
        for q in range( -p, p + 1, 2 ) :
            pqs.append( [p, q] )
        pass
    pass

    return pqs
pass # pq_list

def get_option( key, default = 0 , ** options ) : 
    if key in options :
        return options[ key]
    else :
        return default 
    pass
pass # get_option

def get_core_count(**options) : 
    use_thread = get_option( "use_thread", **options )
    use_gpu = get_option( "use_gpu", **options ) 
    
    core_count = 0 
    
    if not use_thread :
        core_count =  0 
    else :
        if use_gpu :
            core_count = len( GPUtil.getGPUs() )
        else :
            core_count = psutil.cpu_count(logical=True)
        pass
    pass

    log.info( f"core_count = {core_count}, gpu = {use_gpu}, thread = {use_thread}")

    return core_count
pass
    
# 모멘트 계산
def calc_moments( img, T, resolution, circle_type, device, cache=None, debug=0 ) : 
    then = time.time()

    img = cv.resize( img, ( int(resolution), int( resolution) ), interpolation=cv.INTER_AREA )

    if 0 and debug : 
        print( "img shape= ", img.shape ) 
        print( line )
    pass

    img = torch.tensor( img, dtype=torch.complex64, device=device )

    moments = torch.zeros( (T + 1, 2*T + 1), dtype=torch.complex64, device=device )

    grid = rho_theta( resolution, circle_type, device=device, debug=0 )
    
    if 0 and debug : print( f"rho shape = {grid.rho.shape}" )
    
    img_rav = img.ravel()

    for p, q in pq_list( T ) : 
        v_pq = Vpq( p, q, grid, device=device, cache=cache, debug=debug ) 
        
        moment = torch.dot( v_pq, img_rav )*grid.dx*grid.dy

        moment = torch.conj( moment )

        moments[ p, q ] = moment
    pass

    if 0 :
        print( "Moments = ", moments )
    pass

    run_time = time.time() - then

    return moments, run_time
pass # calc_moments

def restore_image(moments, rho, theta, **options) : 
    use_thread = get_option( "use_thread", **options )
    use_gpu = get_option( "use_gpu", **options )
    np = cupy if use_gpu else numpy
    
    
    then = perf_counter()
    
    s = T = moments.shape[0] - 1 
    
    img = np.zeros_like( rho, np.complex_ )
    
    pqs = pq_list( T )
    
    area = 2 # outer type image area in unit_circle
    if "circle_type" in options and "inner" in options["circle_type"] :
        area = pi
    pass        
    
    for p, q in pqs :
        v_pq = Vpq( p, q, rho, theta, ** options )
        img += ((p+1)/area)*moments[p, q]*v_pq
    pass 

    s = int( math.sqrt( len( img ) ) )
    
    img = img.reshape( s, s )
    
    run_time = perf_counter() - then
    
    return img , run_time
pass ## restore_image

def calc_psnr(img_org, img_restored, **options ) : 
    use_thread = get_option( "use_thread", **options )
    use_gpu = get_option( "use_gpu", **options )
    
    #print( f"calc_psnr use_gpu = {use_gpu}" )
        
    np = cupy if use_gpu else numpy
    
    img_diff = img_org - img_restored

    #gmax = np.max( img_org ) # 최대값 
    gmax = 255
    
    mse = np.sum( np.square( img_diff ) ) / img_diff.size
    psnr = 10*math.log10(gmax*gmax/mse)
    
    return psnr
pass # calc_psnr

# Mean of Absolute Differnce
def calc_mad(img_org, img_restored, **options ) : 
    use_thread = get_option( "use_thread", **options )
    use_gpu = get_option( "use_gpu", **options )
    
    #print( f"calc_psnr use_gpu = {use_gpu}" )
        
    np = cupy if use_gpu else numpy
    
    img_diff = img_org - img_restored

    mad = np.sum( np.absolute( img_diff ) ) / img_diff.size 
    
    if use_gpu : 
        mad = cupy.asnumpy( mad )
    pass
    
    return mad
pass # calc_mad

def get_moments_disp(moments, **options ) :
    use_thread = get_option( "use_thread", **options )
    use_gpu = get_option( "use_gpu", **options )
    
    np = cupy if use_gpu else numpy
    
    T = moments.shape[0] - 1
    s = T + 1 
    
    moments_disp = np.zeros( (s, s), np.complex_ )
    
    for p in range( 0, T + 1 ) :
        r = p
        c = 0 
        for q in range( -p, p + 1, 2 ) :
            #print( f"p = {p}, q = {q}, r = {r}, c = {c}" )
            
            moments_disp[ r, c ] = moments[ p, q ]
            
            r = r - 1
            c = c + 1            
        pass
    
        #print()
    pass

    #moments_disp = moments_disp*127.5 + 127.5
    
    return moments_disp    
pass # get_moments_disp

###########################################################################
###########################################################################

# 저니크 모멘트 함수 실험 
def test_zernike_features( feature_info, img_name, img, use_gpus, use_threads, use_hashs, Ks, Ts, **options ) : 
    
    if is_array( use_gpus ) :
        for use_gpu in use_gpus :
            options = {}
            options[ "debug" ] = get_option( "debug", **options )
            options[ "use_gpu" ] = use_gpu 
            options[ "use_thread"] = 0 
            options[ "use_hash" ] = 0 
            
            test_zernike_moments( feature_info, img_name, img, use_gpu, use_threads, use_hashs, Ks, Ts, **options )
        pass
    elif is_array( use_threads ) :
        for use_thread in use_threads :
            options = {}
            options[ "debug" ] = get_option( "debug", **options )
            options[ "use_gpu" ] = use_gpus 
            options[ "use_thread"] = use_thread  
            
            test_zernike_moments( feature_info,  img_name, img, use_gpus, use_thread, use_hashs, Ks, Ts, **options )
        pass        
    elif is_array( use_hashs ) :
        for use_hash in use_hashs :
            options = {}
            options[ "debug" ] = get_option( "debug", **options )
            options[ "use_gpu" ] = use_gpus 
            options[ "use_thread"] = use_threads
            options[ "use_hash" ] = use_hash
            
            test_zernike_moments( feature_info,  img_name, img, use_gpus, use_threads, use_hash, Ks, Ts, **options )
        pass 
    else :
        debug = get_option( "debug", **options )
        use_hash = get_option( "use_hash", ** options )
        use_thread = get_option( "use_thread", **options )
        use_gpu = get_option( "use_gpu", **options )
    
        device = "GPU" if use_gpu else "CPU"
        multi = "M-" if use_thread else ""
        
        if is_scalar( Ks ) :
            Ks = [ Ks ]
        pass
    
        if is_scalar( Ts ) :
            Ts = [ Ts ]
        pass 
        
        key = None
        for K in Ks :
                        
            if len( Ts ) > 1 : 
                key = f"{multi}{device}, Hash={use_hash}, K=({K})"
            else :
                key = f"{multi}{device}, Hash={use_hash}"
            pass    
        
            print( f"device={key}", flush=True) 
        
            options[ "hash" ] = {} 
            
            circle_type = "outer" 

            rho, theta, x, y, dx, dy, k, area = rho_theta( 1000*K, circle_type, **options ) 

            np = cupy if use_gpu else numpy

            #img_infos.append( { "title" : "Image Org", "img" : img } )
            
            img = cv.resize( img, (int(K*1_000), int(K*1_000)), interpolation=cv.INTER_AREA )
            
            if use_gpu : 
                img = np.array( img )
            pass
        
            feature_info [ "img_scaled" ] = { "title" : f"Image Scaled (K={K})", "img" : img }

            if debug : 
                print( "img shape= ", img.shape ) 
                print( line )
            pass
        
            for T in Ts :
                moments, run_time = calc_moments(T, img, rho, theta, dx, dy, **options )
                
                print( f"K = {K:2}, T = {T:2}, Run-time = {run_time:7.2f} (sec.)" )   
                
                feature_info[ "moments" ] = moments 
                feature_info[ "run_time" ] = run_time 
                feature_info[ "K" ] = K
                feature_info[ "T" ] = T 
                
            pass # T
        pass # K
    pass
pass # test_zernike_features

# 모멘트 특징 출력 / 시각호 
def plot_moment_features( feature_info, img_name, **options ) : 

    print( "\nPlotting .... ")
    
    use_gpu = get_option( "use_gpu", **options )
    
    np = cupy if use_gpu else numpy  
    
    img_scaled = feature_info[ "img_scaled" ]
    
    K = feature_info[ "K" ]
    T = feature_info[ "T" ]
    
    moments = feature_info[ "moments" ]
    moments_disp = get_moments_disp( moments, **options )
    
    img_infos = []
    
    img_infos.append( img_scaled )
                
    img_infos.append( { "title" : f"Moment Abs (K={K}, T={T})", "img" : np.absolute( moments_disp ), "colorbar" : 0 } )
    img_infos.append( { "title" : f"Moment Real (K={K}, T={T})", "img" : moments_disp.real, "colorbar" : 0 } )
    img_infos.append( { "title" : f"Moment Imag (K={K}, T={T})", "img" : moments_disp.imag, "colorbar" : 0 } ) 
    
    # 서브 챠트 생성 
    col_cnt = 4
    
    n = len( img_infos ) + T + 1
    row_cnt = n // col_cnt
    if col_cnt*row_cnt < n :
        row_cnt += 1
    pass

    fig, charts = plt.subplots( row_cnt, col_cnt, figsize=(4*col_cnt, 4*row_cnt) )
    charts = charts.ravel() if row_cnt*col_cnt > 1 else [charts]
    chart_idx = 0 
    
    for img_info in img_infos : 
        t_img = img_info[ "img" ]
        title = img_info[ "title" ]
        title_low = title.lower()
        
        if "psnr" in img_info : 
            psnr = img_info[ "psnr"]
            
            title = f"\n{title}, PSNR={psnr:.2f}\n"
        else :
            title = f"\n{title}\n"
        pass
            
        colorbar = False 
        if "colorbar" in img_info :
            colorbar = img_info[ "colorbar" ]

        chart = charts[ chart_idx ] ; chart_idx += 1
        
        chart.set_title( title )
        
        pos = chart.imshow( cupy.asnumpy( t_img ) if options["use_gpu"] else t_img, cmap='gray' )
        colorbar and fig.colorbar(pos, ax=chart)
    pass  

    s = T
    x = numpy.arange( -s, s + 1 )
    
    for p, m_org in enumerate( moments ) : 
        chart = charts[ chart_idx ] ; chart_idx += 1
        
        m = np.zeros_like( m_org )
        
        for q in range( -p, p + 1, 2 ) :
            m[T + q] = m_org[q]
        pass
        
        y = m.real
        real = cupy.asnumpy( y ) if use_gpu else y        
        
        y = m.imag
        imag = cupy.asnumpy( y ) if use_gpu else y
        
        absolute = numpy.sqrt( real*real + imag*imag )
        max_y = numpy.max( absolute )
        chart.set_ylim( -max_y*1.1, max_y*1.1 )
                
        chart.plot( x, absolute, marker="o", label=f"Absolute", color="tab:blue" )
        chart.plot( x, real,     marker="s", label=f"Real", linestyle="solid", color="g" )
        chart.plot( x, imag,     marker="D", label=f"Imag", linestyle="dotted", color="tab:orange" )
        
        title = f"Moment[p={p}]"
        chart.set_title( title )
        chart.set_xlabel( f"q" )
        chart.set_xticks( x[::2] )
        
        chart.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=3 ) 
    pass

    # draw empty chart
    for chart_idx in range( chart_idx, len(charts) ) :
        chart = charts[ chart_idx ]
        chart.plot( [0,0], [0,0] )
        chart.set_axis_off()
    pass

    plt.tight_layout(); plt.show()
    
    print()
    
    if True : # 모멘트 엑셀 저장 및 화면 출력 
        ## convert your array into a dataframe
        data = moments.T
        df = pd.DataFrame( cupy.asnumpy( data ) if use_gpu else data )
        filepath = f'result/moment_{img_name}_{K}_{T}.xlsx'
        df.to_excel(filepath, index=False)
        #print( df )
    pass
    
    if False :
        s = T + 1
        moments = cupy.asnumpy( moments ) if use_gpu else moments
        col_headers = numpy.arange( -s, 2*s + 1 )
        table = tabulate( moments.real , col_headers, tablefmt="fancy_grid", floatfmt = ".2f", showindex="always" )
        
        print( table )
    pass

pass # plot_moment_features

###########################################################################
###########################################################################

print( "Zernike functions are defined.\n")

if __name__ == "__main__" :
    
    if 0 :
        t = 5
        s = torch.arange( -t, t + 1 ) 
        print( f"torch s = {s}")
        print( "torch facotrial(0) = ", factorial( s ) )

        s = numpy.arange( -t, t + 1 ) 
        print( f"numpy s = {s}")
        print( "numpy facotrial(0) = ", factorial( s ) )
    elif 1 :
        print_cpu_info()    
        print()
        print_gpu_info()
    elif 0 :
        t = 3
        s = torch.arange( 0, t + 1 ) 
        t = torch.arange( 0, t + 1 ) 

        print( s*t )
    pass

pass