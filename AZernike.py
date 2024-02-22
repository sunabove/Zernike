# -*- coding: utf-8 -*-

print( f"Hello... Good morning!" )

import os, math, logging as log, cv2 as cv
import psutil , GPUtil, pandas as pd
import torch
import threading, ray # ray for parallel computing

from skimage.color import rgb2gray

from skimage import data
from skimage import io 

import numpy
import scipy
from matplotlib import pyplot as plt
from datetime import datetime
from tabulate import tabulate
from Profiler import *

log.basicConfig(level=log.DEBUG) 

pi = torch.pi

line = line1 = "*"*90 
line2 = "\n" + line + ""
line3 = line2 + "\n"

print( f"Importing python packages was done." )

ray_inited = False 

def ray_init() :
    global ray_inited 

    if not ray_inited : 
        ray_inited = True 
        then = perf_counter()
        
        print( "Initializaing ray ..." )
        
        ray.init()
        
        elapsed = perf_counter() - then
        
        print( f"Initializing ray done. Elapsed time = {elapsed:.3f} (sec.)")
    pass

pass # ray_init

def factorial( n ) :
    if torch.is_tensor( n ) :
        v = (n + 1).lgamma().exp()

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
def _pqs_facotrial( p, q, t, device ) :
    s = numpy.arange( 0, t + 1 ) 

    #fact = factorial( p - s )/factorial( s )/factorial( (p + q)/2 - s)/factorial( (p - q)/2 - s )
    fact = factorial( p - s )
    fact1 = fact/factorial( s )
    fact2 = fact1/factorial( (p + q)/2 - s )
    fact3 = fact2/factorial( (p - q)/2 - s )
    fact4 = torch.tensor( fact3 ).to( device )
    
    R_ps = torch.pow( -1, torch.tensor( s, device=device ) )*( fact4 )    

    #R_ps = torch.pow( -1, s )*( fact3 )

    if R_ps.isnan().any() :
        print( "_pqs_factorial( ....) : Nan encountered." )
    pass

    return R_ps, s 
pass # _pqs_facotrial

def _rps( r_ps, rho, p_2s, device=None, hash=None ) :
    p_2s = int( p_2s )
    
    key = f"rps:{p_2s}:{r_ps}"
    
    rho_power = None
    
    if key in hash :
        rho_power = hash[ key ] 

        rho_power = rho_power.to( device )
        
        return rho_power
    pass

    if p_2s in hash :
        rho_power = hash[ p_2s ]
        
        rho_power = rho_power.to( device )
    else : 
        if p_2s in [ -2, -1, 0, 1, 2 ] :
            rho_power = torch.pow( rho, p_2s )
        else :
            rho_power = _rps( 1, rho, p_2s//2, device=device, hash=hash )
            
            if p_2s % 2 == 1 : 
                rho_power = rho_power*rho_power*rho
            else :
                rho_power = rho_power*rho_power
            pass
        pass
    
        if hash is not None : 
            hash[ p_2s ] = rho_power.to( "cpu" )
        pass
    pass

    if r_ps not in [ 1, 1.0 ] :
        rho_power = r_ps*rho_power
    pass
    
    if hash is not None : 
        hash[ key ] = rho_power.to( "cpu" )
    pass

    #print( f"rho_power type = {rho_power.dtype} " )
    
    return rho_power
pass # _rps

#@profile
# radial function
def Rpq(p, q, rho, device, hash, debug=0 ) :
    q = abs( q )
    
    if abs(q) > p : 
        log.info( f"Invalid argument, abs(q = {q}) < p(={p}) is not satisfied")
        return 
    pass

    if int(p - abs(q))%2 == 1 : 
        log.info( f"Invalid argument, p({p}) - q({q}) should be an even number.")
        return 
    pass

    key = f"rpq:{p}:{q}"
    
    r_pq_rho = None
    
    if hash is not None and key in hash :
        r_pq_rho = hash[ key ]
        
        r_pq_rho = r_pq_rho.to( device )
    
        return r_pq_rho 
    pass

    if p == 1 and q == 1 :
        r_pq_rho = rho
    elif p == 2 and q == 2 :
        r_pq_rho = rho*rho
    else :
        t = max( (p - q)/2, 0 ) 

        R_ps, s = _pqs_facotrial( p, q, t, device=device )

        for r_ps, p_2s in zip( R_ps, p - 2*s ) :
            rps = _rps( r_ps, rho, p_2s, device=device, hash=hash ) 
            
            if r_pq_rho is None :
                r_pq_rho = rps
            else :
                r_pq_rho = r_pq_rho + rps
            pass
        pass 
    pass
    
    if hash is not None : 
        hash[ key ] = r_pq_rho.to( "cpu" )
    pass
        
    if debug : 
        print( line2 )
        print( f"p = {p}, q={q}, (p - |q|)/2 = {t}" )
        print( "s = ", s )
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

#@profile
def Vpq( p, q, rho, theta, device, hash, debug=0) :    
    q = int(q)
    
    key = vpq_key( p, q )
    
    if hash is not None and key in hash :
        v_pq = hash[ key ]
        
        return v_pq.to( device )
    pass
    
    v_pq = None 
    
    if q < 0 : 
        v_pq = Vpq( p, abs(q), rho, theta, device=device, hash=hash, debug=debug )
        
        v_pq = torch.conj( v_pq )
    else : 
        r_pq = Rpq( p, q, rho, device=device, hash=hash, debug=debug )

        if q :
            v_pq = r_pq*torch.exp( (1j*q)*theta )
        else :
            v_pq = r_pq + 0j
        pass
    pass

    if hash :
        hash[ key ] = v_pq.cpu()
    pass
    
    if debug :
        print( f"Vpq({p}, {q}) = ", v_pq )
    pass

    return v_pq.to( device )
pass

#@profile
def rho_theta( resolution, circle_type, device, debug=0 ) :
    img = torch.ones( ( int(resolution), int( resolution) ), dtype=torch.float64, device=device ) 
    
    h = img.shape[0]
    w = img.shape[1]
    
    radius = math.sqrt( w*w + h*h )
    
    debug and print( f"H = {h}, W = {w}, r = {radius}" )
    
    # 직교 좌표계 좌표값들 추출
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
    
    return rho, theta, x, y, dx, dy, kidx, area
pass # rho_theta

# 저니크 피라미드 생성 테스트
def test_zernike_pyramid( row_cnt, col_cnt, circle_type, img_type, **options ) : 
    debug    = options[ "debug" ] if "debug" in options else False  
    use_gpu  = options[ "use_gpu" ] if "use_gpu" in options else False
    use_hash = options[ "use_hash" ] if "use_hash" in options else False 

    print_curr_time()
    print( "\nZernike Pyramid Creation Validation" )
    
    device_no = 0 
    hash = {} if use_hash else None 
    device = torch.device( f"cuda:{device_no}" ) if use_gpu else torch.device( f"cpu" )
    
    K = 2
    res = resolution = 1_000*K
    h = resolution
    w = h

    print( f"use_gpu = {use_gpu}, use_hash = {use_hash}, circle_type = {circle_type}, resolution = {resolution:_}" )
    
    rho, theta, x, y, dx, dy, kidx, area = rho_theta( resolution, circle_type, device=device, debug=debug )
    
    imgs = []
    titles = []
    
    total_cnt = row_cnt*col_cnt
    
    p = 0 
    idx = 0 

    while idx < total_cnt : 
        q = - p 
        while idx < total_cnt and q <= p : 
            if (p - q)%2 ==  0 :         
                title = f"$Z({p}, {q})$"
                            
                v_pl = Vpq( p, q, rho, theta, device=device, hash=hash, debug=debug )
                
                z_img = None # zernike image
                
                if "im" in img_type : 
                    z_img = v_pl.imag 

                    title = f"$Im(Z({p}, {q}))$"
                elif "abs" in img_type : 
                    z_img = torch.absolute( v_pl )

                    title = f"$|Z({p}, {q})|$"
                else :
                    z_img = v_pl.real

                    title = f"$Re(Z({p}, {q}))$"
                pass 

                titles.append( title )
                
                img = torch.zeros( (h, w), dtype=torch.float, device=device )
                img_rav = img.ravel()
                img_rav[ kidx ] = z_img

                imgs.append( img )

                if debug : 
                    print( f"rho size : {rho.size()}" )
                    print( f"v_pl size : {v_pl.size()}" )
                    print( f"z_img size : {z_img.size()}" )
                    print( f"img size : {img.size()}" )
                    print( f"img_rav size : {img_rav.size()}" )
                pass
                
                idx += 1
            pass
        
            q += 1  
        pass
    
        p += 1
    pass

    fs = fontsize = 16
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = fontsize

    fig, charts = plt.subplots( row_cnt, col_cnt, figsize=( 2.5*col_cnt, 2.5*row_cnt) )
    charts = charts.ravel() if row_cnt*col_cnt > 1 else [charts]
    chart_idx = 0 
    
    for idx, img in enumerate( imgs ) : 
        chart = charts[ idx ]

        img = img.cpu()
                
        pos = chart.imshow( img, cmap="Spectral" )

        if idx == 0 : 
            fig.colorbar(pos, ax=chart )
        pass
        
        chart.set_title( f"{titles[idx]}", fontsize=fs+4)

        chart.set_xticks( torch.arange( 0, res + 1, res//4 ) )
        chart.set_yticks( torch.arange( 0, res + 1, res//4 ) )

        if 1 :
            chart.set_xticklabels( [] )
            chart.set_yticklabels( [] ) 
        pass

        if idx == 0 : 
            chart.set_xlim( 0, res )
            chart.set_ylim( 0, res )

            ticks = torch.arange( 0, res + 1, res//4 )
            tick_cnt = ticks.numel()
            
            chart.set_xticks( ticks )
            chart.set_yticks( ticks )

            tick_labels = [ "" ]*tick_cnt
            tick_labels[  0 ] = '-1'
            tick_labels[ -1 ] = '1'

            if "out" in circle_type : 
                tick_labels[  0 ] = '$\\frac{-1}{\\sqrt{2}}$'
                tick_labels[ -1 ] = '$\\frac{1}{\\sqrt{2}}$'
            pass

            chart.set_yticklabels( tick_labels, fontsize=fs )
        pass
    pass

    plt.tight_layout()
    plt.show()

    src_dir = os.path.dirname( os.path.abspath(__file__) )
    result_figure_file = f"{src_dir}/pyramid/zernike_pyramid_{circle_type}_{K:02d}k_{img_type}.png"
    plt.savefig( result_figure_file )
    print( f"result_figure_file = {result_figure_file}" )

pass #create_zernike_pyramid

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
def calc_moments( T, img, rho, theta, dx, dy, device, hash, debug=0 ) : 
    then = time.time()

    moments = torch.zeros( (T + 1, 2*T + 1), torch.complex64 )

    img_rav = img.ravel()

    for p, q in pq_list( T ) : 
        v_pq = Vpq( p, q, rho, theta, device=device, hash=hash, debug=debug ) 
        
        moment = torch.dot( v_pq, img_rav )*dx*dy

        moment = torch.conjugate( moment )

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

# 저니크 모멘트 함수 실험 
def test_zernike_moments( datas, use_gpus, use_hashs, Ks, Ts, debug=0 ) : 
    
    if is_array( use_gpus ) :
        for use_gpu in use_gpus :
            test_zernike_moments( datas, use_gpu, use_hashs, Ks, Ts, debug=debug )
        pass
    elif is_array( use_hashs ) :
        for use_hash in use_hashs :
            test_zernike_moments( datas, use_gpus, use_hash, Ks, Ts, debug=debug )
        pass 
    else :
        use_hash = use_hashs
        use_gpu = use_gpus
        device_no = 0

        hash = {} if use_gpu else None
        device = torch.device( f"cuda:{device_no}" ) if use_gpu else torch.device( f"cpu" )
        
        if is_scalar( Ks ) :
            Ks = [ Ks ]
        pass
    
        if is_scalar( Ts ) :
            Ts = [ Ts ]
        pass 
        
        key = None
        for K in Ks :
                        
            if len( Ts ) > 1 : 
                key = f"{device}, Hash={use_hash}, K=({K})"
            else :
                key = f"{device}, Hash={use_hash}"
            pass    
            
            if not key in datas :
                print()
                print( f"device={key}", flush=True)
            
                data = {}
                data[ "Ks"] = Ks
                data[ "Ts"] = Ts
                data[ "run_times" ] = []
                datas[ key ] = data 
            pass
        
            data = datas[ key ]
            
            run_times = data[ "run_times" ] 
            
            circle_type = "outer"

            resolution = 1_000*K

            rho, theta, x, y, dx, dy, k, area = rho_theta( resolution, circle_type, device=device, debug=debug ) 

            if debug : print( f"rho shape = {rho.shape}" )

            img = cv.imread( 'image/lenna.png', 0 )

            if debug : print( "img shape= ", img.shape )

            img_org = img 

            img = cv.resize( img_org, (int(K*1_000), int(K*1_000)), interpolation=cv.INTER_AREA )

            img_org = img

            if debug : 
                print( "img shape= ", img.shape ) 
                print( line )
            pass

            img = torch.tensor( img )
        
            for T in Ts :
                moments, run_time = calc_moments(T, img, rho, theta, dx, dy, device=device, hash=hash, debug=debug )
                
                print( f"K = {K:2}, T = {T:2}, Run-time = {run_time:7.2f} (sec.)" ) 
                
                run_times.append( run_time )
            pass # T
        pass # K
    pass
pass # test_zernike_moments

def plot_moment_calc_times( datas ) : 

    print( "\nPlotting .... ")
        
    # 서브 챠트 생성 
    row_cnt = 1
    col_cnt = 1

    fig, charts = plt.subplots( row_cnt, col_cnt, figsize=(12*col_cnt, 8*row_cnt) )
    charts = charts.ravel() if row_cnt*col_cnt > 1 else [charts]
    chart_idx = 0 
    chart = charts[ chart_idx ] ; chart_idx += 1
    
    for key in datas : 
        data = datas[ key ]
        
        Ks = data[ "Ks" ]
        Ts = data[ "Ts" ]
        
        run_times = data[ "run_times" ]
        
        x = Ts if "K" in key else Ks
        y = numpy.log10( run_times ) 
        
        label = f"{key}"
        
        linestyle = "solid"
        
        if "M" in label :
            linestyle = "dotted"
        pass
        
        color = "b"
        
        if "GPU" in label :
            color = "g"
        pass

        marker = "*"
        if "Hash=0" in label : 
            marker = "s" 
            if "GPU" in label :
                color = "r"
            pass 
        pass
    
        label = label.replace( "Hash", "H" )

        chart.plot( x, y, marker=marker, color=color, label=label, linestyle=linestyle )
        
        title = f"Run-time"
        
        if len(Ks) == 1 :
            title = f"Run-time(K={Ks[0]})"
        elif len(Ts) == 1 :
            title = f"Run-time(T={Ts[0]})"
        pass
    
        chart.set_title( title ) 
        
        chart.set_xlabel( "Axial Grid Count" )
        chart.set_ylabel( "log10(Times) (sec.)")
        chart.set_xticks( x )
        if "K" in key :
            chart.set_xticklabels( [ f"{t} T" for t in x ] )
        else :
            chart.set_xticklabels( [ f"{k} K" for k in x ] )
        pass    
        
        #chart.set_xlim( numpy.min(x) - 1, numpy.max(x) + 1 )
        
        n = len( datas ) 
        ncol = 4
        for i in range( 2, 8 ) :
            if n % i == 0 :
                ncol = i 
            pass
        pass
        
        chart.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=ncol) 
    pass

    plt.tight_layout(); plt.show()
pass # plot_moment_calc_times

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

print( "Zernike functions are defined.")
print()

if __name__ == "__main__" :
    
    if 1 :
        t = 3 
        s = torch.arange( 0, t + 1 ) 
        print( f"torch s = {s}")
        print( "torch facotrial(0) = ", factorial( s ) )

        s = numpy.arange( 0, t + 1 ) 
        print( f"numpy s = {s}")
        print( "numpy facotrial(0) = ", factorial( s ) )
    elif False :
        print_cpu_info()    
        print()
        print_gpu_info()
    elif False :
        t = 3
        s = torch.arange( 0, t + 1 ) 
        t = torch.arange( 0, t + 1 ) 

        print( s*t )
    elif True : # create_zernike_pyramid
        use_gpu = 1
        use_hash = 1

        row_cnt = 7
        col_cnt = 4

        circle_type = "inner"
        img_type = "real"

        test_zernike_pyramid( row_cnt, col_cnt, circle_type, img_type, use_gpu=use_gpu, use_hash=use_hash )
    pass

pass