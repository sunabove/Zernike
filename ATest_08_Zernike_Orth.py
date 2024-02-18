from AZernike import *
from tqdm import tqdm
from math import isnan

def test_process_unit_spec() :
    import psutil, igpu

    devices = [ 'CPU' ]

    frequencies = [ psutil.cpu_freq().max/1_000 ] 
    memories = [ round(psutil.virtual_memory().total/1e9/10, 2) ]

    for gpu in igpu.devices() : 
        devices.append( f"GPU {gpu.index}" )
        frequencies.append( gpu.clocks.max_graphics/1_000 )
        memories.append( gpu.memory.total/1_000/10 )
    pass

    fs = fontsize =16
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = fontsize

    row_cnt = 1; col_cnt = 1
    fig, charts = plt.subplots( row_cnt, col_cnt, figsize=( 8*col_cnt, 6*row_cnt) )
    charts = charts.flatten() if row_cnt*col_cnt > 1 else [charts]
    chart_idx = 0 
    chart = charts[ chart_idx ]

    w = 0.4
    n = 2
    x = torch.arange(0, len(devices), dtype=torch.float64 )
    x += w*((1.0 - n )/2.0)

    yticks = torch.arange( 0, math.ceil( max( max(frequencies), max(memories)) ), 2 )

    bar = chart.bar( x - w/2 + w*0, memories, width=w, label="Memory (10 Gb)" )
    chart.bar_label( bar, fmt='%.1f', fontsize=fs-2 )
    bar = chart.bar( x - w/2 + w*1, frequencies, width=w, label="Frq. (GHZ)" )
    chart.bar_label( bar, fmt='%.1f', fontsize=fs-2 )

    chart.set_xticks( x, devices )
    chart.set_yticks( yticks )
    chart.grid( axis='y', linestyle="dotted" )
    chart.set_title( "Process Unit Specification" )
    chart.set_xlabel( "Process Unit Name")

    chart.legend()

    plt.tight_layout()
    plt.show()
pass

def test_radial_function_validation() :
    debug = 0

    datas = []

    datas.append( { "order" : (0, 0), "coeffs" : [1] })
    datas.append( { "order" : (1, 1), "coeffs" : [1] })
    datas.append( { "order" : (2, 0), "coeffs" : [2, -1] })
    datas.append( { "order" : (2, 2), "coeffs" : [1] })

    datas.append( { "order" : (3, 1), "coeffs" : [3, -2] })
    datas.append( { "order" : (3, 3), "coeffs" : [1] })

    datas.append( { "order" : (4, 0), "coeffs" : [6, -6, 1] })
    datas.append( { "order" : (4, 2), "coeffs" : [4, -3] })
    datas.append( { "order" : (4, 4), "coeffs" : [1] })

    datas.append( { "order" : (5, 1), "coeffs" : [10, -12, 3] })
    datas.append( { "order" : (5, 3), "coeffs" : [5, -4] })
    datas.append( { "order" : (5, 5), "coeffs" : [1] })

    datas.append( { "order" : (6, 0), "coeffs" : [20, -30, 12, -1] })
    datas.append( { "order" : (6, 2), "coeffs" : [15, -20, 6] })
    datas.append( { "order" : (6, 4), "coeffs" : [6, -5] })
    #datas.append( { "order" : (6, 6), "coeffs" : [1] })

    max_q = 5

    fs = fontsize = 16
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = fontsize
    row_cnt = max_q + 1 ; col_cnt = 1
    fig, charts = plt.subplots( row_cnt, col_cnt, figsize=( 8*col_cnt, 3.9*row_cnt) )
    charts = charts.flatten() if row_cnt*col_cnt > 1 else [ charts ]

    use_gpu = 1
    device_no = 0 
    hash = {} 
    device = torch.device( f"cuda:{device_no}" ) if use_gpu else torch.device( f"cpu" )

    print( f"use_gpu = {use_gpu}, device_no = {device_no}" )

    step = 1/40
    rho = torch.arange( 0, 1 + step, step, device=device )
    rho = rho[ torch.where( rho <= 1 ) ]

    markers = [ ".", "o", "s", "p", "*", "D", "d" ]

    for data in datas : 
        order  = data[ "order" ]
        coeffs = data[ "coeffs" ]
        
        p = order[0]
        q = order[1]
        
        r_pl_numeric = Rpq( p, q, rho, device=device, hash=hash, debug=debug )
        r_pl_analytic = torch.zeros_like( rho, device=device )

        for idx, coeff in enumerate( coeffs ) : 
            r_pl_analytic += coeff*torch.pow( rho, p - 2*idx )
        pass 

        chart = charts[q]
        chart.plot( rho.cpu(), r_pl_analytic.cpu(), linestyle="solid" )
        chart.plot( rho.cpu(), r_pl_numeric.cpu(), markers[p], label=r"$R_{" + f"{p}{q}" + "}$" )
        chart.set_xlim( -0.01, 1.01 )
        chart.set_ylim( -1.05, 1.05 )

        chart.set_title( rf"$q$ = {q}" )
        chart.set_ylabel( r"$R_{pq}$($\rho)$" )
        chart.set_xlabel( rf"$\rho$", fontsize=fs + 2 )
        chart.legend(loc="upper center", ncol=len(coeffs) )
        chart.legend()
    pass

    plt.tight_layout()
    plt.savefig( f"./result/zernike_01_radial_function.png" )
    plt.show()

pass

def validte_radial_polynomial_ortho( T, debug=0) : 
    print_curr_time()

    print( "\nZernike Radial polynomial orthogonality validation\n" ) 

    fs = fontsize = 16
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = fontsize

    row_cnt = 1; col_cnt = 1
    fig, charts = plt.subplots( row_cnt, col_cnt, figsize=( 9*col_cnt, 6*row_cnt) )
    charts = charts.flatten() if row_cnt*col_cnt > 1 else [charts]
    chart_idx = 0 
    chart = charts[ chart_idx ]
    
    ymin = ymax = 0

    for idx, use_gpu in enumerate( [ 1, 0 ] ) : 
        hash = {}
        device_no = 0  
        device = torch.device( f"cuda:{device_no}" ) if use_gpu else torch.device( f"cpu" )

        print( f"device = {device}" )

        memories = []
        error_avgs = []
        elapsed_list = []

        resolutions = [ 2**x for x in range( 10, 14 + 1 ) ]
        
        for grid_count in tqdm( resolutions, desc="Resolution" ):
            then = perf_counter() 

            dr = 1.0/grid_count

            rho = torch.arange( 0, 1 + dr, grid_count, device=device )
            rho = rho[ torch.where( rho <= 1 ) ]

            error_sum = 0
            error_cnt = 0 
            mem_max = 0 

            hash= {}

            for p in range( 0, T + 1 ) :
                for q in range( 0, T + 1 ) :
                    for l in range( -q, q + 1 ) :
                        if abs(l) > p or abs(l) > q:
                            continue
                        elif ( p - abs(l) )%2 == 1 :
                            continue
                        elif ( q - abs(l) )%2 == 1 :
                            continue
                        pass

                        r_pl = Rpq( p, l, rho, device=device, hash=hash, debug=0 )
                        r_ql = Rpq( q, l, rho, device=device, hash=hash, debug=0 )
                        
                        mem = (r_pl.nbytes + r_ql.nbytes)
                        
                        if mem > mem_max :
                            mem_max = mem

                        sum = torch.sum( r_pl*r_ql*rho*dr )*( 2*(p + 1) )

                        expect = [0, 1][ p == q ]

                        error = torch.abs( expect - sum )

                        if not isnan( error ) : 
                            error_sum += error
                            error_cnt += torch.numel( r_pl )
                        pass
                        
                        #debug and print( f"[{p:02d}][{q:02d}] : Rpl({p}, {l:2d})*Rql({q}, {l:2d}) = {sum}, exptect = {expect}, error = {error}", flush=1 )
                    pass
                pass
            pass

            del hash

            error_avg = error_sum/error_cnt
            error_avgs.append( error_avg )
            memories.append( mem_max )

            elapsed = perf_counter() - then
            elapsed_list.append( elapsed )

            if debug : 
                print( line2 )
                print( f"device = {device}" )
                print( f"Radial Grid Count = {grid_count:_}, T = {T}" )
                print( f"Elapsed time = {elapsed:,f}" )
                print( f"Error average = {error_avg:,.10f}" )
                print( f"Memories used = {mem_max:_} bytes" )
            pass
        pass

        x = torch.tensor( resolutions  )
        error_avgs = torch.log10( torch.tensor( error_avgs ) ) + 0.015*idx
        elapsed_list = torch.log10( torch.tensor( elapsed_list ) ) + 0.015*idx
        memories = torch.log10( torch.tensor( memories ) ) + 0.015*idx

        ymin_curr = min( torch.min( error_avgs ), torch.min( elapsed_list ), torch.min( memories ) )
        ymax_curr = max( torch.max( error_avgs ), torch.max( elapsed_list ), torch.max( memories ) )
        
        ymin = min( ymin, ymin_curr )
        ymax = max( ymax, ymax_curr )

        dn = device_name = "GPU" if use_gpu else "CPU"
        linestyle = "solid" if use_gpu else "dotted"
        color = "limegreen" if use_gpu else "violet"

        chart.plot( x, memories,     marker="*", linestyle=linestyle, label=f"{dn}:Memories(Gb)" )
        chart.plot( x, elapsed_list, marker="s", linestyle=linestyle, label=f"{dn}:Elapsed Time(sec.)" )
        chart.plot( x, error_avgs,   marker="D", linestyle=linestyle, label=f"{dn}:Orthogonality Error" )
    pass

    #chart.set_ylim( int( ymin - 1 ), int( ymax + 1 ) )
    chart.set_xticks( resolutions )
    chart.set_xticklabels( [ f"{x/1_000:1.0f}$K$" for x in resolutions ] )  
    
    chart.set_title( f"Zernike Radial Polynomial Orthogonality Error ($p$={T})" )
    chart.set_xlabel( "Grid Tick Counts" )
    chart.set_ylabel( r"$log_{10}(y)$" )

    chart.grid( axis='y', linestyle="dotted" )
    chart.legend( loc="upper center", fontsize=fs-2, ncols=3 )
    chart.legend( fontsize=fs-2 )
    

    plt.tight_layout()
    plt.savefig( f"./result/zernike_02_radial_orthogonality.png" )
    plt.show()

    print_curr_time()
pass # -- validte_radial_polynomial