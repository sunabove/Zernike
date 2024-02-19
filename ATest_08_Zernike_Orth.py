from AZernike import *
from tqdm import tqdm 
import time

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
pass ## test_process_unit_spec()

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

pass ## test_radial_function_validation

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

    tab_rows = [ [], [] ]

    for idx, use_gpu in enumerate( [ 0, 1 ] ) : 
        hash = {}
        hash = None
        device_no = 0  
        device = torch.device( f"cuda:{device_no}" ) if use_gpu else torch.device( f"cpu" )

        dn = device_name = "GPU" if use_gpu else "CPU"
        
        error_avgs = []
        elapsed_list = []

        resolutions = [ 2**x for x in range( 10, 14 + 1 ) ]
        
        for grid_count in tqdm( resolutions, desc="Resolution" ):
            then = time.time()

            dr = 1.0/grid_count

            rho = torch.linspace( 0, 1, grid_count, dtype=torch.float64, device=device )
            rho = rho[ torch.where( rho <= 1 ) ]

            error_sum = 0
            error_cnt = 0

            hash= {}

            for p in range( 0, T + 1 ) :
                for q in range( 0, p + 1 ) :
                    for n in range( 0, T + 1 ) :
                        for m in range( 0, n + 1 ) :
                            if abs(q) > p or abs(m) > n:
                                continue
                            elif ( p - abs(q) )%2 == 1 :
                                continue
                            elif ( n - abs(m) )%2 == 1 :
                                continue
                            pass

                            r_pq = Rpq( p, q, rho, device=device, hash=hash, debug=0 )
                            r_nm = Rpq( n, m, rho, device=device, hash=hash, debug=0 )
                            
                            sum = torch.sum( r_pq*r_nm*rho*dr )*( 2*(p + 1) )

                            expect = [0, 1][ p == q and n == m ]

                            error = torch.abs( expect - sum )

                            if not math.isnan( error ) : 
                                error_sum += error
                                error_cnt += torch.numel( r_pq )
                            else :
                                print( f"{__file__} : Nan is encountred." )
                                True
                            pass
                            
                            if 0*debug :
                                if expect == 1 : print( line )
                                print( f"[{grid_count:04d}] R[{p:02d}][{q:02d}] , [{n:02d}][{n:02d}] : exptect = {expect}, sum = {sum}, error = {error}", flush=1 )
                            pass
                        pass
                    pass
                pass
            pass

            del hash

            error_avg = error_sum/error_cnt
            error_avgs.append( error_avg )

            elapsed = time.time() - then
            elapsed_list.append( elapsed )

            if debug : 
                print( line2 )            
                print( f"device = {device}, Radial Grid Count = {grid_count:_}, T = {T}, rho_len = {len(rho):_}" )
                print( f"Elapsed time = {elapsed:,.3f}, Error average = {error_avg:,.8f}" )
            pass
        pass

        if False : 
            tab = tab_rows[ idx*1 + 0 ]
            tab.extend( [ device_name, "Elapsed Time (sec.)" ] )
            tab.extend( elapsed_list.copy() ) 

        tab = tab_rows[ idx*1 + 0 ]
        tab.extend( [ device_name, "Error" ] )
        tab.extend( error_avgs.copy() )

        x = torch.log10( torch.tensor( resolutions  ) )
        error_avgs   = torch.tensor( error_avgs ) 
        elapsed_list = torch.tensor( elapsed_list ) 

        ymin_curr = min( torch.min( error_avgs ), torch.min( elapsed_list ) )
        ymax_curr = max( torch.max( error_avgs ), torch.max( elapsed_list ) )
        
        ymin = min( ymin, ymin_curr )
        ymax = max( ymax, ymax_curr )

        linestyle = "solid" if use_gpu else "dotted"
        color = "limegreen" if use_gpu else "violet"

        w = ( max( x ) - min( x ) )/ len( x ) * 0.45
        n = 2
        
        bar = chart.bar( x - w/2 + w*idx, error_avgs, width=w, label=f"{device_name}" )
        #chart.bar_label( bar, fmt="%.2e", fontsize=fs-2 )

        #chart.plot( x, elapsed_list, marker="s", linestyle=linestyle, label=f"{dn}:Elapsed Time(sec.)" )
        #chart.plot( x, error_avgs,   marker="D", linestyle=linestyle, label=f"{dn}:Orthogonality Error" )
        
        chart.set_xticks( x )
        chart.set_xticklabels( [ f"{x/1_000:1.0f}$K$" for x in resolutions ] )
    pass

    #chart.set_ylim( int( ymin - 1 ), int( ymax + 1 ) )
    
    chart.set_title( f"Zernike Radial Polynomial Orthogonality Error ($p$={T})" )
    chart.set_xlabel( f"Grid Tick Counts" )
    chart.set_ylabel( f"Error Average" )
    #chart.set_ylabel( r"$log_{10}(y)$" )

    chart.grid( axis='y', linestyle="dotted" )
    chart.legend( loc="center", bbox_to_anchor=(0.5, 0.35), fontsize=fs, ncols=1 )
    #chart.legend( fontsize=fs, ncols=2 )
    chart.legend()

    plt.tight_layout()
    
    src_dir = os.path.dirname( os.path.abspath(__file__) )
    result_figure_file = f"{src_dir}/result/zernike_02_radial_orthogonality.png"
    plt.savefig( result_figure_file )
    print( f"result_figure_file = {result_figure_file}" )
    
    plt.show()

    tab_header = [ "Device", "Item" ]
    tab_header.extend( [ f"{x/1_000:1.0f}K" for x in resolutions ] )

    print( tabulate( tab_rows, headers=tab_header ) )

    excelData = []
    excelData.append( tab_header )
    excelData.extend( tab_rows )
    df = pd.DataFrame( excelData )
    df.to_excel( f"{src_dir}/result/zernike_02_radial_orthogonality.xlsx", index=False, header=False, sheet_name='poly orth')

    print()
    print_curr_time()
pass # -- validte_radial_polynomial

def test_zernike_function_ortho( T, Ks, use_gpu, use_hash=0, debug = 0 ) : 
    print()

    hash = {}
    hash = None
    device_no = 0  
    device = torch.device( f"cuda:{device_no}" ) if use_gpu else torch.device( f"cpu" )
    dn = device_name = "GPU" if use_gpu else "CPU"

    print( f"device = {device_name}, hash = { hash is not None }" )

    success_ratios = []
    error_avgs = []
    elapsed_list = []

    for K in tqdm( Ks, desc="K" ) :
        resolution = int( 1_000*K )

        if 1 or debug : 
            print( line2 )
            print( f"K = {K}, Resolution = {resolution:_}, T = {T}" )
        pass
        
        then = time.time() 

        rho, theta, x, y, dx, dy, k, area = rho_theta( resolution, circle_type="inner", device=device, debug=debug )
        
        good_cnt = 0 
        fail_cnt = 0 

        error_sum = 0
        
        hash= {}
        
        for p1 in range( 0, T + 1 ) :
            for q1 in range( -p1, p1 + 1, 2 ) :
                for p2 in range( 0, T +1 ) :
                    for q2 in range( -p2, p2 + 1, 2 ) : 
                        v_pl = Vpq( p1, q1, rho, theta, device=device, hash=hash, debug=debug )
                        v_ql = Vpq( p2, q2, rho, theta, device=device, hash=hash, debug=debug )

                        sum_arr = torch.sum( torch.conj(v_pl)*v_ql )
                        sum_integration = sum_arr*dx*dy*(p1 +1)/pi
                        sum = torch.absolute( sum_integration )

                        expect = [0, 1][ p1 == p2 and q1 == q2 ]
                        error = abs(expect -sum)
                        error_sum += error
                        success = error < 1/1_000 
                        success_t = 'Good' if success else 'Fail'

                        good_cnt += success
                        fail_cnt += (not success)

                        if not use_hash :
                            del v_pl, v_ql, sum_arr, sum_integration
                        pass

                        if debug : print( f"[{p1:02d}][{q1:02d}] {success_t} : V*pl({p1}, {q1:2d})*Vpl({p2}, {q2:2d}) = {sum:.4f}, exptect = {expect}, error={error:.4f}", flush=1 )
                    pass
                pass
            pass
        pass

        del hash

        success_ratio = good_cnt/(good_cnt + fail_cnt)
        success_ratios.append( success_ratio )
        error_avg = error_sum/(good_cnt + fail_cnt)
        error_avgs.append( error_avg )
        
        elapsed = time.time() - then
        elapsed_list.append( elapsed )

        if 1 or debug : 
            print( f"Elapsed time = {elapsed:_.4f}" )
            print( f"Error avg. = {error_avg:_.10f}" )
            #print( f"Success = {success_ratio*100:.2f}%, Fail count = {fail_cnt}, Good count = {good_cnt}", flush="True" )
        pass
    pass

    print( "\nPlotting .... ", flush="True" )

    fs = fontsize = 16
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = fontsize

    row_cnt = 1; col_cnt = 1
    fig, charts = plt.subplots( row_cnt, col_cnt, figsize=( 8.1*col_cnt, 5*row_cnt) )
    charts = charts.flatten() if row_cnt*col_cnt > 1 else [charts]
    chart_idx = 0 
    chart = charts[ chart_idx ]

    Ks = torch.array( Ks )
    error_avgs = torch.log10( torch.array( error_avgs ) )
    success_ratios = torch.array( success_ratios )
    elapsed_list = torch.array( elapsed_list )
    elapsed_list = torch.log10( elapsed_list )

    chart.plot( Ks, error_avgs, marker="D", label="Orthogonality Error" )
    chart.plot( Ks, elapsed_list, marker=".", label="Elapsed Time(secs)" )
    chart.plot( Ks, success_ratios, marker="*", label="Success Ratio" )

    chart.set_title( f"\nZerinike Function Orthogonality Error (T={T}, GPU={use_gpu})\n" )
    chart.set_xlabel( "\nAxial Grid Count\n" )
    chart.set_ylabel( "log10(y)" )
    chart.set_xticks( Ks ) 
    chart.set_xticklabels( [ f"{x} K" for x in Ks ] ) 
    chart.grid( axis='y', linestyle="dotted" )
    chart.legend()

    plt.tight_layout()

    src_dir = os.path.dirname( os.path.abspath(__file__) )
    result_figure_file = f"{src_dir}/result/zernike_03_function_orthogonality.png"
    plt.savefig( result_figure_file )
    print( f"result_figure_file = {result_figure_file}" ) 

    plt.show()
pass # test_zernike_function_orthogonality

if __name__ == "__main__" :
    if 0 : 
        T = 40 # 40 6 #10 # 20
        validte_radial_polynomial_ortho( T, debug=1 )
    elif 1 :
        T = 5 #20 #4 #5 #10 # 20 
        Ks = torch.arange( 0.5, 5.5, 0.5 )

        print( f"T = {T}, Ks = {Ks}" )
            
        use_gpu = 1
        use_hash = 1
        test_zernike_function_ortho(T, Ks, use_gpu=use_gpu, use_hash=use_hash, debug=0)
    pass
pass