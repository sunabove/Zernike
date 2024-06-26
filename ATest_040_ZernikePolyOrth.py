from AZernike import *

def test_radial_polynomail_validation() :
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
    fig, charts = plt.subplots( row_cnt, col_cnt, figsize=( 8*col_cnt, 3.9*row_cnt), tight_layout=1 )
    charts = charts.flatten() if row_cnt*col_cnt > 1 else [ charts ]

    use_gpu = 1
    device_no = 0 
    hash = {} 
    device = torch.device( f"cuda:{device_no}" ) if use_gpu else torch.device( f"cpu" )

    print( f"use_gpu = {use_gpu}, device_no = {device_no}" )

    step = 1/40
    rho = torch.arange( 0, 1 + step, step, device=device )
    rho = rho[ torch.where( rho <= 1 ) ]

    colors  = [ mcolors.TABLEAU_COLORS[key] for key in mcolors.TABLEAU_COLORS ]
    markers = [ "o", "s", "p", "*", "D", "^", "X", "2", "p", "h", "+" ]

    for data in datas : 
        order  = data[ "order" ]
        coeffs = data[ "coeffs" ]
        
        p = order[0]
        q = order[1]
        
        r_pl_numeric = Rpq( p, q, rho, device=device, debug=debug )
        r_pl_analytic = torch.zeros_like( rho, device=device )

        for idx, coeff in enumerate( coeffs ) : 
            r_pl_analytic += coeff*torch.pow( rho, p - 2*idx )
        pass 

        marker = markers[ p%len(markers) ]
        color = colors[ p%len(colors) ]
        label = f"$R_{'{' + str(p) + '}'}^{'{' + str(q) + '}'}$"
        chart = charts[q]
        chart.plot( rho.cpu(), r_pl_analytic.cpu(), color=color, linestyle="solid" )
        chart.plot( rho.cpu(), r_pl_numeric.cpu(),  color=color, marker=marker, label=label )
        chart.set_xlim( -0.01, 1.01 )
        chart.set_ylim( -1.05, 1.05 )

        chart.set_title( rf"$q$ = {q}" )
        chart.set_ylabel( r"$R_{p}^{q}$($\rho)$" )
        #chart.set_ylabel( r"$R_{pq}$($\rho)$" )
        chart.set_xlabel( rf"$\rho$", fontsize=fs + 2 )
        chart.legend(loc="upper center", ncol=len(coeffs) )
        chart.legend()
    pass

    src_dir = os.path.dirname( os.path.abspath(__file__) )
    file_stem = Path( __file__ ).stem.lower()
    plt.savefig( f"{src_dir}/result/{file_stem.lower}_radial_function.png" )
    plt.show()

pass ## test_radial_function_validation

def validte_radial_polynomial_ortho( T, Ks, debug=0 ) : 
    print_curr_time()

    print( "\nZernike Radial polynomial orthogonality validation\n" ) 

    fs = fontsize = 16
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = fontsize

    row_cnt = 1; col_cnt = 1
    fig, charts = plt.subplots( row_cnt, col_cnt, figsize=( 9*col_cnt, 6*row_cnt), tight_layout=1 )
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
        
        resolutions = []
        error_avgs = []
        elapsed_list = []
        
        for K in tqdm( Ks, desc="K" ) :
            then = time.time()

            resolution = int( 1_000*K )
            resolutions.append( resolution )

            circle_type = "inner"
            
            grid = rho_theta( resolution, circle_type, device=device, debug=debug )

            ds = grid.dx*grid.dy
            rho = grid.rho

            error_sum = 0
            error_cnt = 0

            hash= {}

            for p, q in get_pq_list( T ) :
                for n, m in get_pq_list( T ) :
                    if abs(q) > p or abs(m) > n:
                        continue
                    elif ( p - abs(q) )%2 == 1 :
                        continue
                    elif ( n - abs(m) )%2 == 1 :
                        continue
                    pass

                    r_pq = Rpq( p, q, rho, device=device, debug=0 )
                    r_nm = Rpq( n, m, rho, device=device, debug=0 )
                    
                    sum = torch.sum( r_pq*r_nm*rho )*( ds*2*(p + 1) )

                    expect = [0, 1][ p == q and n == m ]

                    error = torch.abs( expect - sum )

                    if math.isnan( error ) : 
                        print( f"{__file__} : Nan is encountred." )
                        True
                    else :
                        error_sum += error
                        error_cnt += torch.numel( r_pq )
                    pass
                    
                    if 0*debug :
                        if expect == 1 : print( line )
                        print( f"[{resolution:04d}] R[{p:02d}][{q:02d}] , [{n:02d}][{n:02d}] : exptect = {expect}, sum = {sum}, error = {error}", flush=1 )
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
                print( f"device = {device}, Radial Tick Count = {resolution:_}, P = {T}, rho_len = {len(rho):_}" )
                print( f"Error average = {error_avg:,.8f}, Elapsed time = {elapsed:,.3f} {timedelta(seconds=elapsed)}" )
            pass
        pass

        if False : 
            tab = tab_rows[ idx*1 + 0 ]
            tab.extend( [ device_name, "Elapsed Time (sec.)" ] )
            tab.extend( elapsed_list.copy() ) 

        tab = tab_rows[ idx*1 + 0 ]
        tab.extend( [ device_name, "Error" ] )
        tab.extend( error_avgs.copy() )

        x = Ks.clone().detach()
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
        chart.set_xticklabels( [ f"{float(x):.0f}$K$" for x in Ks ] )
    pass

    #chart.set_ylim( int( ymin - 1 ), int( ymax + 1 ) )
    
    chart.set_title( f"Zernike Radial Polynomial Orthogonality Error ($P$={T})" )
    chart.set_xlabel( f"Grid Tick Counts" )
    chart.set_ylabel( f"Error Average" )
    #chart.set_ylabel( r"$log_{10}(y)$" )

    chart.grid( axis='y', linestyle="dotted" )
    chart.legend( loc="center", bbox_to_anchor=(0.5, 0.35), fontsize=fs, ncols=1 )
    #chart.legend( fontsize=fs, ncols=2 )
    chart.legend()

    src_dir = os.path.dirname( os.path.abspath(__file__) )
    file_stem = Path( __file__ ).stem
    result_figure_file = f"{src_dir}/result/{file_stem.lower()}_poly_orth.png"
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
    excel_file = f"{src_dir}/result/{file_stem.lower()}_poly_orth.xlsx"
    df.to_excel( excel_file, index=False, header=False )
    print( f"Excel file = {excel_file}" )

    print()
    print_curr_time()
pass # -- validte_radial_polynomial

def test_zernike_function_ortho( Ps, Ks, use_gpus=[0], debug = 0 ) : 
    print()

    fs = fontsize = 16
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = fontsize

    row_cnt = 1; col_cnt = 1
    fig, charts = plt.subplots( row_cnt, col_cnt, figsize=( 8.1*col_cnt, 5*row_cnt), tight_layout=1 )
    charts = charts.flatten() if row_cnt*col_cnt > 1 else [charts]
    markers = [ "o", "s", "p", "*", "D", "^", "X", "2", "p", "h", "+" ]

    min_y = None
    max_y = None

    tot_idx = len( use_gpus )*len( Ps )*len( Ks )
    cur_idx = 0 

    for use_gpu in use_gpus : 
        device_no = 0  
        device = torch.device( f"cuda:{device_no}" ) if use_gpu else torch.device( f"cpu" )
        dn = device_name = "GPU" if use_gpu else "CPU"

        print( f"device = {device_name}" )

        for idx, P in enumerate( Ps )  :
            error_avgs = []
            elapsed_list = []

            for K in Ks :
                pct = int( (100.0*cur_idx)/tot_idx ) 

                resolution = int( 1_000*K )

                if 1 or debug : 
                    print( f"[ {pct:3d} % ] {device_name}, P = {P}, K = {K}, Resolution = {resolution:_}" , flush=1 )
                pass
                
                then = time.time() 

                grid = rho_theta( resolution, circle_type="inner", device=device, debug=debug )

                dx = grid.dx
                dy = grid.dy
                
                error_sum = 0
                error_cnt = 0
                pq_cnt = 0 

                for p, q in get_pq_list( P ) : 
                    for n, m in get_pq_list( P ) :
                        v_pl = Vpq( p, q, grid, device=device, debug=debug )
                        v_ql = Vpq( n, m, grid, device=device, debug=debug )

                        sum_arr = torch.sum( torch.conj(v_pl)*v_ql )
                        sum_integration = sum_arr*dx*dy*(p +1)/pi
                        sum = torch.absolute( sum_integration )

                        expect = [ 0, 1 ][ p == n and q == m ]
                        error = abs( expect - sum )
                        error_sum += error

                        result = error < 1e-4

                        if not result :
                            error_cnt += 1
                        pass

                        if True : # memory clear
                            del v_pl, v_ql, sum_arr, sum_integration
                            v_pl = v_ql = sum_arr = sum_integration = None
                        pass

                        if debug : print( f"[{pq_cnt:04d}] : V*pl({p}, {q:2d})*Vpl({n}, {m:2d}) = {sum:.4f}, exptect = {expect}, error={error:.6f} result = {result}", flush=1 )

                        pq_cnt += 1
                    pass
                pass # pq

                error_avg = error_sum/pq_cnt
                error_avgs.append( error_avg )
                
                elapsed = time.time() - then
                elapsed_list.append( elapsed )

                cur_idx += 1
                pct = int( (100.0*cur_idx)/tot_idx )
                    
                if 1 or debug : 
                    run_time_human = f"{timedelta(seconds=elapsed)}".split('.')[0]
                    print( f"[ {pct:3d} % ] Error avg. = {error_avg:_.10f}, Elapsed time = {elapsed:_.4f}, {run_time_human}" )
                    #print( f"Success = {success_ratio*100:.2f}%, Fail count = {fail_cnt}, Good count = {good_cnt}", flush="True" )
                pass
            pass # K

            chart_idx = 0
            chart = charts[ chart_idx ]

            Ks = Ks.clone().detach()
            error_avgs = torch.log10( torch.tensor( error_avgs ) )
            elapsed_list = torch.log10( torch.tensor( elapsed_list ) )

            if idx == 0 :
                min_y = torch.min( error_avgs)
                max_y = torch.max( error_avgs)
            else : 
                min_y = min( min_y, torch.min( error_avgs) )
                max_y = max( max_y, torch.max( error_avgs) )
            pass

            marker = markers[ idx%len( markers ) ]
            linestyle = "solid" if use_gpu else "dashed"
            label = f"{dn}:(${P}P$)"

            chart.plot( Ks, error_avgs, marker=marker, linestyle=linestyle, label=label )
            #chart.plot( Ks, elapsed_list, marker=".", label="Elapsed Time (Sec.)" )

            chart.set_title( f"Zerinike Function Orthogonality Error Average" )
            chart.set_xlabel( "Grid Tick Count" )
            chart.set_ylabel( "Error Average : $log_{10}(y)$" )
            chart.set_xticks( Ks ) 
            chart.set_xticklabels( [ f"{int(x)}$K$" for x in Ks ] )
            chart.grid( axis='x', linestyle="dotted" )
            chart.grid( axis='y', linestyle="dotted" )
            chart.legend( fontsize=fs-2 ) 
        pass # P
    pass # use_gpus

    if 1 : 
        min_y = math.floor( min_y )
        max_y = math.ceil( max_y )

        chart.set_ylim( min_y, max_y )
    pass

    src_dir = os.path.dirname( os.path.abspath(__file__) )
    file_stem = Path( __file__ ).stem
    result_figure_file = f"{src_dir}/result/{file_stem.lower}_func_orth.png"
    plt.savefig( result_figure_file )
    print( f"\nresult_figure_file = {result_figure_file}" ) 

    plt.show()
pass # test_zernike_function_orthogonality

if __name__ == "__main__" :
    if 0 : 
        T = 10 # 40 6 #10 # 20
        test_radial_polynomail_validation( T, debug=1 )
    elif 1 :
        T = 5 #20 #4 #5 #10 # 20 
        Ks = torch.arange( 0.5, 5.5, 0.5 )

        print( f"T = {T}, Ks = {Ks}" )
            
        use_gpu = 1
        use_hash = 1
        test_zernike_function_ortho(T, Ks, use_gpu=use_gpu, use_hash=use_hash, debug=0)
    pass
pass