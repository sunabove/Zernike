from AZernike import *

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

        for idx, K in enumerate( Ks )  :
            error_avgs = []
            elapsed_list = []

            resolution = int( 1_000*K )

            grid = rho_theta( resolution, circle_type="inner", device=device, debug=debug )

            dx = grid.dx
            dy = grid.dy

            for P in Ps :
                pct = int( (100.0*cur_idx)/tot_idx ) 

                if 1 or debug : 
                    print( f"[ {pct:3d} % ] {device_name}, P = {P}, K = {K}, Resolution = {resolution:_}" , flush=1 )
                pass
                
                then = time.time()
                
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
    if 1 :
        T = 5 #20 #4 #5 #10 # 20 
        Ks = torch.arange( 0.5, 5.5, 0.5 )

        print( f"T = {T}, Ks = {Ks}" )
            
        use_gpu = 1
        use_hash = 1
        test_zernike_function_ortho(T, Ks, use_gpu=use_gpu, use_hash=use_hash, debug=0)
    pass
pass