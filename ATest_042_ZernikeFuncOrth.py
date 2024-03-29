from AZernike import *

def test_zernike_function_ortho( Ks, P, use_gpus=[0], debug = 0 ) : 
    print()

    fs = fontsize = 16
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = fontsize
    row_cnt = 1; col_cnt = 1

    markers = [ "o", "s", "p", "*", "D", "^", "X", "2", "p", "h", "+" ]

    tot_idx = len( use_gpus )*len( Ks )*len( get_pq_list(P) )*len( get_pq_list(P) )
    cur_idx = 0 
    pct = 0

    for use_gpu in use_gpus : 

        device_no = 0  
        device = torch.device( f"cuda:{device_no}" ) if use_gpu else torch.device( f"cpu" )
        dn = device_name = "GPU" if use_gpu else "CPU"

        for idx, K in enumerate( Ks )  :
            fig, charts = plt.subplots( row_cnt, col_cnt, figsize=( 8*col_cnt, 7*row_cnt), tight_layout=1 )
            charts = charts.flatten() if row_cnt*col_cnt > 1 else [charts]
            chart_idx = 0
            chart = charts[ chart_idx ]
    
            resolution = int( 1_000*K )

            grid = rho_theta( resolution, circle_type="inner", device=device, debug=debug )

            dx = grid.dx
            dy = grid.dy

            if 1 or debug : 
                print( f"[ {pct:3d} % ] {device_name}, P = {P}, K = {K}, Resolution = {resolution:_}" , flush=1 )
            pass
            
            then = time.time()
            
            pq_list = get_pq_list( P )
            nm_list = get_pq_list( P )

            array = torch.zeros( ( len(pq_list), len(nm_list) ), dtype=torch.float64, device=device )

            for i, [p, q] in enumerate( pq_list ) :
                for k, [n, m] in enumerate( nm_list ) :
                    pct = int( (100.0*cur_idx)/tot_idx ); cur_idx += 1
            
                    v_pl = Vpq( p, q, grid, device=device, debug=debug )
                    v_ql = Vpq( n, m, grid, device=device, debug=debug )

                    sum_arr = torch.sum( torch.conj(v_pl)*v_ql )
                    sum_integration = sum_arr*dx*dy*(p +1)/pi
                    sum = torch.absolute( sum_integration )

                    expect = [ 0, 1 ][ p == n and q == m ]
                    error = abs( expect - sum )

                    array[i,k] = sum

                    if True : # memory clear
                        del v_pl, v_ql, sum_arr, sum_integration
                        v_pl = v_ql = sum_arr = sum_integration = None
                    pass

                    if debug : print( f"[{pct:3}] : V*pl({p}, {q:2d})*Vpl({n}, {m:2d}) = {sum:.4f}, exptect = {expect}, error={error:.6f}", flush=1 )
                pass
            pass # pq, nm

            elapsed = time.time() - then

            cur_idx += 1
            pct = int( (100.0*cur_idx)/tot_idx )
                
            if 1 or debug : 
                run_time_human = f"{timedelta(seconds=elapsed)}".split('.')[0]
                print( f"[ {pct:3d} % ] Elapsed time = {elapsed:_.4f}, {run_time_human}" )
                #print( f"Success = {success_ratio*100:.2f}%, Fail count = {fail_cnt}, Good count = {good_cnt}", flush="True" )
            pass

            im = chart.matshow( array.cpu() )
            #chart.imshow( array.cpu() )
            #plt.imshow( arr )
            plt.colorbar(im, shrink=0.93, aspect=10, ax=chart)

            dev_info = "GPU" if use_gpu else "CPU"
            title = f"Zerinike Function Orthogonality ({dev_info}, {K}K, {P}P)"

            chart.set_title( title )
            chart.grid( axis='x', linestyle="dotted" )
            chart.grid( axis='y', linestyle="dotted" )

            plt.show()

            src_dir = os.path.dirname( os.path.abspath(__file__) )
            file_stem = Path( __file__ ).stem.lower()
            result_figure_file = f"{src_dir}/result/{file_stem}_func_orth_{dev_info}_{K}K_{P}P.png"
            plt.savefig( result_figure_file )
            print( f"\nresult_figure_file = {result_figure_file}" )

        pass # P
    pass # use_gpus

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