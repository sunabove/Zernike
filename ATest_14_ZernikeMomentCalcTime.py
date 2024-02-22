from AZernike import *

# 저니크 모멘트 함수 실험 
def test_zernike_moments_calc_times( datas, use_gpus, use_hashs, Ks, Ps, debug=0 ) : 
    
    if is_array( use_gpus ) :
        for use_gpu in use_gpus :
            test_zernike_moments_calc_times( datas, use_gpu, use_hashs, Ks, Ps, debug=debug )
        pass
    elif is_array( use_hashs ) :
        for use_hash in use_hashs :
            test_zernike_moments_calc_times( datas, use_gpus, use_hash, Ks, Ps, debug=debug )
        pass 
    else :
        use_hash = use_hashs
        use_gpu = use_gpus

        device_no = 0
        device = torch.device( f"cuda:{device_no}" ) if use_gpu else torch.device( f"cpu" )
        dn = device_name = "GPU" if use_gpu else "CPU"
        
        if is_scalar( Ks ) :
            Ks = [ Ks ]
        pass
    
        if is_scalar( Ps ) :
            Ps = [ Ps ]
        pass 
        
        img_org = cv.imread( 'image/lenna.png', 0 )

        if debug : print( "img shape= ", img_org.shape )

        for P in Ps :
                        
            key = f"{device_name},P=({P}),H={use_hash}"
            
            if not key in datas :
                print()
                print( f"key = {key}", flush=True)
            
                data = {}

                data[ "P" ] = P
                data[ "Ks" ] = Ks
                data[ "device_name" ] = device_name
                data[ "use_hash" ] = use_hash
                data[ "run_times" ] = []

                datas[ key ] = data 
            pass
        
            data = datas[ key ]
            
            run_times = data[ "run_times" ]

            if debug : 
                print( "img shape= ", img.shape ) 
                print( line )
            pass

            for K in Ks :
                hash = {} if use_hash else None
        
                circle_type = "outer"

                resolution = 1_000*K

                rho, theta, x, y, dx, dy, k, area = rho_theta( resolution, circle_type, device=device, debug=debug ) 

                if debug : print( f"rho shape = {rho.shape}" )

                img = cv.resize( img_org, ( int(resolution), int( resolution) ), interpolation=cv.INTER_AREA )

                img = torch.tensor( img, dtype=torch.complex64, device=device )
            
                moments, run_time = calc_moments(P, img, rho, theta, dx, dy, device=device, hash=hash, debug=debug )
                
                print( f"P = {P:3}, K = {K:2}, Run-time = {run_time:7.2f} (sec.)" ) 
                
                run_times.append( run_time )
            pass # T
        pass # K
    pass
pass # test_zernike_moments_calc_times

def test_plot_zernike_moment_calc_times( datas ) : 

    print( "\nPlotting .... ")
        
    # 서브 챠트 생성 
    fs = fontsize = 16
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = fontsize

    row_cnt = 1
    col_cnt = 1

    fig, charts = plt.subplots( row_cnt, col_cnt, figsize=(9*col_cnt, 8*row_cnt), tight_layout=1 )
    charts = charts.ravel() if row_cnt*col_cnt > 1 else [charts]
    chart_idx = 0 
    chart = charts[ chart_idx ] ; chart_idx += 1
    
    Ps = []
    Ks = []
    run_times_all = []
    use_hash = 0 
    device_name = "" 

    for key in datas : 
        data = datas[ key ]
        
        Ks = data[ "Ks" ]
        P = data[ "P" ]
        run_times = data[ "run_times" ]
        use_hash = data[ "use_hash" ]
        device_name = data[ "device_name" ]

        Ps.append( P )
        run_times_all.append( run_times )
    pass

    x = Ks
    y = Ps
    x, y = numpy.meshgrid( x, y )    
    z = torch.log10( torch.tensor( run_times_all ) ) 

    vmin = int( torch.min( z ) - 0.5 )
    vmax = int( torch.max( z ) + 1.5 )

    pos = chart.pcolormesh( x, y, z, vmin=vmin, vmax=vmax, cmap=plt.cm.Pastel1 )
    fig.colorbar( pos, ax=chart, label="$Log(seconds)$" )

    chart.set_title( f"Zernike Moment Run-time ({device_name}, Cache={bool(use_hash)})" )
    chart.set_xlabel( "Grid Tick Count" )
    chart.set_ylabel( "Order($P$)")

    chart.set_xticks( Ks )
    chart.set_xticklabels( [ f"${K}K$" for K in Ks ] ) 

    chart.set_yticks( Ps )
    chart.set_yticklabels( [ f"${P}$" for P in Ps ] ) 

    plt.show()
pass # test_plot_zernike_moment_calc_times

def test_plot_zernike_moment_calc_times_old( datas ) : 

    print( "\nPlotting .... ")
        
    # 서브 챠트 생성 
    fs = fontsize = 16
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = fontsize

    row_cnt = 1
    col_cnt = 1

    fig, charts = plt.subplots( row_cnt, col_cnt, figsize=(12*col_cnt, 8*row_cnt) )
    charts = charts.ravel() if row_cnt*col_cnt > 1 else [charts]
    chart_idx = 0 
    chart = charts[ chart_idx ] ; chart_idx += 1
    
    for key in datas : 
        data = datas[ key ]
        
        Ks = data[ "Ks" ]        
        run_times = data[ "run_times" ]

        P = data[ "P" ]
        device_name = data[ "device_name" ]
        use_hash = data[ "use_hash" ]
        
        label = f"{device_name},P={P},H={use_hash}"
        
        linestyle = "solid"
        marker = "*"
        color = "b"
        
        if "cuda" in device_name or "GPU" in device_name :
            linestyle = "dotted"
            marker = "s" 
            color = "orange"
        pass

        if use_hash : linestyle = "dashed"

        y = torch.log10( torch.tensor( run_times ) ) 
    
        chart.plot( Ks, y, marker=marker, color=color, label=label, linestyle=linestyle )
        
        chart.set_title( f"Zernike Moment Run-time" )
        chart.set_xlabel( "Grid Tick Count" )
        chart.set_ylabel( "$Log_{10}(y)$ (sec.)")
        chart.set_xticks( Ks )
        chart.set_xticklabels( [ f"${K} K$" for K in Ks ] )
        
        chart.legend( loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4 ) 
    pass

    plt.tight_layout()
    plt.show()
pass # test_plot_zernike_moment_calc_times_old
