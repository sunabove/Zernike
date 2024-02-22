from AZernike import *

# 저니크 모멘트 함수 실험 
def test_zernike_moments_calc_times( datas, use_gpus, use_hashs, Ks, Ts, debug=0 ) : 
    
    if is_array( use_gpus ) :
        for use_gpu in use_gpus :
            test_zernike_moments_calc_times( datas, use_gpu, use_hashs, Ks, Ts, debug=debug )
        pass
    elif is_array( use_hashs ) :
        for use_hash in use_hashs :
            test_zernike_moments_calc_times( datas, use_gpus, use_hash, Ks, Ts, debug=debug )
        pass 
    else :
        use_hash = use_hashs
        use_gpu = use_gpus
        device_no = 0

        hash = {} if use_gpu else None
        device = torch.device( f"cuda:{device_no}" ) if use_gpu else torch.device( f"cpu" )
        dn = device_name = "GPU" if use_gpu else "CPU"
        
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
            pass    
            
            if not key in datas :
                print()
                print( f"device={key}", flush=True)
            
                data = {}
                data[ "Ks" ] = Ks
                data[ "Ts" ] = Ts
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

            img = torch.tensor( img, dtype=torch.complex64, device=device )
        
            for T in Ts :
                moments, run_time = calc_moments(T, img, rho, theta, dx, dy, device=device, hash=hash, debug=debug )
                
                print( f"K = {K:2}, T = {T:2}, Run-time = {run_time:7.2f} (sec.)" ) 
                
                run_times.append( run_time )
            pass # T
        pass # K
    pass
pass # test_zernike_moments_calc_times

def test_plot_zernike_moment_calc_times( datas ) : 

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
        marker = "*"
        color = "b"
        
        if "cuda" in label :
            linestyle = "dotted"
            color = "g"
            marker = "s" 
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
        
        chart.set_xlabel( "Grid Tick Count" )
        chart.set_ylabel( "$Log_{10}(y)$ (sec.)")
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
pass # test_plot_zernike_moment_calc_times
