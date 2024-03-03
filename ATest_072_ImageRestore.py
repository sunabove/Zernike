from AZernike import *

def test_image_restore(img_org, Ks, Ps, debug=0) :
    print( line2 )
    
    use_gpu = 1
    use_cache = 1

    device_no = 0 
    cache = {} if use_cache else None
    device = torch.device( f"cuda:{device_no}" ) if use_gpu else torch.device( f"cpu" )

    # 서브 챠트 생성 
    col_cnt = 5
    
    row_cnt = int( ( len( Ps ) + 1 + 0.5) / col_cnt )*len(Ks) 

    w = 2.5
    fig, charts = plt.subplots( row_cnt, col_cnt, figsize=(w*col_cnt, w*row_cnt), tight_layout=1 )
    charts = charts.ravel() if row_cnt*col_cnt > 1 else [charts]
    chart_idx = 0 

    for kidx, K in enumerate( Ks ) : 
        print( line2 )

        circle_type = "outer"

        resolution = int( 1000*K )

        grid = rho_theta( resolution, circle_type, device, debug=debug ) 

        img = cv.resize( img_org, (resolution, resolution), interpolation=cv.INTER_AREA )
        
        for pidx, P in enumerate( Ps ) : 
            then = time.time()

            moments, moment_run_time = calc_moments(P, img, rho, theta, dx, dy, **options )
            img_restored, psnr_run_time = restore_image(moments, rho, theta, **options )
            
            t_img = img_restored.real
            
            psnr = calc_psnr( img, t_img, **options )
            mad = calc_mad( img, t_img, **options )
            
            img_info = { "title" : f"K={K}, P={P}, PSNR={psnr:.2f}", "img" : t_img, "psnr" : psnr, "K" : K, "P" : P }
            img_info[ "mad" ] = mad
            img_info[ "moment_run_time" ] = moment_run_time 
            img_info[ "psnr_rum_time" ] = psnr_run_time 
            
            img_infos.append( img_info ) 

            elapsed = time.time() - then

            print( f"K = {K}, T = {T}, Elapsed = {elapsed:.2f}(sec.)" )
        pass

    pass 

    

    def plot_psnr( chart, Ts, psnrs, mads ) :
        chart.plot( Ts, psnrs, marker="D", label=f"PSNR(K={K})", color="tab:orange" )
        chart.plot( Ts, mads, marker="s", label=f"MAD(K={K})", color="tab:blue" )
        
        chart.set_title( f"PSNR(K={K})" )
        chart.set_xlabel( f"T" )
        chart.legend()
    pass
    
    for img_info in img_infos : 
        t_img = img_info[ "img" ]
        title = img_info[ "title" ]
        
        K = img_info[ "K" ]        
        
        if K_prev is not None and K != K_prev and len(Ts) > 0 : 
            chart = charts[ chart_idx ] ; chart_idx += 1
            
            plot_psnr( chart, Ts, psnrs, mads )
            
            psnrs = []
            mads = []
        pass
    
        if "psnr" in img_info : 
            psnr = img_info[ "psnr"]
            psnrs.append( psnr )
            
            mad = img_info[ "mad"] 
            mads.append( mad )
        pass
            
        colorbar = False 
        if "colorbar" in img_info :
            colorbar = img_info[ "colorbar" ]

        chart = charts[ chart_idx ] ; chart_idx += 1
        
        chart.set_title( title )
        
        pos = chart.imshow( cupy.asnumpy( t_img ) if options["use_gpu"] else t_img, cmap='gray' )
        colorbar and fig.colorbar(pos, ax=chart) 
        
        K_prev = K
    pass

    if K is not None and len(Ks) > 0 and len( psnrs ) > 0 : 
        chart = charts[ chart_idx ] ; chart_idx += 1
        
        plot_psnr( chart, Ts, psnrs, mads )
    pass

    # draw empty chart
    for chart_idx in range( chart_idx, len(charts) ) :
        chart = charts[ chart_idx ]
        chart.plot( [0,0], [0,0] )
        chart.set_axis_off()
    pass

    plt.show() 
pass # test image restore