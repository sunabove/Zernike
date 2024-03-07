from AZernike import *

def test_image_restore( img_org, Ks, Ps, use_cache=1, debug=0 ) :
    print( line2 )
    
    use_gpu = 1
    circle_type = "outer"

    device_no = 0 
    device = torch.device( f"cuda:{device_no}" ) if use_gpu else torch.device( f"cpu" )
    
    cache = { } if use_cache else None 

    if cache is not None :
        # zernike cache load on cpu
        load_vpq_cache( max(Ps), Ks, circle_type, cache, device=torch.device("cpu"), debug=debug)
    pass

    if use_gpu :
        warm_up_gpus( debug=1 )
    pass

    # 서브 챠트 생성 
    col_cnt = 5
    
    row_cnt = int( ( len( Ps ) + 1 + 0.5) / col_cnt )*len(Ks) 

    w = 2.5
    fig, charts = plt.subplots( row_cnt, col_cnt, figsize=(w*col_cnt, w*row_cnt), tight_layout=1 )
    charts = charts.ravel() if row_cnt*col_cnt > 1 else [charts] 

    for kidx, K in enumerate( Ks ) : 
        print( line2 )

        chart = charts[ col_cnt*kidx ] 
        chart.imshow( img_org, cmap="gray" )

        resolution = int( 1000*K )

        grid = rho_theta( resolution, circle_type, device, debug=debug ) 

        img = cv.resize( img_org, (resolution, resolution), interpolation=cv.INTER_AREA )

        for pidx, P in enumerate( Ps ) : 
            if cache is not None and use_gpu :
                # zernike cache load on gpu
                load_vpq_cache( P, K, circle_type, cache, device=torch.device("cuda:0"), debug=debug)
            pass
        
            then = time.time()

            moments, run_time = calc_moments( img, P, resolution, circle_type, device=device, cache=cache, debug=debug )
            img_restored, restore_run_time = restore_image( moments, grid, device, cache, debug=debug )
            
            img_real = img_restored.real

            psnr = calc_psnr( img, img_real )
            rmse = calc_rmse( img, img_real ) 

            elapsed = time.time() - then

            print( f"K = {K}, P = {P:02d}, elapsed = {elapsed:.2f}(sec.)" )

            chart = charts[ col_cnt*kidx + pidx + 1 ]
            chart.imshow( img_real.to( "cpu" ).numpy(), cmap="gray" )
        pass # P
    pass # K

    plt.show() 

    def plot_psnr( chart, Ts, psnrs, rmses ) :
        chart.plot( Ts, psnrs, marker="D", label=f"PSNR(K={K})", color="tab:orange" )
        chart.plot( Ts, rmses, marker="s", label=f"MAD(K={K})", color="tab:blue" )
        
        chart.set_title( f"PSNR(K={K})" )
        chart.set_xlabel( f"T" )
        chart.legend()
    pass
    
    for img_info in [] : 
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
 
pass # test image restore