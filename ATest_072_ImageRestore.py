from AZernike import *

def test_image_restore( img_lbls, Ks, col_cnt=4, row_cnt=2, step=4, use_cache=1, debug=0 ) :
    print( line2 )
    
    use_gpu = 1
    circle_type = "outer"

    device_no = 0 
    device = torch.device( f"cuda:{device_no}" ) if use_gpu else torch.device( f"cpu" )

    Ps = torch.arange( step, step*(row_cnt*col_cnt -1), step )
    
    cache = { } if use_cache else None 

    if cache is not None :
        # zernike cache load on cpu
        load_vpq_cache( max(Ps), Ks, circle_type, cache, device=torch.device("cpu"), debug=debug)
    pass

    if use_gpu :
        warm_up_gpus( debug=1 )
    pass

    # 서브 챠트 생성 
    for i_idx, [ img_org, img_label ] in enumerate( img_lbls ): 
        
        fs = fontsize = 16 ; w = 2.5
        fig, charts = plt.subplots( row_cnt, col_cnt, figsize=(w*col_cnt, w*row_cnt), tight_layout=0 )
        charts = charts.ravel() if row_cnt*col_cnt > 1 else [charts]
    
        shape = img_org.shape
        channel = shape[ -1 ] if len( shape ) > 2 else 1

        print( f"[{i_idx:2d}] {img_label} img org shape = {img_org.shape}, min={numpy.min(img_org)}, max={numpy.max(img_org)}" )
        
        if channel == 3 : 
            img_org = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
            #img_org = skimage.color.rgb2gray( img_org ) 
        pass

        print( f"[{i_idx:2d}] {img_label} img gry shape = {img_org.shape}, min={numpy.min(img_org)}, max={numpy.max(img_org)}" )  

        for kidx, K in enumerate( Ks ) : 
            print( line2 )

            chart = charts[ (kidx)*row_cnt*col_cnt ] 
            chart.imshow( img_org, cmap="gray" )
            chart.set_title( f"Image Org.", fontsize=fs )
            img_width  = img_org.shape[1]
            img_height = img_org.shape[0]
            chart.set_xticks( torch.arange( 0, img_width, math.pow(10, int(math.log10(img_width) ) ) ) )
            chart.set_yticks( torch.arange( 0, img_height, math.pow(10, int(math.log10(img_height) ) ) ) )

            resolution = int( 1000*K )

            grid = rho_theta( resolution, circle_type, device, debug=debug ) 

            img = cv.resize( img_org + 0, (resolution, resolution), interpolation=cv.INTER_AREA )

            psnrs = []

            for pidx, P in enumerate( Ps ) : 
                if cache is not None and use_gpu :
                    # zernike cache load on gpu
                    load_vpq_cache( P, K, circle_type, cache, device=torch.device("cuda:0"), debug=debug)
                pass
            
                then = time.time()

                moments, run_time = calc_moments( img, P, resolution, circle_type, device=device, cache=cache, debug=debug )
                img_restored, restore_run_time = restore_image( moments, grid, device, cache, debug=debug )
                
                img_real = img_restored.real
                #img_real = torch.abs( img_restored )

                psnr = calc_psnr( img, img_real )
                rmse = calc_rmse( img, img_real ) 

                elapsed = time.time() - then

                psnrs.append( int( psnr ) )

                print( f"K = {K}, P = {P:02d}, elapsed = {elapsed:.2f}(sec.), psnr = {psnr:7.3f}, rmse = {rmse:.1e}, img restored min = {torch.min( img_real):.1f}, max = {torch.max( img_real):.1f}", flush=1 )

                chart = charts[ kidx**row_cnt*col_cnt + pidx + 1 ] 
                img_cpu = img_real.cpu().numpy()
                im = chart.imshow( img_cpu, cmap="gray" )
                if 0 : plt.colorbar(im)
                chart.set_title( f"$PSNR = {psnr:.1f} ({P} P)$", fontsize=fs )
                #chart.set_xlabel( f"${K}K,{P}P$", fontsize=fs )
                kstep = 1000
                yticks = numpy.arange( 0, img.shape[0] + 1, kstep )[::-1]
                xticks = numpy.arange( kstep, img.shape[1] + 1, kstep )
                chart.set_yticks( yticks )
                chart.set_xticks( xticks )                
                chart.set_yticklabels( [ f"${t/1000:.0f}K$" for t in yticks ] )
                chart.set_xticklabels( [ f"${t/1000:.0f}K$" for t in xticks ] )

                pass
            pass # P

            #plot psnr
        
            chart = charts[ kidx**row_cnt*col_cnt + len(Ps) + 1 ]
            chart.plot( numpy.array( Ps.cpu() ) , numpy.array( psnrs ), marker="*" )
            chart.set_title( f"$PSNR({K}K)$", fontsize=fs )
            #chart.set_xlabel( f"$Order(P)$", fontsize=fs-4 )
            #chart.set_ylabel( f"$PSNR$", fontsize=fs-4 )
            chart.set_xticks( torch.arange( 0, max(Ps + 1), step ) )
            chart.set_xticklabels( [ f"${int(t)}P$" for t in torch.arange( 0, max(Ps + 1), step ) ])
        
        pass # K

        plt.show()

        src_dir = os.path.dirname( os.path.abspath(__file__) )
        file_stem = Path( __file__ ).stem

        result_figure_file = f"{src_dir}/result/{file_stem.lower()}_{int(max(Ps))}P_{int(max(Ks))}K_{i_idx:02}_{img_label}.png"
        plt.savefig( result_figure_file )
        print( f"\nresult_figure_file = {result_figure_file}" )
    pass # img_orgs

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