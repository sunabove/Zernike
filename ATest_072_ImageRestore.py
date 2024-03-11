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

    print( f"use_gpu = {bool(use_gpu)}" )
    if use_gpu :
        warm_up_gpus( debug=1 )
    pass

    restore_times_k = { }

    max_idx = len( img_lbls )*len(Ps)*len(Ks)
    cur_idx = 0.0 

    # 서브 챠트 생성 
    for i_idx, [ img_org, img_label ] in enumerate( img_lbls ):
    
        shape = img_org.shape
        channel = shape[ -1 ] if len( shape ) > 2 else 1

        print( f"[{i_idx:2d}] {img_label} img org shape = {img_org.shape}, min={numpy.min(img_org)}, max={numpy.max(img_org)}" )
        
        if channel == 3 : 
            img_org = cv2.cvtColor(img_org, cv2.COLOR_RGB2GRAY)
            #img_org = skimage.color.rgb2gray( img_org ) 
        pass

        print( f"[{i_idx:2d}] {img_label} img gry shape = {img_org.shape}, min={numpy.min(img_org)}, max={numpy.max(img_org)}" )  

        for kidx, K in enumerate( Ks ) : 
            print( line2 )

            K = int(K)

            if K not in restore_times_k :
                restore_times_k[ K ] = [ ]
            pass

            fs = fontsize = 16 ; w = 2.5
            fig, charts = plt.subplots( row_cnt, col_cnt, figsize=(w*col_cnt, w*row_cnt), tight_layout=0 )
            charts = charts.ravel() if row_cnt*col_cnt > 1 else [charts]
            markers = [ "o", "s", "p", "*", "D", "^", "X", "2", "p", "h", "+" ]
            colors  = [ mcolors.TABLEAU_COLORS[key] for key in mcolors.TABLEAU_COLORS ]

            chart = charts[ 0 ] 
            chart.imshow( img_org, cmap="gray" )
            chart.set_title( f"[{img_label.capitalize()}] Image Org.", fontsize=fs )
            img_width  = img_org.shape[1]
            img_height = img_org.shape[0]
            chart.set_xticks( torch.arange( 0, img_width, math.pow(10, int(math.log10(img_width) ) ) ) )
            chart.set_yticks( torch.arange( 0, img_height, math.pow(10, int(math.log10(img_height) ) ) ) )

            resolution = int( 1000*K )

            grid = rho_theta( resolution, circle_type, device, debug=debug ) 

            img = cv.resize( img_org , (resolution, resolution), interpolation=cv.INTER_AREA )

            psnrs = [ ]
            rmses = [ ]
            restore_times = [ ]

            for pidx, P in enumerate( Ps ) : 
                cur_idx += 1
                pct = cur_idx/max_idx

                if cache is not None and use_gpu :
                    # zernike cache load on gpu
                    load_vpq_cache( P, K, circle_type, cache, device=torch.device("cuda:0"), debug=debug)
                pass
            
                then = time.time()

                moments, run_time = calc_moments( img, P, resolution, circle_type, device=device, cache=cache, debug=debug )
                img_restored, restore_time = restore_image( moments, grid, device, cache, debug=debug )
                
                img_real = img_restored.real
                #img_real = torch.abs( img_restored )

                img_real = cv.resize( numpy.array( img_real.cpu() ), img_org.shape[:2], interpolation=cv.INTER_AREA )

                psnr = calc_psnr( img_org, img_real )
                rmse = calc_rmse( img_org, img_real )

                psnrs.append( float( psnr ) )
                rmses.append( float( rmse ) )
                restore_times.append( restore_time )

                elapsed = time.time() - then

                print( f"[ {pct*100:5.1f} % ]  K = {K}, P = {P:02d}, elapsed = {elapsed:7.2f}(sec.), restore = {restore_time:7.2f}(sec.), psnr = {psnr:7.3f}, rmse = {rmse:.1e}, img restored min = {numpy.min( img_real):.1f}, max = {numpy.max( img_real):.1f}", flush=1 )

                chart = charts[ pidx + 1 ] 
                img_cpu = img_real
                im = chart.imshow( img_cpu, cmap="gray" )
                if 0 : plt.colorbar(im)
                chart.set_title( f"$PSNR={psnr:.1f} ({P} P, {K} K)$", fontsize=fs )
                #chart.set_xlabel( f"${K}K,{P}P$", fontsize=fs )

                if 1 or pidx == 0 : 
                    kstep = 100
                    yticks = numpy.arange( 0, img_cpu.shape[0] + 1, kstep )
                    xticks = numpy.arange( kstep, img_cpu.shape[1] + 1, kstep )
                    chart.set_yticks( yticks )
                    chart.set_xticks( xticks )
                    chart.set_yticklabels( [ f"${t/1:.0f}$" for t in numpy.flip( yticks ) ] )
                    chart.set_xticklabels( [ f"${t/1:.0f}$" for t in xticks ] )
                elif 1 :
                    chart.set_yticks( [] )
                    chart.set_xticks( [] )
                pass

                pass
            pass # P

            restore_times_k[ K ].append( numpy.array( restore_times ) )

            #plot psnr
        
            chart = charts[ -1 ]
            x = numpy.array( Ps.cpu() )
            cidx = 0 
            ls = "solid"
            chart.plot( x, numpy.array( rmses ), marker=markers[cidx%len(markers)], label="$RMSE$", linestyle=ls )
            cidx += 1
            chart.plot( x, numpy.array( psnrs )*5, marker=markers[cidx%len(markers)], label="$PSNR*5$", linestyle=ls )
            cidx += 1

            if 0 :
                chart.plot( x, numpy.array( restore_times )*10, marker=markers[cidx%len(markers)], label="Restore time(centi sec.)$", linestyle=ls )
                cidx += 1
            pass

            chart.set_title( f"Restoration Rate$({K}K)$", fontsize=fs )
            #chart.set_xlabel( f"$Order(P)$", fontsize=fs-4 )
            #chart.set_ylabel( f"$PSNR$", fontsize=fs-4 )
            xticks = torch.linspace( min(Ps), max(Ps), 5 )
            chart.set_xticks( xticks )
            chart.set_xticklabels( [ f"${int(t)}P$" for t in xticks ])
            chart.legend( fontsize=fontsize/2 )

            plt.show()

            src_dir = os.path.dirname( os.path.abspath(__file__) )
            file_stem = Path( __file__ ).stem

            result_figure_file = f"{src_dir}/result/{file_stem.lower()}_{int(max(Ps))}P_{K}K_{i_idx:02}_{img_label}.png"
            plt.savefig( result_figure_file )
            print( f"\nresult_figure_file = {result_figure_file}" )
        
        pass # K 
    pass # img_orgs

    # plot restore times of k resolution vs. order p

    fs = fontsize = 16 ; w = 2.5
    row_cnt = 1
    col_cnt = 1
    fig, charts = plt.subplots( row_cnt, col_cnt, figsize=(8*col_cnt, 6*row_cnt), tight_layout=0 )
    charts = charts.ravel() if row_cnt*col_cnt > 1 else [charts]
    markers = [ "o", "s", "p", "*", "D", "^", "X", "2", "p", "h", "+" ]
    colors  = [ mcolors.TABLEAU_COLORS[key] for key in mcolors.TABLEAU_COLORS ]

    chart = charts[ 0 ] 
    
    for cidx, K in enumerate( Ks ) :
        K = int(K)

        restore_times_sum = numpy.zeros( len(Ps), dtype=numpy.float_ )
        
        for restore_times in restore_times_k[ K ] :
            restore_times_sum += restore_times
        pass
        
        restore_times_avg = restore_times_sum/len( restore_times_k[ K ] )

        cidx = 0 
        ls = "solid"
        x = numpy.array( Ps.cpu() )
        chart.plot( x, restore_times_avg, marker=markers[cidx%len(markers)], label=f"Grid Tick Count: ${int(K)} K$", linestyle=ls )
    pass

    chart.set_title( f"Restoration Time", fontsize=fs )
    chart.set_xlabel( f"Moment Order", fontsize=fs-2 )
    chart.set_ylabel( f"Times(sec.)", fontsize=fs-2 )

    xticks = numpy.array( Ps )
    chart.set_xticks( xticks )
    chart.set_xticklabels( [ f"${int(t)}P$" for t in xticks ])
    chart.legend( fontsize=fontsize - 2 )

    plt.show()

    src_dir = os.path.dirname( os.path.abspath(__file__) )
    file_stem = Path( __file__ ).stem

    result_figure_file = f"{src_dir}/result/{file_stem.lower()}_{int(max(Ps))}P_{int(max(Ks))}K_restore_times.png"
    plt.savefig( result_figure_file )
    print( f"\nresult_figure_file = {result_figure_file}" )
 
pass # test image restore