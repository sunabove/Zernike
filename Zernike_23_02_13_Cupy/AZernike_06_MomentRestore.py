# 저니크 모멘트 계산 

from AZernike import *

# 차수에 따른 연산 속도 

print_curr_time()

def test_moment_restore(img_infos, img_input, Ks, Ts, **options) :
    print( line2 )
    print( f"use_gpu = { options['use_gpu'] }, use_hash={ options['use_hash'] }, use_thread = { options['use_thread'] }" )

    #{ "name" : "lena",         "img" : rgb2gray( io.imread( 'image/lenna.png' ) ) } , 
    
    use_gpu = get_option( "use_gpu", **options )

    for K in Ks : 
        print( line2 )
        
        options[ "hash" ] = {}

        circle_type = "outer"

        rho, theta, x, y, dx, dy, k, area = rho_theta( 1000*K, circle_type, **options ) 

        img = cv.resize( img_input, (int(K*1_000), int(K*1_000)), interpolation=cv.INTER_AREA )
        
        np = cupy if use_gpu else numpy
        
        if use_gpu :
            img = cupy.asarray( img )
        pass

        # image scaled
        img_infos.append( { "title" : f"Origianl Image (K={K})", "img" : img, "K" : K } )

        for T in Ts : 
            then = perf_counter()

            moments, moment_run_time = calc_moments(T, img, rho, theta, dx, dy, **options )
            img_restored, psnr_run_time = restore_image(moments, rho, theta, **options )
            
            t_img = img_restored.real
            psnr = calc_psnr( img, t_img, **options )
            mad = calc_mad( img, t_img, **options )
            
            img_info = { "title" : f"K={K}, T={T}, PSNR={psnr:.2f}", "img" : t_img, "psnr" : psnr, "K" : K, "T" : T }
            img_info[ "mad" ] = mad
            img_info[ "moment_run_time" ] = moment_run_time 
            img_info[ "psnr_rum_time" ] = psnr_run_time 
            
            img_infos.append( img_info ) 

            elapsed = perf_counter() - then

            print( f"K = {K}, T = {T:2d}, Elapsed = {elapsed:.2f}(sec.), psnr = {psnr:.4f}" )
        pass
        
        if "hash" in options : 
            del options[ "hash" ]
        pass
    pass
pass # test_moment_restore

def plot_moment_restore(img_infos, Ks, Ts, **options ) :
    # 서브 챠트 생성 
    col_cnt = len(Ks) if len(Ks) > len(Ts) else len(Ts)
    col_cnt += 2 

    max_col_cnt = 5
    if col_cnt > max_col_cnt :
        col_cnt = max_col_cnt

    row_cnt = len( img_infos ) // col_cnt
    if col_cnt*row_cnt < len( img_infos ) + len(Ks) :
        row_cnt += 1
    pass

    w = 2.5
    fig, charts = plt.subplots( row_cnt, col_cnt, figsize=(w*col_cnt, w*row_cnt) )
    charts = charts.ravel() if row_cnt*col_cnt > 1 else [charts]
    chart_idx = 0 

    K_prev = None 
    psnrs = []
    mads = []
    K = None
    
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

    plt.tight_layout(); plt.show() 
pass