from AZernike import *

# 저니크 피라미드 생성 테스트
def test_zernike_pyramid( row_cnt, col_cnt, circle_type, img_type, use_gpu, use_cache=0, tight_layout=0, debug=0 ) : 
    print_curr_time()
    print( "\nZernike Pyramid Creation Validation\n" )
    
    device_no = 0 
    device = torch.device( f"cuda:{device_no}" ) if use_gpu else torch.device( f"cpu" )
    
    K = 2
    T = 2*col_cnt 

    cache = { } if use_cache else None

    if use_cache :
        cache = load_vpq_cache( T, K, circle_type, cache, device=device, debug=debug)
    pass

    print( f"use_gpu = {use_gpu}, circle_type = {circle_type}, K = {K:_}, cache = {cache != None}" )

    res = resolution = 1_000*K
    
    grid = rho_theta( resolution, circle_type, device=device, debug=0 )

    h = resolution
    w = h
    
    pq_title_imgs = [] 
    
    total_cnt = row_cnt*col_cnt
    idx = 0
    
    for p in range( 0, T + 1, 1 ) : 
        if idx >= total_cnt :
            break
        pass

        q = - p 

        for q in range( -p, p + 1, 2 ) : 
            c = (p + q)//2
            r = p - c

            if c >= col_cnt or r >= row_cnt :
                continue
            elif idx >= total_cnt :
                break
            elif idx < total_cnt :
                title = f"$Z({p}, {q})$"

                print( f"p = {p:3d}, q = {q:3d}, img type = {img_type}" )
                            
                v_pl, cache_device = Vpq( p, q, grid, device=device, cache=cache, debug=debug )

                z_img = None # zernike image
                
                z_title = f"Z_{'{'+str(p)+'}'}^{'{'+str(q)+'}'}"
                z_title = f"Z({p}, {q})"
                if "im" in img_type : 
                    z_img = v_pl.imag 

                    title = f"$Im({z_title})$"
                elif "abs" in img_type : 
                    z_img = torch.abs( v_pl )

                    title = f"$|{z_title}|$"
                else :
                    z_img = v_pl.real

                    title = f"$Re({z_title})$"
                pass 

                img = torch.zeros( (h, w), dtype=torch.float, device=cache_device )
                img_rav = img.ravel()
                img_rav[ grid.kidx ] = z_img
                
                if 1 : # z values normailization to min(-1) and max(1)
                    img_rav[0]  =  1
                    img_rav[-1] = -1
                pass

                pq_title_imgs.append( [ ( p, q ), title, img ] )

                if 0 and debug : 
                    print( f"rho size : {grid.rho.size()}" )
                    print( f"v_pl size : {v_pl.size()}" )
                    print( f"z_img size : {z_img.size()}" )
                    print( f"img size : {img.size()}" )
                    print( f"img_rav size : {img_rav.size()}" )
                pass
                
                idx += 1
            pass 
        pass # q
    pass # p

    print( "\nPlotting ...")

    fs = fontsize = 16
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = fontsize

    fig, charts = plt.subplots( row_cnt, col_cnt, figsize=( 2.5*col_cnt, 2.5*row_cnt), tight_layout=tight_layout )
    charts = charts.ravel() if row_cnt*col_cnt > 1 else [charts]
    
    cmap = "Spectral" # "jet"
    for ( p, q ), title, img in pq_title_imgs : 
        c = (p + q)//2
        r = p - c

        chart = charts[ col_cnt*r + c ]

        img = img.cpu()
                
        pos = chart.imshow( img, cmap=cmap )

        chart.set_title( f"{title}", fontsize=fs+4)

        chart.set_xticks( torch.arange( 0, res + 1, res//4 ) )
        chart.set_yticks( torch.arange( 0, res + 1, res//4 ) )

        if 1 :
            chart.set_xticklabels( [] )
            chart.set_yticklabels( [] ) 
        pass

        if p == 0 and q == 0 : 
            fig.colorbar( pos, ax=chart, format=lambda x, _: f"{x:.1f}" )

            chart.set_xlim( 0, res )
            chart.set_ylim( 0, res )

            ticks = torch.arange( 0, res + 1, res//4 )
            tick_cnt = ticks.numel()
            
            chart.set_xticks( ticks )
            chart.set_yticks( ticks )

            tick_labels = [ "" ]*tick_cnt
            tick_labels[  0 ] = '-1'
            tick_labels[ -1 ] = '1'

            if "out" in circle_type : 
                tick_labels[  0 ] = '$\\frac{-1}{\\sqrt{2}}$'
                tick_labels[ -1 ] = '$\\frac{1}{\\sqrt{2}}$'
            pass

            #chart.set_xticklabels( tick_labels, fontsize=fs )
            chart.set_yticklabels( tick_labels, fontsize=fs )
        pass
    pass

    plt.show()

    src_dir = os.path.dirname( os.path.abspath(__file__) )
    file_stem = Path( __file__ ).stem
    result_figure_file = f"{src_dir}/result/{file_stem.lower()}_{circle_type}_{K:02d}k_{img_type}.png"
    plt.savefig( result_figure_file )
    print( f"result_figure_file = {result_figure_file}" )

pass #create_zernike_pyramid

if __name__ == "__main__" :
    use_gpu = 1
    debug = 1

    col_cnt = 5 # 4
    row_cnt = 5 # 4 6 7

    circle_type = "inner"
    img_type = "real"

    use_cache = 1

    test_zernike_pyramid( row_cnt, col_cnt, circle_type, img_type, use_cache=use_cache, use_gpu=use_gpu, debug=debug )
pass