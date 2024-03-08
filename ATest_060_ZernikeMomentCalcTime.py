from AZernike import *

def _get_moment_calc_time( img, P, resolution, device, circle_type="outer", cache=None, debug=0) :
    
    moments, run_time = calc_moments(img, P, resolution, circle_type, device=device, cache=cache, debug=debug )
    
    return run_time
pass # _test_moment_calc_time

# 저니크 모멘트 함수 실험 
def test_zernike_moments_calc_times( use_gpus, use_caches, Ps, Ks, debug=0 ) : 

    # 서브 챠트 생성 
    fs = fontsize = 16
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = fontsize

    row_cnt = 1
    col_cnt = 1

    fig, charts = plt.subplots( row_cnt, col_cnt, figsize=(8*col_cnt, 6*row_cnt), tight_layout=1 )
    charts = charts.ravel() if row_cnt*col_cnt > 1 else [charts]
    chart_idx = 0 
    chart = charts[ chart_idx ] ; chart_idx += 1

    markers = [ "o", "s", "p", "*", "D", "^", "X", "2", "p", "h", "+" ]
    colors  = [ mcolors.TABLEAU_COLORS[key] for key in mcolors.TABLEAU_COLORS ]

    tot_idx = len( use_gpus )*len( use_caches )*len( Ps )*len( Ks )
    cur_idx = 0 

    warm_up = { }

    tab_rows = []

    fit_datas = {}

    circle_type = "outer"

    miny = None
    maxy = None

    x = Ks
    
    for gidx, use_gpu in enumerate( use_gpus ) :

        device_no = 0
        device = torch.device( f"cuda:{device_no}" ) if use_gpu else torch.device( f"cpu" )
        dn = device_name = "GPU" if use_gpu else "CPU"

        if use_gpu :
            warm_up_gpus( debug=1 )
        pass

        for cidx, use_cache in enumerate( use_caches ): 

            fit_data = { "as" : [], "bs" : [] }
            cache_label = ""
            if 1 or len( use_caches ) > 1 :
                cache_label = ":CACHE" if use_cache else ":NOCAC"
            pass

            fit_datas[ f"{device_name}{cache_label}" ] = fit_data

            if not device in warm_up :
                # warm up device by assing temporary memory
                warm_up[ device ] = True

                temp = torch.zeros( (1_000, 1_000), dtype=torch.complex64, device=device )
                temp = 1
                del temp
                temp = None
            pass
            
            src_dir = os.path.dirname( os.path.abspath(__file__) )
            img = cv.imread( f"{src_dir}/image/lenna.png", 0 )

            if debug : print( "img shape= ", img.shape )

            cache = { } if use_cache else None 

            if cache is not None :
                # zernike cache load on cpu
                load_vpq_cache( max(Ps), Ks, circle_type, cache, device=torch.device("cpu"), debug=debug)
            pass    

            for pidx, P in enumerate( Ps ) :
                tab_row = []
                tab_rows.append( tab_row )

                tab_row.append( device_name )
                tab_row.append( "CACHE" if use_cache else "NOCAC" )

                run_times = [ ]

                pct = float( (100.0*cur_idx)/tot_idx )

                for K in Ks :
                    cur_idx += 1
                    curr_K = K

                    resolution = int( 1_000*K )

                    if cache is not None and use_gpu :
                        # zernike cache load on gpu
                        load_vpq_cache( P, K, circle_type, cache, device=torch.device("cuda:0"), debug=debug)
                    pass

                    run_time = _get_moment_calc_time( img, P, resolution, device=device, circle_type=circle_type, cache=cache, debug=debug )
                    run_times.append( run_time )

                    pct = float( (100.0*cur_idx)/tot_idx )
                    run_time_human = f"{timedelta(seconds=run_time)}".split('.')[0]

                    if cache is not None and "cuda" in f"{device}" :
                        # clear gpu cache only

                        if 1 : print( f"--- clearing gpu cache resolution = {resolution}" )

                        a = cache["GPU"][resolution]

                        del cache["GPU"][resolution]
                        a = None

                        torch.cuda.empty_cache()
                    pass

                    desc = f"[ {pct:3.0f} % ] {dn}: Cache = {use_cache}, P = {P:3}, K = {K:2}, Run-time = {run_time:7.2f} (sec.) {run_time_human}"

                    if 1 : print( desc )
                pass # K

                y = torch.log10( torch.tensor( run_times ) )

                if miny is None or maxy is None :
                    miny = torch.min( y )
                    maxy = torch.max( y )
                else : 
                    miny = min( miny, torch.min( y ) )
                    maxy = max( maxy, torch.max( y ) )
                pass

                if True : # fitting line
                    import numpy
                    fit = numpy.polyfit( numpy.log10(x), numpy.array( y ), 1 )
                    mx = numpy.median( x ) + (max(x) - min(x))*.12
                    my = fit[0]*numpy.log10( mx ) + fit[1]
                    a = fit[0]
                    b = fit[1]

                    fit_data[ "as" ].append( a )
                    fit_data[ "bs" ].append( b )
                    
                    plot_fit_line = 0 

                    if plot_fit_line : 
                        sign = "+" if b >= 0 else "-"

                        fit_text = f"$y = {a:.1f}*log_{'{10}'}(x) {sign} {abs(b):.1f}$"
                        
                        x2 = numpy.linspace( min(x), max(x), 100 )
                        
                        color = colors[ pidx%len(colors) ]
                        text_color = colors[ pidx%len(colors) ]
                        linestyle = "dotted"
                        linewidth = 1.2
                    
                        chart.plot( x2, a*numpy.log10(x2) + b, color=color, linestyle=linestyle, linewidth=linewidth )
                        # chart.text( mx, my, fit_text, color=text_color, fontsize=fs-2 )
                    pass

                    tab_row.append( int( P ) )
                    tab_row.append( a )
                    tab_row.append( b )
                    
                pass # fitting line

                marker = markers[ pidx%len(markers) ]
                color = colors[ pidx%len(colors) ]
                linestyle = "solid" 
                
                if len( use_gpus ) > 1 :
                    color = colors[ gidx%len(colors) ]
                    linestyle = "solid" if use_gpu else "dotted"
                elif len( use_caches ) > 1 : 
                    color = colors[ cidx%len(colors) ]
                    linestyle = "solid" if use_cache else "dotted"
                pass

                fit_data[ "color" ] = color
                fit_data[ "linestyle" ] = linestyle
                
                label = ""
                if use_cache :
                    label = f"{dn}-CACHE: {P:2d}$P$"
                else :
                    label = f"{dn}-NOCAC: {P:2d}$P$"
                pass

                linewidth = 2

                chart.plot( x, y, marker=marker, color=color, label=label, linestyle=linestyle, linewidth=linewidth )

                run_avg = sum( run_times )/len( run_times )
                tab_row.extend( run_times )
                tab_row.append( run_avg )
                
            pass # P

            # add empty legend to distinguis legend groups
            if len( use_caches ) > 1 and ( cidx < len( use_caches ) - 1 ) :
                chart.plot( [min(x)], [miny], linewidth=0, label=" ")
            pass
        pass # use_cache 

        # add empty legend to distinguis legend groups
        if len( use_gpus ) > 1 and ( gidx < len( use_gpus ) - 1 ) :
            chart.plot( [min(x)], [miny], linewidth=0, label=" ")
        pass
    pass  # use_gpus

    title = f"Zernike Moment Run-time"
    
    if len( use_gpus ) == 1 :
        use_gpu = use_gpus[0]
        if use_gpu :
            title = "GPU " + title
        else :
            title = "CPU " + title
        pass
    pass

    chart.set_title( title )

    chart.set_xlabel( f"Grid Tick Count" )
    chart.set_ylabel( f"$log_{'{10}'}(seconds)$")

    chart.set_xticks( Ks )
    chart.set_xticklabels( [ f"${K}K$" for K in Ks ] )  
    
    chart.grid( axis='x', linestyle="dotted" )
    chart.grid( axis='y', linestyle="dotted" )

    main_legend = chart.legend( loc="upper left", labelcolor='linecolor', fontsize=fs-5 )

    if 1 :
        import matplotlib.patches as mpatches

        sub_legends = [ ]

        for pidx, key in enumerate( fit_datas ) :
            fit_data = fit_datas[ key ]
            
            fas = numpy.polyfit( numpy.array( Ps ), numpy.array( fit_data[ "as" ] ), 1 )
            fbs = numpy.polyfit( numpy.array( Ps ), numpy.array( fit_data[ "bs" ] ), 1 )

            label = f"{key}: $Y = ({fas[1]:+.2f} {fas[0]:+.3f}*P)*log_{'{10}'}(K) {fbs[0]:+.3f}*P {fbs[1]:+.2f}$"
            linestyle = fit_data[ "linestyle" ]
            color = fit_data[ "color" ]
            labelcolor = color

            legend = mpatches.Patch( label=label, linestyle=linestyle, color=color )
            sub_legends.append( legend )
        pass

        chart.legend( handles=sub_legends, loc="lower right", labelcolor="linecolor", fontsize=fs-5 )

        chart.add_artist( main_legend )
    pass

    plt.show()

    device_names = "_".join( [ ["CPU", "GPU"][use_gpu] for use_gpu in use_gpus ] )
    use_cache_names = "_".join( [ ["NOC", "CAC"][use_cache] for use_cache in use_caches ] )

    src_dir = os.path.dirname( os.path.abspath(__file__) )
    file_stem = Path( __file__ ).stem

    result_figure_file = f"{src_dir}/result/{file_stem.lower()}_{device_names}_{use_cache_names}_{int(max(Ps))}P_{int(max(Ks))}K.png"
    plt.savefig( result_figure_file )
    print( f"\nresult_figure_file = {result_figure_file}" )

    tab_header = [ "Device", "CACHE", "P", "a", "b" ]
    tab_header.extend( [ f"{int(K)} K" for K in Ks ] )
    tab_header.append( "AVG" )

    print()
    print( tabulate( tab_rows, headers=tab_header ) )
    print()

    excelData = []
    excelData.append( tab_header )
    excelData.extend( tab_rows )
    df = pd.DataFrame( excelData )
    file_stem = Path( __file__ ).stem

    excel_file = f"{src_dir}/result/{file_stem.lower()}_{device_names}_{use_cache_names}_{int(max(Ps))}P_{int(max(Ks))}K.xlsx"
    df.to_excel( excel_file, index=False, header=False )
    print( f"\nExcel file = {excel_file}" )

pass # test_plot_zernike_moment_calc_times

if __name__ == "__main__" :
    if 1 :
        Ps = torch.arange( 5, 25 + 1, 5 )
        Ks = torch.arange( 1,  6 + 1, 1 )
        Ks = torch.arange( 1,  5 + 1, 1 )

        Ps = torch.arange( 5, 30 + 1, 5 )
        Ks = torch.arange( 1,  6 + 1, 1 )

        Ps = torch.arange( 5, 10 + 1, 5 )
        Ks = torch.arange( 1,  2 + 1, 1 )

        use_gpus  = [ 1 ] 
        use_caches = [ 0, 1 ]

        use_gpus   = [ 1 ] 
        use_caches = [ 1 ]

        debug = 0

        test_zernike_moments_calc_times( use_gpus, use_caches, Ps, Ks, debug=debug )

        print( "Done." )
    pass
pass