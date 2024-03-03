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
    
    for use_gpu in use_gpus :

        device_no = 0
        device = torch.device( f"cuda:{device_no}" ) if use_gpu else torch.device( f"cpu" )
        dn = device_name = "GPU" if use_gpu else "CPU"

        for use_cache in use_caches : 

            fit_data = { "as" : [], "bs" : [] }
            cache_label = ""
            if len( use_caches ) > 1 :
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

            miny = None
            maxy = None

            cache = { } if use_cache else None 

            if cache is not None :
                # zernike cache load on cpu
                load_vpq_cache( max(Ps), Ks, circle_type, cache, device=torch.device("cpu"), debug=debug)
            pass    

            for idx, P in enumerate( Ps ) :
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

                        if 1 : print( f"clearing gpu cache resolution = {resolution}" )

                        a = cache["GPU"][resolution]

                        del cache["GPU"][resolution]
                        a = None

                        torch.cuda.empty_cache()
                    pass

                    desc = f"[ {pct:3.0f} % ] {dn}: Cache = {use_cache}, P = {P:3}, K = {K:2}, Run-time = {run_time:7.2f} (sec.) {run_time_human}"

                    if 1 : print( desc )
                pass # K

                x = Ks
                y = torch.log10( torch.tensor( run_times ) )

                if idx == 0 :
                    miny = torch.min( y )
                    maxy = torch.max( y )
                else : 
                    miny = min( miny, torch.min( y ) )
                    maxy = max( maxy, torch.max( y ) )
                pass

                if True : # fitting pologon
                    import numpy
                    fit = numpy.polyfit( numpy.log10(x), numpy.array( y ), 1 )
                    mx = numpy.median( x ) + (max(x) - min(x))*.12
                    my = fit[0]*numpy.log10( mx ) + fit[1]
                    a = fit[0]
                    b = fit[1]

                    fit_data[ "as" ].append( a )
                    fit_data[ "bs" ].append( b )
                    
                    sign = "+" if b >= 0 else "-"

                    fit_text = f"$y = {a:.1f}*log_{'{10}'}(x) {sign} {abs(b):.1f}$"
                    
                    x2 = numpy.linspace( min(x), max(x), 100 )
                    
                    color = colors[ idx%len(colors) ]
                    text_color = colors[ idx%len(colors) ]
                    linestyle = "dotted"
                    linewidth = 1.2

                    chart.plot( x2, a*numpy.log10(x2) + b, color=color, linestyle=linestyle, linewidth=linewidth )
                    chart.text( mx, my, fit_text, color=text_color, fontsize=fs-2 )

                    tab_row.append( int( P ) )
                    tab_row.append( a )
                    tab_row.append( b )
                    
                pass # fit polygon

                marker = markers[ idx%len(markers) ]
                color = colors[ idx%len(colors) ]
                linestyle = "solid" if use_gpu else "dashed"

                if len( use_caches ) > 1 : 
                    linestyle = "dashed" if use_cache else "solid"
                pass

                fit_data[ "linestyle" ] = linestyle
                
                label = f"{dn}: {P:2d}$P$"
                if len( use_caches ) > 1 :
                    if use_cache :
                        label = f"{dn}-CACHE: {P:2d}$P$"
                    else :
                        label = f"{dn}-NOCAC: {P:2d}$P$"
                    pass
                pass

                linewidth = 2

                chart.plot( x, y, marker=marker, color=color, label=label, linestyle=linestyle, linewidth=linewidth )

                tab_row.extend( run_times )
            pass # P

            print()

        pass # use_cache

        print()
    
    pass  # use_gpu

    dn = ""
    if len( use_gpus ) == 1 :
        use_gpu = use_gpus[0]
        dn = "GPU" if use_gpu else "CPU"
        dn += " "
    pass

    chart.set_title( f"{dn}Zernike Moment Run-time" )

    chart.set_xlabel( f"Grid Tick Count" )
    chart.set_ylabel( f"$log_{'{10}'}(seconds)$")

    chart.set_xticks( Ks )
    chart.set_xticklabels( [ f"${K}K$" for K in Ks ] )  
    
    chart.grid( axis='x', linestyle="dotted" )
    chart.grid( axis='y', linestyle="dotted" )

    chart.legend( loc="lower center", bbox_to_anchor=(0.5, -0.36), fontsize=fs-4, ncols=3 )
    leg_1 = chart.legend( loc="upper left", fontsize=fs-4 )

    if 1 : 

        import matplotlib.patches as mpatches

        legends = [ ]
        lines = [ ]

        for idx, key in enumerate( fit_datas ) :
            fit_data = fit_datas[ key ]
            
            fa_mean = numpy.mean( fit_data[ "as" ][1:] )
            fbs = numpy.polyfit( numpy.array( Ps[1:] ), numpy.array( fit_data[ "bs" ][1:] ), 1 )

            label = f"{key}: $y = {fa_mean:.3f}*log_{'{10}'}(K) {fbs[0]:+.3f}*P {fbs[1]:+.2f}$"
            linestyle = fit_data[ "linestyle" ]

            legend = mpatches.Patch( label=label, linestyle=linestyle )
            legends.append( legend )

            line = chart.plot( [ min(Ks) ] , [ miny ] )
            lines.append( line )
        pass

        chart.legend( handles=legends, loc="lower right", fontsize=fs-4 )
        #leg_2 = chart.legend( lines, legends, fontsize=fs-4 )

        chart.add_artist(leg_1)
    pass

    plt.show()

    src_dir = os.path.dirname( os.path.abspath(__file__) )
    file_stem = Path( __file__ ).stem    
    result_figure_file = f"{src_dir}/result/{file_stem.lower()}_{device_name}_{int(max(Ps))}P_{int(max(Ks))}K.png"
    plt.savefig( result_figure_file )
    print( f"\nresult_figure_file = {result_figure_file}" )

    tab_header = [ "Device", "CACHE", "P", "a", "b" ]
    tab_header.extend( [ f"{int(K)} K" for K in Ks ] )

    print()
    print( tabulate( tab_rows, headers=tab_header ) )
    print()

    excelData = []
    excelData.append( tab_header )
    excelData.extend( tab_rows )
    df = pd.DataFrame( excelData )
    file_stem = Path( __file__ ).stem  
    excel_file = f"{src_dir}/result/{file_stem.lower()}_{device_name}_{int(max(Ps))}P_{int(max(Ks))}K.xlsx"
    df.to_excel( excel_file, index=False, header=False )
    print( f"\nExcel file = {excel_file}" )

pass # test_plot_zernike_moment_calc_times

def test_zernike_moments_calc_times_by_p( use_gpus, Ks, P, debug=0 ) : 
    
    datas = [ ]

    for use_gpu in use_gpus : 

        device_no = 0
        device = torch.device( f"cuda:{device_no}" ) if use_gpu else torch.device( f"cpu" )
        dn = device_name = "GPU" if use_gpu else "CPU"
        
        src_dir = os.path.dirname( os.path.abspath(__file__) )
        img_org = cv.imread( f"{src_dir}/image/lenna.png", 0 )

        if debug : print( "img shape= ", img_org.shape )
                        
        data = { }
        run_times = []

        data[ "device_name" ] = device_name
        data[ "run_times" ] = run_times

        datas.append( data )

        for K in Ks :
            run_time = _get_moment_calc_time( img_org, P, K, device=device, debug=debug )
            run_times.append( run_time )

            print( f"{dn}, P = {P:3}, K = {K:2}, Run-time = {run_time:7.2f} (sec.)" )
        pass

        print()
    pass

    print( "\nPlotting .... ")
        
    # 서브 챠트 생성 
    fs = fontsize = 16
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = fontsize

    row_cnt = 1
    col_cnt = 1

    fig, charts = plt.subplots( row_cnt, col_cnt, figsize=(9*col_cnt, 6*row_cnt), tight_layout=1 )
    charts = charts.ravel() if row_cnt*col_cnt > 1 else [charts]
    chart_idx = 0 
    chart = charts[ chart_idx ] ; chart_idx += 1
    
    min_y = 0
    max_y = 0 

    tab_rows = []

    for data in datas : 
        device_name = data[ "device_name" ]
        run_times = data[ "run_times" ]
        
        tab_row = []
        tab_rows.append( tab_row )

        tab_row.append( device_name )
        tab_row.append( P )
        tab_row.extend( run_times )

        label = f"{device_name}"

        linestyle = "solid"
        marker = "*"
        color = "b"
        color = None
        
        if "cuda" in device_name or "GPU" in device_name :
            #linestyle = "dotted"
            marker = "s" 
            color = "orange"
            color = None
        pass

        x = Ks
        y = torch.log10( torch.tensor( run_times ) )

        import numpy
        fit = numpy.polyfit( numpy.log10(x), numpy.array( y ), 1)

        if 1 or debug : print( f"fit = {fit}" )

        max_y = max( max_y, max(y) )
        min_y = min( min_y, min(y) )

        x2 = numpy.linspace( min(x), max(x), 100 )

        chart.plot( Ks, y, marker=marker, color=color, label=label, linestyle=linestyle )
        chart.plot( x2, fit[0]*numpy.log10(x2) + fit[1], color=color, linestyle="dashed" )

        # add annotation to log fitting line
        mx = numpy.median( x )
        my = fit[0]*numpy.log10( mx ) + fit[1]
        a = fit[0]
        b = fit[1]
        sign = "+" if b >= 0 else "-"
        chart.text( mx, my, f"$y = {a:.1f}*log_{'{10}'}(x)$ {sign} {abs(b):.1f}", fontsize=fs-2)
        
        chart.set_title( f"Zernike Moment Run-time at Order(P=${P}$)" )
        chart.set_xlabel( "Grid Tick Count" )
        chart.set_xticks( Ks )
        chart.set_xticklabels( [ f"${k}K$" for k in Ks ] )
        
        chart.set_ylabel( f"$log_{'{10}'}(seconds)$")

        chart.grid( axis='x', linestyle="dotted" )
        chart.grid( axis='y', linestyle="dotted" )
        
        #chart.legend( loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=len(datas) ) 
        chart.legend()
    pass

    max_y = max_y + 0.1 
    min_y = min( min_y, 0 )

    if min_y < 0 : min_y = math.floor( min_y )
    chart.set_ylim( min_y, max_y )

    plt.show()

    src_dir = os.path.dirname( os.path.abspath(__file__) )
    file_stem = Path( __file__ ).stem
    result_figure_file = f"{src_dir}/result/{file_stem.lower()}_{P}P.png"
    plt.savefig( result_figure_file )
    print( f"result_figure_file = {result_figure_file}" )

    tab_header = [ "Device", "P" ]
    tab_header.extend( [ f"{int(K)} K" for K in Ks ] )

    print()
    print( tabulate( tab_rows, headers=tab_header ) )
    print()

    excelData = []
    excelData.append( tab_header )
    excelData.extend( tab_rows )
    df = pd.DataFrame( excelData )
    file_stem = Path( __file__ ).stem
    excel_file = f"{src_dir}/result/{file_stem.lower()}_{P}P.xlsx"
    df.to_excel( excel_file, index=False, header=False )
    print( f"Excel file = {excel_file}" )

pass # test_zernike_moments_calc_times_by_p

def test_zernike_moments_calc_times_by_k( use_gpus, K, Ps, debug=0 ) : 
    
    datas = [ ]

    for use_gpu in use_gpus : 

        device_no = 0
        device = torch.device( f"cuda:{device_no}" ) if use_gpu else torch.device( f"cpu" )
        dn = device_name = "GPU" if use_gpu else "CPU"
        
        src_dir = os.path.dirname( os.path.abspath(__file__) )
        img_org = cv.imread( f"{src_dir}/image/lenna.png", 0 )

        if debug : print( "img shape= ", img_org.shape )
                        
        data = { }
        run_times = []

        data[ "device_name" ] = device_name
        data[ "run_times" ] = run_times

        datas.append( data )

        for P in Ps :
            run_time = _get_moment_calc_time( img_org, P, K, device=device, debug=debug )
            run_times.append( run_time )

            print( f"{dn}, P = {P:3}, K = {K:2}, Run-time = {run_time:7.2f} (sec.)" )
        pass

        print()    
    pass

    print( "\nPlotting .... ")
        
    # 서브 챠트 생성 
    fs = fontsize = 16
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = fontsize

    row_cnt = 1
    col_cnt = 1

    fig, charts = plt.subplots( row_cnt, col_cnt, figsize=(9*col_cnt, 6*row_cnt), tight_layout=1 )
    charts = charts.ravel() if row_cnt*col_cnt > 1 else [charts]
    chart_idx = 0 
    chart = charts[ chart_idx ] ; chart_idx += 1
    
    min_y = 0
    max_y = 0 

    tab_rows = []

    for data in datas : 
        device_name = data[ "device_name" ]
        run_times = data[ "run_times" ]

        tab_row = []
        tab_rows.append( tab_row )

        tab_row.append( device_name )
        tab_row.append( K )
        tab_row.extend( run_times )
        
        label = f"{device_name}"
        
        linestyle = "solid"
        marker = "*"
        color = "b"
        color = None
        
        if "cuda" in device_name or "GPU" in device_name :
            linestyle = "dotted"
            marker = "s" 
            color = "orange"
            color = None
        pass

        x = Ps
        y = torch.log10( torch.tensor( run_times ) )

        import numpy
        fit = numpy.polyfit( numpy.log10(x), numpy.array( y ), 1)

        if 1 or debug : print( f"fit = {fit}" )

        max_y = max( max_y, max(y) )
        min_y = min( min_y, min(y) )

        x2 = numpy.linspace( min(x), max(x), 100 )

        chart.plot( x, y, marker=marker, color=color, label=label, linestyle=linestyle )
        chart.plot( x2, fit[0]*numpy.log10(x2) + fit[1], color=color, linestyle="dashed" )

        # add annotation to log fitting line
        mx = numpy.median( x )
        my = fit[0]*numpy.log10( mx ) + fit[1]
        a = fit[0]
        b = fit[1]
        sign = "+" if b >= 0 else "-"
        chart.text( mx, my, f"$y = {a:.1f}*log_{'{10}'}(x)$ {sign} {abs(b):.1f}", fontsize=fs-2)
        
        chart.set_title( f"Zernike Moment Run-time at Grid Count(${K}K$)" )
        chart.set_xlabel( "Zernike Order" )
        chart.set_xticks( Ps )
        chart.set_xticklabels( [ f"${p}$" for p in Ps ] )

        chart.set_ylabel( f"$log_{'{10}'}(seconds)$")
        chart.grid( axis='x', linestyle="dotted" )
        chart.grid( axis='y', linestyle="dotted" )
        
        #chart.legend( loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=len(datas) ) 
        chart.legend()
    pass

    max_y = max_y + 0.1 
    min_y = min( min_y, 0 )

    if min_y < 0 : min_y = math.floor( min_y )

    chart.set_ylim( min_y, max_y )

    plt.show()

    src_dir = os.path.dirname( os.path.abspath(__file__) )
    file_stem = Path( __file__ ).stem
    result_figure_file = f"{src_dir}/result/{file_stem.lower}_{K}K.png"
    plt.savefig( result_figure_file )
    print( f"result_figure_file = {result_figure_file}" )

    tab_header = [ "Device", "K" ]
    tab_header.extend( [ f"{int(P)} P" for P in Ps ] )

    print()
    print( tabulate( tab_rows, headers=tab_header ) )
    print()

    excelData = []
    excelData.append( tab_header )
    excelData.extend( tab_rows )
    df = pd.DataFrame( excelData )
    excel_file = f"{src_dir}/result/{file_stem.lower()}_{K}K.xlsx"
    df.to_excel( excel_file, index=False, header=False )
    print( f"Excel file = {excel_file}" )

pass # test_zernike_moments_calc_times_by_k

if __name__ == "__main__" :
    if 0 : 
        use_gpus  = [ 1, 0 ]

        Ks = torch.arange( 1, 6 + 1, 1 )
        Ks = torch.arange( 1, 3 + 1, 1 )

        P  = 30
        P  = 10

        datas = { }

        test_zernike_moments_calc_times_by_p( use_gpus, Ks, P, debug=0 )

        print( "Done")
    elif 1 :
        Ps = torch.arange( 5, 25 + 1, 5 )
        Ks = torch.arange( 1,  6 + 1, 1 )
        Ks = torch.arange( 1,  5 + 1, 1 )

        Ps = torch.arange( 5, 10 + 1, 5 )
        Ks = torch.arange( 1,  2 + 1, 1 )

        use_gpus  = [ 1 ] 
        use_caches = [ 0 ]
        use_caches = [ 1 ]

        debug = 1

        test_zernike_moments_calc_times( use_gpus, use_caches, Ps, Ks, debug=debug )

        print( "Done." )
    pass
pass