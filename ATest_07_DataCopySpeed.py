import os, math, psutil, time
import torch

from matplotlib import pyplot as plt
from matplotlib import colors as mcolors

from ACommon import *

print( "Hello ...\n" )

def test_data_copy_speed( Ks, debug=0 ) :

    fs = fontsize = 16
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = fontsize

    row_cnt = 1; col_cnt = 1
    
    fig, charts = plt.subplots( row_cnt, col_cnt, figsize=( 8*col_cnt, 6*row_cnt), tight_layout=1 )
    
    charts = charts.flatten() if row_cnt*col_cnt > 1 else [charts]
    chart_idx = 0
    chart = charts[ chart_idx ]
    chart_idx +=1 

    markers = [ "o", "s", "p", "*", "D", "^", "X", "2", "p", "h", "+" ]
    colors  = [ mcolors.TABLEAU_COLORS[key] for key in mcolors.TABLEAU_COLORS ]

    device_fr_to_list = [ ]
    device_fr_to_list.append( [ "cpu", "cuda:0" ] )
    device_fr_to_list.append( [ "cuda:0", "cpu" ] )
    device_fr_to_list.append( [ "cuda:0", "cuda:1" ] )

    max_y = None
    min_y = None

    tab_rows = [ ] 

    for idx, [ device_fr, device_to ] in enumerate( device_fr_to_list ):

        for use_gpu in [ 0, 1 ] :
            free_mem_bytes, total_mem_bytes, free_ratio = get_free_mem_bytes( use_gpu, device_no=0, verbose=0 ) 
        
            free_mem_bytes_prev = free_mem_bytes
            device_name = "GPU" if use_gpu else "CPU"

            print( f"{device_name} : free_mem_bytes = {free_mem_bytes:_} bytes {free_ratio:.0%}%", flush=1 )
        pass

        print()

        tab_row = [ ]
        tab_rows.append( tab_row )
        tab_row.append( device_fr )
        tab_row.append( device_to )

        memory_sizes = [ ]
        speeds = []

        device_warm_up = { } 

        if 1 :
            for device in [ device_fr, device_to ] : 
                if not device in device_warm_up :
                    # warm up device by assing temporary memory
                    device_warm_up[ device ] = True

                    temp = torch.zeros( (4_000, 4_000), dtype=torch.complex64, device=device )
                    temp = 1

                    del temp
                    temp = None

                    torch.cuda.empty_cache()
                pass
            pass
        pass

        for K in Ks :

            elapsed = 0.0
            
            resolution = 1_000*K
            dtype=torch.complex64

            a = torch.zeros( (resolution, resolution), dtype=dtype, device=device_fr )
            memory_size = a.nbytes

            repeat_cnt = 10

            for _ in range( repeat_cnt ) : 
                then = time.time()

                b = a.to( device=device_to )

                elapsed += time.time() - then 

                del b
                b = None
            pass

            del a 
            a = None 

            torch.cuda.empty_cache()

            speed = float(memory_size/1.0e6)*repeat_cnt/(elapsed)

            memory_sizes.append( memory_size )
            speeds.append( speed )

            print( f"device fr = {device_fr}, to = {device_to}, k = {K}, speed = {speed:,.4f} (Mb/s), size = {memory_size/1e6:6.2f} Mb, run_time = {elapsed:.6f} (sec.)", flush=1 )
        pass # K

        tab_row.append( sum(speeds)/len(speeds) )
        tab_row.extend( speeds )

        color = colors[1:][ idx%(len(colors) - 1) ]
        
        x = Ks 

        if idx == 0 : 
            y2 = torch.log10( torch.tensor( memory_sizes )/1e6 )
            chart.plot( x, y2, marker=markers[0], color=colors[0], linestyle="dashed", label= f"Memory size(Mb)" ) 

            for txt_idx, [ xi, yi ] in enumerate( zip(x, y2) ) :
                chart.annotate(f"{memory_sizes[txt_idx]/1e6:.0f}Mb", (xi, yi), textcoords="offset points", xytext=(0, 6), ha='center', fontsize=fs-6 )
            pass
        pass

        y1 = torch.log10( torch.tensor( speeds ) )
        label = f"[{device_fr.upper():<6} -> {device_to.upper():<6}] Speed(Mb/s)"
        chart.plot( x, y1, marker=markers[1], color=color, linestyle="solid", label=label ) 

        chart.set_xticks( x )
        chart.set_xticklabels( [ f"${k}K$" for k in Ks ], fontsize=fs-2 )

        print()
        
    pass # device_fr_to_list
    
    chart.set_title( f"Data copy speed between devices" )
    
    chart.set_xlabel( f"Grid Tick Count" )
    chart.set_ylabel( f"$log_{'{10}'}(y)$" )

    chart.grid( axis='x', linestyle="dotted" )
    chart.grid( axis='y', linestyle="dotted" )
    
    chart.legend( fontsize=fs-4 )
    chart.legend( loc="lower center", bbox_to_anchor=( 0.5, -0.32 ), fontsize=fs-3, ncols=2 )
    
    src_dir = os.path.dirname( os.path.abspath(__file__) )
    result_figure_file = f"{src_dir}/result/test_070_data_copy_speed.png"
    print( f"Result figure file = {result_figure_file}" )
    plt.savefig( result_figure_file )
    
    plt.show(); 

    # 테이블 생성
    import pandas as pd
    from tabulate import tabulate

    tab_header = [ "FROM", "TO", "AVG" ]
    tab_header.extend( [ f"{int(k)}K" for k in Ks ] )

    print( tabulate( tab_rows, headers=tab_header ) )
    print()

    excelData = []
    excelData.append( tab_header )
    excelData.extend( tab_rows )
    df = pd.DataFrame( excelData )
    excel_file = f"{src_dir}/result/test_070_data_copy_speed.xlsx"
    df.to_excel( excel_file, index=False, header=False, sheet_name='poly orth')
    print( f"Excel file = {excel_file}" )

    print( "\nDone." )
pass # -- test_array_memory

if __name__ == "__main__":
    test_data_copy_speed( use_gpu=use_gpu, operation=operation, debug=1, verbose=0 )
pass

