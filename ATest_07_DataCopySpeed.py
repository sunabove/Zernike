import os, math, psutil
from time import *
from time import perf_counter
from matplotlib import pyplot as plt

import torch

from ACommon import *

print( "Hello..." )

def test_data_copy_speed( device_fr_to_list , Ks, debug=0 ) :

    fs = fontsize = 16
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = fontsize

    row_cnt = 1; col_cnt = 1
    
    fig, charts = plt.subplots( row_cnt, col_cnt, figsize=( 6*col_cnt, 6*row_cnt), tight_layout=1 )
    
    charts = charts.flatten() if row_cnt*col_cnt > 1 else [charts]
    chart_idx = 0
    chart = charts[ chart_idx ]
    chart_idx +=1 
    
    for [ device_fr, device_to ] in device_fr_to_list :

        free_mem_bytes, total_mem_bytes, free_ratio = get_free_mem_bytes( 0, device=0, verbose=0 ) 
        
        free_mem_bytes_prev = free_mem_bytes
        print( f"free_mem_bytes = {free_mem_bytes:_} bytes {free_ratio:.0%}%", flush=1 )

        memory_sizes = [ ]
        run_times = []

        for K in Ks : 
            memory_size = 0 
            run_time = 0
            error = None
            
            if debug : print( f"{K}K", flush=1 )

            resolution = 1_000*K
            dtype=torch.complex64

            a = torch.zeros( (resolution, resolution), dtype=dtype, device=device_fr )
            memory_size = a.nbytes

            for _ in range( 100 ) : 
                then = time.time()

                b = torch.tensor( a, device=device_to )

                run_time += time.time() - then 

                del b
                b = None
            pass

            del a 
            a = None 

            torch.cuda.empty_cache()

            print( f"device fr = {device_fr}, to = {device_to},  size = {memory_size/1e9:_.1f} Gb, run_time = {run_time:.2f} (sec.)", flush=1 )

            memory_sizes.append( memory_size )
            run_times.append( run_time )
        pass # K

        x = Ks 
        y1 = torch.tensor( memory_sizes )
        y2 = torch.log10( torch.tensor( run_times ) )

        chart.plot( x, y1, marker="D", label= f"{device_fr} to {device_to} Memory size(Gb)" ) 
        chart.plot( x, y2, marker="s", label= f"{device_fr} to {device_to} Run-time(sec.)" ) 

        chart.set_xticks( x )
        
    pass # device_fr_to_list
    
    chart.set_title( f"Data Copy Speed" )
    chart.set_xlabel( f"Grid Tick Count" ) 
    chart.set_xlabel( f"Grid Tick Count" ) 

    chart.grid( axis='2', linestyle="dotted" )
    chart.grid( axis='y', linestyle="dotted" )
    
    chart.legend( fontsize=fs-2)
    
    src_dir = os.path.dirname( os.path.abspath(__file__) )
    result_figure_file = f"{src_dir}/result/data_copy_speed.png"
    print( f"Result figure file = {result_figure_file}" )
    plt.savefig( result_figure_file )
    
    plt.show(); 

    print( "\nDone." )
pass # -- test_array_memory

if __name__ == "__main__":
    test_data_copy_speed( use_gpu=use_gpu, operation=operation, debug=1, verbose=0 )
pass

