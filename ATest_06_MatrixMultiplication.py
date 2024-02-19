import os, gc, math, psutil
from time import *
from time import perf_counter

import torch

from ACommon import *

def test_memory_multiplication_performance( device_names, debug=1 ) :

    from matplotlib import pyplot as plt
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = "16"
    row_cnt = 1; col_cnt = 1
    fig, charts = plt.subplots( row_cnt, col_cnt, figsize=( 8*col_cnt, 6*row_cnt) )
    charts = charts.flatten() if row_cnt*col_cnt > 1 else [charts]
    chart_idx = 0 
    chart = charts[ chart_idx ]
    
    for device_name in device_names : 

        alloc_sizes = []
        durations = []
        tick_counts = [] 

        data_types = [ torch.int, torch.double, torch.cfloat, torch.cdouble ]
        data_type_strs = []

        use_gpu = "gpu" in device_name.lower()

        device = torch.device( "cuda" if use_gpu else "cpu" )
        
        for data_type in data_types : 
            print( f"Device = {device_name}" )

            array = torch.zeros( [1], dtype=data_type, device=device )
            data_type_size = array.nbytes

            del array
            array = None
            
            data_type_str = f"{data_type}".split( " " )[-1].split(".")[-1].split( "'")[0]
            type_detail_str = device_name + " " + data_type_str + f"({data_type_size} bytes)"
            data_type_strs.append( data_type_str )
            
            debug and print( type_detail_str, flush=1 )

            free_mem, total_mem, free_ratio = get_free_mem_bytes( use_gpu, device=0, verbose=0 )
            free_mem_prev = free_mem
            print( f"free mem = {free_mem:_} bytes {free_ratio:.0%}", flush=1 )
        
            tick_count = int( math.sqrt( free_mem*.95/data_type_size/3 ) )

            then = perf_counter()

            print( f"Tick count = {tick_count:_}" )

            x = torch.ones( (tick_count, tick_count), dtype=data_type, device=device )
            y = torch.ones( (tick_count, tick_count), dtype=data_type, device=device )
            z = torch.multiply( x, y )

            arrays = [ x, y, z ]

            alloc_size = 0 

            for array in arrays :
                alloc_size += array.nbytes
                
                del array
            pass

            x = y = z = None
            del arrays

            import gc
            gc.collect()

            if use_gpu : torch.cuda.empty_cache()

            elapsed = perf_counter() - then

            alloc_sizes.append( alloc_size )
            tick_counts.append( tick_count )
            durations.append( elapsed )

            print( f"Memory alloc. = {alloc_size/1e9:.2f} Gb", f" Elapsed = {elapsed:.2f} seconds" )
            print()
        pass 

        x = torch.arange( len(data_types) )

        alloc_sizes = torch.tensor( alloc_sizes )/1e9
        tick_counts = torch.tensor( tick_counts )/1_000
        durations = torch.tensor( durations ) 

        linestyle = "solid" if use_gpu else "dotted"
        color = "orange" if use_gpu else "green"

        chart.plot( x, alloc_sizes, marker="D", color=color, linestyle="solid",  label=f"{device_name} Memory (Gb)" )
        chart.plot( x, tick_counts, marker="s", color=color, linestyle="dashed", label=f"{device_name} Tick Count (K)" )
        chart.plot( x, durations,   marker="*", color=color, linestyle="dotted", label=f"{device_name} Run-Time (sec.)" )

        chart.set_xticks( x )
        chart.set_xticklabels( data_type_strs, fontsize=13 )
        chart.set_xlabel( "Data Types" )

        chart.grid( axis='y', linestyle="dotted" )
        
    pass # device

    title = f"Array Multiplication Performance"
    chart.set_title( title )
    #chart.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=n)
    chart.legend( fontsize=13 )

    plt.tight_layout()

    src_dir = os.path.dirname( os.path.abspath(__file__) )
    result_figure_file = f"{src_dir}/result/array_multiplication_performance.png"
    print( f"Result figure file = {result_figure_file}" )
    plt.savefig( result_figure_file )
    
    plt.show()

pass # test_memory_multiply_performance

if __name__ == "__main__":
    devices = [ "CPU", "GPU" ]
    test_memory_multiplication_performance( devices )
pass