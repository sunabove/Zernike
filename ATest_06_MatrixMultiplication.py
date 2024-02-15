import os, gc, math, psutil
from time import *
from time import perf_counter

import torch

def test_memory_multiplication_performance( devices ) :     

    memories = []
    durations = []
    grid_counts = []
    
    all_devices = []
    
    for device in devices : 
        device = device.lower()
    
        device_info = device.replace( "multi", "m").upper() 
            
        all_devices.append( device_info )
    
        rs = None

        scheduler = "single-threaded" 

        if device.__contains__( "multi" ) :
            scheduler = "threads"
        pass 

        s = 9_350
        
        free_memory = psutil.virtual_memory().total*0.8
        
        if device.__contains__( "cpu" ):
            #print( f"device is {device}")
            pass
        elif device.__contains__( "gpu" ) :
            #print( f"device is {device}")
            import GPUtil
    
            gpu = GPUtil.getGPUs()[0]
            
            free_memory_prev = gpu.memoryFree*1e6*3/4
            free_memory = gpu.memoryTotal*1e6*3/4 
        pass
    
        s = int( math.sqrt( free_memory/8 ) )//3

        then = perf_counter()

        print( f"device = {device}, grid_count = {s:_}, scheduler = {scheduler}, dask = {use_dask}" )

        t = np.random.rand( s*s ) + 1j*np.random.rand( s*s )
        t = t.reshape( s, s )
        if "gpu" in device : 
            t = cupy.array( t )
        pass
    
        x = t 
        
        t = np.random.rand( s*s ) + 1j*np.random.rand( s*s )
        t = t.reshape( s, s )
        if "gpu" in device : 
            t = cupy.array( t )
        pass
    
        y = t 

        z = x*y 

        arrays = [ x, y, z ]

        bytes = 0 

        for array in arrays :
            bytes += array.nbytes
            
            del array
        pass

        elapsed = perf_counter() - then

        memories.append( bytes )
        durations.append( elapsed )
        grid_counts.append( s )

        print( f"Memory = {bytes/1e9:.4f} Gb", f" Elapsed = {elapsed:.2f} seconds" )
        print() 
        
    pass # device


    from matplotlib import pyplot as plt
    row_cnt = 1; col_cnt = 1
    fig, charts = plt.subplots( row_cnt, col_cnt, figsize=( 8*col_cnt, 6*row_cnt) )
    charts = charts.flatten() if row_cnt*col_cnt > 1 else [charts]
    chart_idx = 0 
    chart = charts[ chart_idx ]

    n = len( all_devices )
    x = numpy.arange( n )

    memories = numpy.array( memories )/1e6
    durations = numpy.array( durations )*1e3

    w = 0.2 

    idx = -1
    chart.bar ( x + w*idx, memories, label="Memory (Mb)", width=w ); idx+=1
    chart.bar ( x + w*idx, grid_counts, label="Grid Count", width=w ); idx+=1
    chart.bar ( x + w*idx, durations, label="Running Time(mili sec.)", width=w );
    chart.plot( x + w*idx, durations, marker="D" )

    title = f"\nArray Multiplication Performance\n" 
    
    chart.set_title( title )
    chart.set_xlabel( "Device\n" )
    chart.set_xticks( x )
    chart.set_xticklabels( all_devices ) 
    chart.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=n)

    plt.tight_layout()
    src_dir = os.path.dirname( os.path.abspath(__file__) )
    result_figure_file = f"{src_dir}/result/array_multiplication_performance.png.png"
    print( f"Result figure file = {result_figure_file}" )
    plt.savefig( result_figure_file )
    plt.show()

pass # test_memory_multiply_performance

if __name__ == "__main__":
    devices = [ "MULTI-GPU", "Multi-CPU", "GPU", "CPU" ]
    test_memory_multiplication_performance( devices )
pass
