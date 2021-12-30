import gc, math, psutil
import numpy, cupy
import numpy as np, cupy as cp, dask.array as da
#from dask_cuda import LocalCUDACluster
#from dask.distributed import Client
from time import *
from time import perf_counter

def test_memory_multiply_performance( devices ) :     

    memories = []
    durations = []
    grid_counts = []
    
    all_devices = []
    
    for use_dask in [0, 1 ] : 

        for device in devices : 
            device = device.lower()
        
            if use_dask : 
                all_devices.append( f"{device} DSK" )
            else : 
                all_devices.append( device )
            pass
        
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
        
            x = None 
            if use_dask : 
                x = da.from_array( t, chunks=(2**14, 2*14) )
                del t
            else :
                x = t
            pass
            
            t = np.random.rand( s*s ) + 1j*np.random.rand( s*s )
            t = t.reshape( s, s )
            if "gpu" in device : 
                t = cupy.array( t )
            pass
        
            y = None        
            if use_dask :
                y = da.from_array( t, chunks=(2**14, 2*14) )
                del t
            else :
                y = t
            pass

            z = x*y
            if use_dask : 
                z = z.compute(scheduler=scheduler)
            pass

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
    pass # use_dask

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
    plt.savefig( f"./result/array_multiplication_performance.png" )
    plt.show()

pass # test_memory_multiply_performance

