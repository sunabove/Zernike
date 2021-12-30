import numpy, cupy
import numpy as np, cupy as cp, math, psutil, dask.array as da
#from dask_cuda import LocalCUDACluster
#from dask.distributed import Client
from time import *
from time import perf_counter

def test_memory_multiply_performance(devices) :     

    memories = []
    durations = []
    grid_counts = []

    for device in devices : 
        device = device.lower()
        rs = None

        scheduler = "single-threaded" 

        if device.__contains__( "multi" ) :
            scheduler = "threads"
        pass 

        s = 9_350
        
        free_memory = psutil.virtual_memory().available
        
        if device.__contains__( "cpu" ):
            #print( f"device is {device}")
            pass
        elif device.__contains__( "gpu" ) :
            #print( f"device is {device}")
            import GPUtil
    
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                free_memory = gpu.memoryFree*1e6*3/4
                break
            pass
        pass
    
        s = int( math.sqrt( free_memory/8 ) )//2

        then = perf_counter()

        print( f"device = {device}, grid_count = {s:_}, scheduler = {scheduler}" )

        t = np.random.rand( s*s ) + 1j*np.random.rand( s*s )
        t = t.reshape( s, s )
        if "gpu" in device : 
            t = cupy.array( t )
        pass
        x = da.from_array( t, chunks=(2**14, 2*14) )
        del t
        
        t = np.random.rand( s*s ) + 1j*np.random.rand( s*s )
        t = t.reshape( s, s )
        if "gpu" in device : 
            t = cupy.array( t )
        pass
        y = da.from_array( t, chunks=(2**14, 2*14) )    
        del t

        z = x*y
        z = z.compute(scheduler=scheduler)

        arrays = [ x, y, z ]

        bytes = 0 

        for array in arrays :
            bytes += array.nbytes
        pass

        elapsed = perf_counter() - then

        memories.append( bytes )
        durations.append( elapsed )
        grid_counts.append( s )

        print( f"Memory = {bytes/1e9:.4f} Gb", f" Elapsed = {elapsed:.2f} seconds" )
        print() 

        for array in arrays : 
            del array
        pass
    pass

    return memories, durations, grid_counts

pass # test_memory_multiply_performance