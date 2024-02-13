import math, psutil
from time import *
from time import perf_counter
from matplotlib import pyplot as plt

import torch

print( "Hello...\n" )

def get_free_mem( use_gpu, device = 0, verbose = 0 ) :
    if use_gpu : 
        import torch
        free_mem, total_mem = torch.cuda.mem_get_info( device )
        used_mem = total_mem - free_mem

        verbose and print( f"GPU mem : total = {total_mem:_}, free = {free_mem:_}, used = {used_mem:_} " )

        return free_mem
    else :
        import psutil
        ps_mem = psutil.virtual_memory() ; 
        total_mem, free_mem = ps_mem[0], ps_mem[1]
        used_me= total_mem - free_mem

        verbose and print( "PSU = ", psutil.virtual_memory())
        verbose and print( f"PSU mem : total = {total_mem:_}, free = {free_mem:_}, used = {used_mem:_} " )
        return free_mem
    pass
pass

def test_array_memory( use_gpu , operation="", debug=0, verbose=0) : 
    
    max_grid_count = 90_000
    #max_grid_count = int( math.sqrt( psutil.virtual_memory().available/3 ) )

    if len( operation ) < 1 : 
        max_grid_count = int( math.sqrt( psutil.virtual_memory().available ) )
    pass

    if debug : 
        print( f"max_grid_count = {max_grid_count:_}" )
        print()
        print( f"use_gpu = {use_gpu}, operation = {operation}" )
        print( flush=1 )
    pass

    device_name = "cuda" if use_gpu else "cpu"

    device = torch.device( device_name) 

    #data_types = [ np.int_, np.double, np.csingle, np.cdouble ]
    data_types = [ torch.int, torch.double, torch.cfloat, torch.cdouble ]
    
    dx = 0

    types = [ ]
    memories = [ ]
    grid_counts = [ ]
    elapsed_times = []
    
    for idx, data_type in enumerate( data_types ):
        array = torch.zeros( [1], dtype=data_type, device=device )
        data_type_size = array.nbytes
        
        type_str = f"{data_type}".split( " " )[-1].split(".")[-1].split( "'")[0]
        
        types.append( type_str )
        
        type_str = device_name + " " + type_str + f"({data_type_size} bytes)"  
        
        debug and print( type_str, flush=1 )
        
        
        grid_count_succ = 2**8
        grid_count = 2**9
        grid_count_max = None

        print( f"grid_count_max = {grid_count_max}" )
        
        memory_size = 0 
        elapsed = 0 
        error = None
        
        while grid_count_max is None or abs( grid_count_max - grid_count_succ ) > 1 :
            
            arrays = [ ]
            try : 
                if grid_count_max is None : 
                    grid_count = min( grid_count_succ*2, max_grid_count )
                else :
                    grid_count = (grid_count_max + grid_count_succ)//2
                pass
            
                verbose and print( f"grid count = {grid_count:_}, " , end="", flush=1 )
            
                then = perf_counter()

                if len( operation ) < 1 : 
                    a = torch.zeros( (grid_count, grid_count), dtype=data_type, device=device )
                    
                    memory_size = a.nbytes
                    arrays.append( a )
                else :
                    a = torch.zeros( (grid_count, grid_count), dtype=data_type, device=device )
                    arrays.append( a )
                    
                    memory_size = a.nbytes
                    
                    b = torch.zeros( (grid_count, grid_count), dtype=data_type, device=device )
                    arrays.append( b )
                    
                    c= a*b
                    arrays.append( c )
                pass
            
                elapsed = perf_counter() - then
                
                grid_count_succ = grid_count 
                
                if verbose : print( f"Elapsed = {elapsed}, grid_count = {grid_count:_}" )

            except Exception as e:
                error = e
                grid_count_max = min( grid_count, max_grid_count )
            finally :
                for array in arrays :
                    import gc

                    array.cpu()
                    del array
                    
                    gc.collect()
                    
                    use_gpu and torch.cuda.empty_cache()
                pass

                if use_gpu :
                    torch.cuda.empty_cache()
                pass

                if grid_count_succ > max_grid_count*0.9 :
                    break
                pass
            pass
        pass
    
        if verbose : print()
            
        if debug:   
            if error  : 
                print( error, flush=1 )
            pass

            print( f"grid_count = {grid_count:_}, size = {memory_size/1e9:_.1f} Gb, run-time = {elapsed:.2f} (sec.)", flush=1 )
    
        grid_counts.append( grid_count ) 
        memories.append( memory_size )
        elapsed_times.append( elapsed )
        
        debug and print( ) 
    pass

    x = types 
    
    y1 = grid_counts / 1e3
    y2 = memories / 1e9    
    y3 = elapsed_times
    
    #y1 = numpy.array( grid_counts )/1e3
    #y2 = numpy.array( memories )/1e9    
    #y3 = numpy.array( elapsed_times )
    
    ymax = ( max( torch.max( y1 ), torch.max( y2 ), torch.max( y3 ) ) )
    
    ymax = int( ymax +  1.5 + 10**int( math.log10(ymax/10) ) )
        
    row_cnt = 1; col_cnt = 1
    fig, charts = plt.subplots( row_cnt, col_cnt, figsize=( 8*col_cnt, 6*row_cnt) )
    charts = charts.flatten() if row_cnt*col_cnt > 1 else [charts]
    chart_idx = 0
    chart = charts[ chart_idx ]
    chart_idx +=1 
    
    chart.plot( x, y1, marker="s", label="Axial Grid count(K)" ) 
    chart.plot( x, y2, marker="D", label="Memory(Gb)" ) 
    chart.plot( x, y3, marker="*", label="Rum-time (sec.)" ) 

    device = [ "CPU", "GPU" ][ use_gpu ]
    xlabel = [ "NumPy", "Cupy" ][ use_gpu ]
    
    op_title = "for Multiplication" if operation else ""
    chart.set_xticks( x )
    chart.set_title( f"\n{device} Array Memory Allocation Maximum Size {op_title}\n" )
    chart.set_xlabel( f"\n{xlabel} Data Type" ) 
    chart.set_ylim( 0, ymax )
    # chart.legend(loc="lower center", bbox_to_anchor=(0.5, -0.26), ncol=3 ) 
    chart.legend()
    
    plt.tight_layout()
    plt.savefig( f"./result/memory_allocation_{use_gpu}_{len(operation)}.png" )
    plt.show(); 

    print( "Done." )
pass # -- test_array_memory

if __name__ == "__main__":
   test_array_memory( use_gpu=0, operation="x", debug=1, verbose=0 )
pass

