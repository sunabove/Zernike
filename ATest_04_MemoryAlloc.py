import os, math, psutil
from time import *
from time import perf_counter
from matplotlib import pyplot as plt

import torch

print( "Hello..." )

def get_free_mem_bytes( use_gpu, device = 0, verbose = 0 ) :
    if use_gpu : 
        import torch
        free_mem, total_mem = torch.cuda.mem_get_info( device )
        used_mem = total_mem - free_mem

        verbose and print( f"GPU mem : total = {total_mem:_}, free = {free_mem:_}, used = {used_mem:_} " )

        return free_mem, total_mem, free_mem/total_mem
    else :
        import psutil
        ps_mem = psutil.virtual_memory() ; 
        total_mem, free_mem = ps_mem[0], ps_mem[1]
        used_mem = total_mem - free_mem

        verbose and print( "PSU = ", psutil.virtual_memory())
        verbose and print( f"PSU mem : total = {total_mem:_}, free = {free_mem:_}, used = {used_mem:_} " )
        return free_mem, total_mem, free_mem/total_mem
    pass
pass

def test_array_memory_alloc( use_gpu , operation="", debug=0, verbose=0) :
    
    if len( operation ) < 1 : 
        max_tick_count = int( math.sqrt( psutil.virtual_memory().available ) )
    pass

    if debug : 
        print()
        print( f"use_gpu = {use_gpu}, operation = {operation}", flush=1 )
        print()
    pass

    device_name = "cuda" if use_gpu else "cpu"

    device = torch.device( device_name) 

    data_types = [ torch.int, torch.double, torch.cfloat, torch.cdouble ]

    types = [ ]
    memories = [ ]
    tick_counts = [ ]
    elapsed_times = []
    
    for idx, data_type in enumerate( data_types ):
        print( f"idx = [{idx}]", flush=1 )

        array = torch.zeros( [1], dtype=data_type, device=device )
        data_type_size = array.nbytes
        
        type_str = f"{data_type}".split( " " )[-1].split(".")[-1].split( "'")[0]
        
        type_detail_str = device_name + " " + type_str + f"({data_type_size} bytes)"  
        
        debug and print( type_detail_str, flush=1 )

        if use_gpu : torch.cuda.empty_cache()

        free_mem_bytes, total_mem_bytes, free_ratio = get_free_mem_bytes( use_gpu, device=0, verbose=0 ) 

        free_mem_bytes_prev = free_mem_bytes
        print( f"free_mem_bytes = {free_mem_bytes:_} bytes {free_ratio*100:.0f}%", flush=1 )

        tick_count = math.sqrt( free_mem_bytes/data_type_size*0.95 )
        tick_count = int( tick_count )

        if operation : 
            tick_count = math.sqrt( free_mem_bytes/3/data_type_size*0.95 )
            tick_count = int( tick_count )
        pass
        
        memory_size = 0 
        elapsed = 0
        error = None
        
        arrays = [ ]
        try : 
            verbose and print( f"grid count = {tick_count:_}, " , end="", flush=1 )
        
            then = perf_counter()

            if len( operation ) < 1 : 
                a = torch.zeros( (tick_count, tick_count), dtype=data_type, device=device )
                
                memory_size = a.nbytes
                arrays.append( a )

                a = None
            else :
                a = torch.zeros( (tick_count, tick_count), dtype=data_type, device=device )
                arrays.append( a )
                
                memory_size = a.nbytes
                
                b = torch.zeros( (tick_count, tick_count), dtype=data_type, device=device )
                arrays.append( b )
                
                c= a*b
                arrays.append( c )

                a = b = c = None
            pass
        
            elapsed = perf_counter() - then 
            
            if verbose : print( f"Elapsed = {elapsed}, tick_count = {tick_count:_}", flush=1 )

        except Exception as e:
            error = e 
        finally :
            for array in arrays :
                del array
            pass

            del arrays

            arrays = None

            import gc
            gc.collect()

            if use_gpu : torch.cuda.empty_cache()

            free_mem_bytes, total_mem_bytes,free_ratio = get_free_mem_bytes( use_gpu, device=0, verbose=0 )
            print( f"free_mem_bytes after gc= {free_mem_bytes:_} bytes {free_ratio*100:.0f}%" , flush=1)
        pass

        if debug:   
            error and print( error, flush=1 )

            print( f"tick_count = {tick_count:_}, size = {memory_size/1e9:_.1f} Gb, run-time = {elapsed:.2f} (sec.)", flush=1 )
        pass
        
        if True : 
            types.append( type_str )
            tick_counts.append( tick_count ) 
            memories.append( memory_size )
            elapsed_times.append( elapsed )
        pass

        if True : 
            duration = 5
            print( f"sleep( {duration} )")        
            sleep( duration )
        pass
        
        debug and print( ) 
    pass

    x = types 
    
    y1 = torch.tensor( tick_counts ) / 1_000
    y2 = torch.tensor( memories ) / 1e9
    y3 = torch.tensor( elapsed_times ) * 10
    
    #y1 = numpy.array( tick_counts )/1e3
    #y2 = numpy.array( memories )/1e9    
    #y3 = numpy.array( elapsed_times )
    
    ymax = max( torch.max( y1 ), torch.max( y2 ), torch.max( y3 ) )
    
    if True : 
        log10 = int( math.log10( ymax ) )
        yunit = pow( 10, log10 )

        ymax = yunit*( int( ymax/yunit + 1 ) + 0.2)
    pass
        
    row_cnt = 1; col_cnt = 1
    
    fig, charts = plt.subplots( row_cnt, col_cnt, figsize=( 8*col_cnt, 6*row_cnt) )
    
    charts = charts.flatten() if row_cnt*col_cnt > 1 else [charts]
    chart_idx = 0
    chart = charts[ chart_idx ]
    chart_idx +=1 
    
    chart.plot( x, y2, marker="D", label="Memory(Gb)" ) 
    chart.plot( x, y1, marker="s", label="Tick count(K)" ) 
    chart.plot( x, y3, marker="*", label="Rum-time(centi sec.)" ) 

    xlabel = device_name.upper()
    
    op_title = "for Multiplication" if operation else ""

    chart.set_xticks( x )
    chart.set_title( f"{device_name.upper()} Array Memory Max. Size {op_title}" )
    chart.set_xlabel( f"{xlabel} Data Type" ) 
    chart.set_ylim( 0, ymax )
    chart.grid( axis='y', linestyle="dotted" )
    #chart.legend()
    chart.legend(loc="upper center", ncol=3, fontsize=13 )
    
    plt.tight_layout()
    src_dir = os.path.dirname( os.path.abspath(__file__) )
    result_figure_file = f"{src_dir}/result/memory_allocation_{use_gpu}_{len(operation)}.png"
    print( f"Result figure file = {result_figure_file}" )
    plt.savefig( result_figure_file )
    plt.show(); 

    print( "\nDone." )
pass # -- test_array_memory

if __name__ == "__main__":
    for use_gpu in [ 0, 1 ] : 
        for operation in [ "", "*" ] : 
            test_array_memory_alloc( use_gpu=use_gpu, operation=operation, debug=1, verbose=0 )
        pass
    pass
pass

