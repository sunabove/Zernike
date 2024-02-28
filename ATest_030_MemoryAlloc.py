import os, math, psutil, time
import torch

from matplotlib import pyplot as plt
from matplotlib import colors as mcolors

from ACommon import *

print( "Hello..." )

def test_array_memory_alloc( use_gpus , operation="", debug=0 ) :
    
    if len( operation ) < 1 : 
        max_tick_count = int( math.sqrt( psutil.virtual_memory().available ) )
    pass

    if debug : 
        print()
        print( f"operation = {operation}", flush=1 )
        print()
    pass

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
    
    miny = None
    maxy = None 

    for use_gpu in use_gpus :

        device_name = "cuda" if use_gpu else "cpu"
        device = torch.device( device_name)

        print( f"device name = {device_name}", flush=1 )

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

            free_mem_bytes, total_mem_bytes, free_ratio = get_free_mem_bytes( use_gpu, device_no=0, verbose=0 ) 
            
            free_mem_bytes_prev = free_mem_bytes
            print( f"free_mem_bytes = {free_mem_bytes:_} bytes {free_ratio:.0%}%", flush=1 )

            tick_count = math.sqrt( free_mem_bytes/data_type_size*0.95 )
            tick_count = int( tick_count )

            if operation : 
                tick_count = math.sqrt( free_mem_bytes/data_type_size*0.95/3 )
                tick_count = int( tick_count )
            pass
            
            memory_size = 0 
            elapsed = 0
            error = None
            
            arrays = [ ]
            try : 
                then = time.time()

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
            
                elapsed = time.time() - then

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

                free_mem_bytes, total_mem_bytes,free_ratio = get_free_mem_bytes( use_gpu, device_no=0, verbose=0 )
                print( f"free_mem_bytes after gc= {free_mem_bytes:_} bytes {free_ratio:.0%}%" , flush=1)
            pass

            if debug:   
                error and print( error, flush=1 )

                print( f"tick_count = {tick_count:_}, size = {memory_size/1e9:_.1f} Gb, run-time = {elapsed:.6f} (sec.)", flush=1 )
            pass
            
            if True : 
                types.append( type_str )
                tick_counts.append( tick_count ) 
                memories.append( memory_size )
                elapsed_times.append( elapsed )
            pass

            if True : 
                duration = 1
                print( f"sleep( {duration} )")        
                time.sleep( duration )
            pass
            
            debug and print( ) 
        pass

        x = types 
        
        y1 = torch.tensor( tick_counts ) / 1_000
        y2 = torch.tensor( memories ) / 1e9
        y3 = torch.log10( torch.tensor( elapsed_times ) )

        if miny is None :
            miny = min( torch.min( y1 ), torch.min( y2 ), torch.min( y3 ) )
            maxy = max( torch.max( y1 ), torch.max( y2 ), torch.max( y3 ) )
        else :
            miny = min( miny, min( torch.min( y1 ), torch.min( y2 ), torch.min( y3 ) ) )
            maxy = max( maxy, max( torch.max( y1 ), torch.max( y2 ), torch.max( y3 ) ) )
        pass
        
        
        ls = linestyle = "solid" if use_gpu else "dashed"
        dn = "GPU" if use_gpu else "CPU"
        
        chart.plot( x, y1, color=colors[0%len(colors)], marker=markers[0%len(markers)], linestyle=ls, label=f"{dn}: Tick count(K)" ) 
        chart.plot( x, y2, color=colors[1%len(colors)], marker=markers[1%len(markers)], linestyle=ls, label=f"{dn}: Memory(Gb)" ) 
        chart.plot( x, y3, color=colors[2%len(colors)], marker=markers[2%len(markers)], linestyle=ls, label=f"{dn}: $log_{'{10}'}$(Rum-time sec.)" ) 
        
        chart.set_xticks( x )
        chart.set_xlabel( f"Data Type" ) 
    pass
    
    op_title = "for Multiplication" if operation else ""
    title = f"Array Memory Max. Size {op_title}"

    if len( use_gpus ) == 1 :
        use_gpu = use_gpus[ 0 ]
        dn = "GPU" if use_gpu else "CPU"
        title = f"{dn} {title}"
    pass

    chart.set_title( title )

    if 1 : 
        log10 = int( math.log10( abs( maxy - miny ) ) )
        yunit = pow( 10, log10 - 1 )

        if yunit < 5 :
            yunit = 5

        maxy = yunit*( int( maxy/yunit + 1 ) + 0.2 )

        if miny < 0 :
            miny = - yunit*( int( abs(miny)/yunit + 1 ) + 0.2 )
        else :
            miny = min( 0, miny ) 
        pass
        

        print( f"miny = {miny}, maxy = {maxy}" )
        chart.set_ylim( miny, maxy )
    pass

    chart.grid( axis='x', linestyle="dotted" )
    chart.grid( axis='y', linestyle="dotted" )
    
    chart.legend(loc="upper center", ncol=3, fontsize=fs-4 )
    chart.legend()
    
    src_dir = os.path.dirname( os.path.abspath(__file__) )
    result_figure_file = f"{src_dir}/result/test_030_memory_allocation_{len(operation)}.png"
    print( f"Result figure file = {result_figure_file}" )
    plt.savefig( result_figure_file )
    plt.show(); 

    print( "\nDone." )
pass # -- test_array_memory

if __name__ == "__main__":
    test_array_memory_alloc( use_gpu=1, operation="", debug=1, verbose=1 )
pass

