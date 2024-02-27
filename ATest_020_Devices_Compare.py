from AZernike import *
from tqdm import tqdm 
import time

def test_process_unit_spec() :
    import psutil, igpu

    devices = [ 'CPU' ]

    frequencies = [ psutil.cpu_freq().max/1_000 ] 
    memories = [ round(psutil.virtual_memory().total/1e9/10, 2) ]

    for gpu in igpu.devices() : 
        devices.append( f"GPU {gpu.index}" )
        frequencies.append( gpu.clocks.max_graphics/1_000 )
        memories.append( gpu.memory.total/1_000/10 )
    pass

    fs = fontsize =16
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = fontsize

    row_cnt = 1; col_cnt = 1
    fig, charts = plt.subplots( row_cnt, col_cnt, figsize=( 8*col_cnt, 6*row_cnt) )
    charts = charts.flatten() if row_cnt*col_cnt > 1 else [charts]
    chart_idx = 0 
    chart = charts[ chart_idx ]

    w = 0.4
    n = 2
    x = torch.arange(0, len(devices), dtype=torch.float64 )
    x += w*((1.0 - n )/2.0)

    yticks = torch.arange( 0, math.ceil( max( max(frequencies), max(memories)) ), 2 )

    bar = chart.bar( x - w/2 + w*0, memories, width=w, label="Memory (10 Gb)" )
    chart.bar_label( bar, fmt='%.1f', fontsize=fs-2 )
    bar = chart.bar( x - w/2 + w*1, frequencies, width=w, label="Frq. (GHZ)" )
    chart.bar_label( bar, fmt='%.1f', fontsize=fs-2 )

    chart.set_xticks( x, devices )
    chart.set_yticks( yticks )
    chart.grid( axis='y', linestyle="dotted" )
    chart.set_title( "Process Unit Specification" )
    chart.set_xlabel( "Process Unit Name")

    chart.legend()

    plt.tight_layout()
    plt.show()
pass ## test_process_unit_spec()

def test_gpu_specs_compare() :
    from matplotlib import pyplot as plt
    import pandas as pd

    print( pd.__version__ )

    df = excelData = pd.read_excel( r"./data/NvidiaGPU_Compare.xlsx" ) 

    fs = fontsize = 16
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = fontsize
    markers = [ "o", "s", "p", "*", "D", "^", "X", "2", "p", "h", "+" ]
    row_cnt = 1; col_cnt = 1
    fig, charts = plt.subplots( row_cnt, col_cnt, figsize=( 8*col_cnt, 4.5*row_cnt) )
    charts = charts.flatten() if row_cnt*col_cnt > 1 else [ charts ]

    chart = charts[0]

    chart.set_title( "GeForce RTX Spec." ) 

    cols = df.columns 
    x_axis = df[ cols[0] ]
    x_axis = x_axis.str.replace( " ", "\n" )

    for i in range( 1, len( cols ) ) : 
        chart.plot( x_axis, df[ cols[i] ], marker=markers[i], label=cols[i] ) 
        chart.xaxis.set_tick_params(labelsize=fs-3)
    pass

    chart.legend(fontsize=fs-4, ncols=3) 
    chart.set_ylim( 0, 2500 )
    chart.grid( axis='y', linestyle="dotted" )
    chart.set_xlabel( "GPU Name")

    plt.show()

pass ## test_gpu_specs_compare