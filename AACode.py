# 메모리 현황 구하기
import os
import torch

print( "source dirname = ", os.path.dirname( os.path.abspath(__file__) ) )

total_mem, used_mem, free_mem = map( int, 
    os.popen('free -b').readlines()[-1].split()[1:] )

print( f"CPU mem : total = {total_mem:_}, free = {free_mem:_}, used = {used_mem:_} " )

import psutil
ps_mem = psutil.virtual_memory() ; 
total_mem, free_mem = ps_mem[0], ps_mem[1]
used_me= total_mem - free_mem
print( "PSU = ", psutil.virtual_memory())
print( f"PSU mem : total = {total_mem:_}, free = {free_mem:_}, used = {used_mem:_} " )

# CUDA memory

import torch
free_mem, total_mem = torch.cuda.mem_get_info( 0 )
used_mem = total_mem - free_mem

print( f"GPU mem : total = {total_mem:_}, free = {free_mem:_}, used = {used_mem:_} " )

from matplotlib import pyplot as plt

fs = fontsize = 16
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = fontsize

row_cnt = 1
col_cnt = 1
fig, charts = plt.subplots( row_cnt, col_cnt, figsize=(9*col_cnt, 8*row_cnt), tight_layout=1 )
charts = charts.ravel() if row_cnt*col_cnt > 1 else [charts]
chart_idx = 0 
chart = charts[ chart_idx ] ; chart_idx += 1

chart.grid( axis='y', linestyle="dotted" )
chart.legend( loc="lower center", bbox_to_anchor=(0.5, -0.26), fontsize=fs-4, ncols=3 )

src_dir = os.path.dirname( os.path.abspath(__file__) )
result_figure_file = f"{src_dir}/result/zernike_02_radial_orthogonality.png"
plt.savefig( result_figure_file )
print( f"result_figure_file = {result_figure_file}" )

use_gpu = 1
device_no = 0 
hash = {} if use_gpu else None
device = torch.device( f"cuda:{device_no}" ) if use_gpu else torch.device( f"cpu" )

kidx = torch.where( rho_square <= 1.0 )

Ks = torch.arange( 1, 6 + 1, 1 )

# 테이블 생성
tab_header = [ "Device", "Item" ]
tab_header.extend( [ f"{x/1_000:1.0f}K" for x in resolutions ] )

print( tabulate( tab_rows, headers=tab_header ) )

excelData = []
excelData.append( tab_header )
excelData.extend( tab_rows )
df = pd.DataFrame( excelData )
df.to_excel( f"{src_dir}/result/zernike_02_radial_orthogonality.xlsx", index=False, header=False, sheet_name='poly orth')

# 로그 피팅

torch.fit

