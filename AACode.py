# 메모리 현황 구하기
import os

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

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = "16"

chart.grid( axis='y', linestyle="dotted" )

use_gpu = 1
device_no = 0 
hash = {} 
device = torch.device( f"cuda:{device_no}" ) if use_gpu else torch.device( f"cpu" )