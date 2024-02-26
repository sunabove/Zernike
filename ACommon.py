# common file

def get_free_mem_bytes( use_gpu, device_no = 0, verbose = 0 ) :
    if use_gpu : 
        import torch
        free_mem, total_mem = torch.cuda.mem_get_info( device_no )
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