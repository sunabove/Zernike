# -*- coding: utf-8 -*-

from Zernike import *

if __name__ == '__main__':
    log.info( "Hello ...\n" ) 
    
    zernike = Zernike() 
    
    max_n = 4
    max_m = max_n
    
    for n in range( max_n ) :
        for m in range( max_m ) :
            sum = 0

            dx = 0.01
            dy= dx
            
            for x in np.arange(0, 1 + dx, dx ) :
                for y in np.arange(0, 1 + dy, dy ) :
                    p = zernike.zernike_function(n, m, x, y).conjugate()
                    q = zernike.zernike_function(n, m, x, y)
                    ds = p*q*dx*dy
                    sum += ds
                pass
            pass
        
            sum = sum*(n + 1)/pi
            sum = abs( sum ) 
        
            log.info( f"sum({n}, {m}) = {sum:.10f}" )
        pass
    pass     
    
    print_profile()

    log.info( "\nGood bye!" )
pass