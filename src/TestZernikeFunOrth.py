# -*- coding: utf-8 -*-

from Zernike import *

if __name__ == '__main__':
    log.info( "Hello ..." ) 
    
    zernike = Zernike() 
    
    max_n = 5
    
    for n in range( max_n + 1 ) :
        for m in range( 0, n + 1 ) :
            sum = 0

            dx = 0.01
            dy= dx
            
            for x in np.arange( -1, 1, dx ) :
                for y in np.arange( -1, 1, dy ) :
                    if x*x + y*y <= 1 : 
                        p = zernike.function(n, m, x, y)
                        q = zernike.function(n, m, x, y)
                        ds = p.conjugate()*q*dx*dy
                        sum += ds
                    pass
                pass
            pass
        
            sum = sum*(n + 1)/pi
            sum = abs( sum ) 
        
            log.info( f"Vsum({n}, {m}, {n}, {m}) = {sum:.10f}" )
        pass
    pass     
    
    print_profile()

    log.info( "Good bye!" )
pass