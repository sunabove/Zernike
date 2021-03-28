# -*- coding: utf-8 -*-

from Zernike import *

if __name__ == '__main__':
    log.info( "Hello ...\n" ) 
    
    zernike = Zernike() 
    
    dr = 0.0001 
    for n in range( 10 ) :
        for m in range( n + 1 ) :
            sum = 0
            
            for rho in np.arange(0, 1 + dr, dr) :
                p = zernike.polynomial(n, m, rho)
                q = zernike.polynomial(m, n, rho)
                ds = p*q*rho*dr
                sum += ds
            pass
        
            sum = 2*(n + 1)*sum
        
            log.info( f"Rsum({n}, {m}) = {sum:.10f}" )
        pass
    pass     
    
    print_profile()

    log.info( "Good bye!" )
pass