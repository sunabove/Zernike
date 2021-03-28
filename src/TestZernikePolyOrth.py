# -*- coding: utf-8 -*-

from Zernike import *

def test_zernike_polynomial_orth() :
    log.info( "Hello ..." ) 
    
    zernike = Zernike() 
    
    dr = 0.0001
    rhos = np.arange(0, 1 + dr, dr)
             
    for n in range( 10 ) :
        for m in range( n + 1 ) :
            sum = 0
            
            ss = np.zeros( [ rhos.shape[0] ], dtype=np.float )
            
            idx = 0 
            for rho in rhos :
                p = zernike.polynomial(n, m, rho)
                q = zernike.polynomial(m, n, rho)
                ds = p*q*rho*dr
                ss[ idx ] = ds
                idx +=1 
            pass
        
            sum = np.sum( ss )
            sum = 2*(n + 1)*sum
        
            print( f"Rsum({n}, {m}) = {sum:.10f}" )
        pass
    pass     
    
    print_profile()

    log.info( "Good bye!" )
pass

if __name__ == '__main__':
    test_zernike_polynomial_orth()
pass