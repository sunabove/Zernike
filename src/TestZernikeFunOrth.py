# -*- coding: utf-8 -*-

from Zernike import *

def test_zenike_function_orth() :
    log.info( "Hello ..." ) 
    
    zernike = Zernike() 
    
    max_n = 5
    
    div_count = 400
    dx = 2.0/div_count
    dy= dx
    
    xs = np.arange( -1, 1, dx )
    ys = np.arange( -1, 1, dy )
    
    for n in range( max_n + 1 ) :
        for m in range( 0, n + 1 ) :            
            for n2 in range( max_n + 1 ) :
                for m2 in range( 0, n2 + 1 ) :            
                    ss = np.zeros( [div_count, div_count], dtype=np.complex )
        
                    for i, x in enumerate( xs ) :
                        for k, y in enumerate( ys ) :
                            if x*x + y*y <= 1 : 
                                p = zernike.function(n, m, x, y)
                                q = zernike.function(n2, m2, x, y)
                                ds = p.conjugate()*q*dx*dy
                                ss[ i, k ] = ds
                            pass
                        pass
                    pass
                
                    sum = np.sum( ss )
                    sum = sum*(n + 1)/pi
                    sum = abs( sum ) 
                
                    print( f"Vsum({n}, {m}, {n2}, {m2}) = {sum:.10f}" )
                                        
                pass
            pass
        pass
    pass     
    
    print_profile()

    log.info( "Good bye!" )
pass

if __name__ == '__main__':
    test_zenike_function_orth()
pass