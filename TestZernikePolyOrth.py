# -*- coding: utf-8 -*-

from Zernike import *

def test_zernike_polynomial_orth(div_count = 10_000) :
    log.info( "Hello ..." ) 
    
    zernike = Zernike() 
    
    dr = 1.0/div_count
    rhos = np.arange(0, 1 + dr, dr)
    
    order = 10 
    for p in range( order + 1 ) :
        for q in range( order + 1 ) :
            for l in range( min(p, q) + 1 ):
                
                log.info( f"p={p}, q={q}, l={l}" )
                
                sum = 0
                
                ss = np.zeros( [ rhos.shape[0] ], dtype=np.float )
                
                idx = 0 
                for rho in rhos :
                    zpl = zernike.polynomial(p, l, rho)
                    zql = zernike.polynomial(q, l, rho)
                    ds = zpl*zql*rho*dr
                    ss[ idx ] = ds
                    idx +=1 
                pass
            
                sum = np.sum( ss )
                sum = 2*(p + 1)*sum
                
                result = 'Bad'
                
                if p == q and sum > 0.9 :
                    result = 'Good'
                elif sum < 0.01 :
                    result = 'Good'
                pass
            
                print( f"Rsum({p}, {q}) = {sum:.10f}, result = {result}" )
                
                if result == 'Bad' :
                    import sys
                    sys.exit( 0 )
                pass
            pass
        pass
    pass     
    
    print_profile()

    log.info( "Good bye!" )
pass

if __name__ == '__main__':
    test_zernike_polynomial_orth()
pass