# -*- coding: utf-8 -*-

from Zernike import *

import matplotlib.pyplot as plt
    
if __name__ == '__main__':    
    log.info( "Hello ...\n" )
    
    zernike = Zernike()
    
    n_max = 4
    m_max = 0
    
    ncols = 4
    nrows = n_max + 1
    nrows = nrows*(nrows+1)/2/ncols + 1
    nrows = int( nrows )
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    axes = axes.ravel()
    
    ax_idx = -1
    
    e = 0.001
    
    log.info( f"e = {e}")
    
    debug = 0 
    
    for n in range( 0, n_max + 1 ) :
        for m in range( 0, n + 1 ):
            x = np.arange(0, 1, e) 
            
            y_analytic = zernike.np_analytic_polynomial(x, n, m)            
            y_numeric = zernike.np_numeric_polynomial(x, n, m)
            
            if debug :
                log.info( f"x = {x}")
                log.info( f"y = {y_analytic}")
            pass
            
            ax_idx += 1     
            ax = axes[ ax_idx ]
            #ax.set_xlim([0, 1]) 
            
            diff = y_analytic - y_numeric
            
            ax.plot(x, y_analytic,  '-', color='C2', label='Analytic', linewidth=1 )
            ax.plot(x, diff , '-', color='C1', label='Diff' , linewidth=1 )
            
            ax.legend()            
            ax.set_xlabel( f"n = {n}, m = {m}" )
        pass
    pass
    
    print_profile()
    
    plt.tight_layout()
    plt.get_current_fig_manager().canvas.set_window_title('Pseudo-Zernike Polynomial Validation')
    plt.show()
    
    log.info( "Good bye!" )
pass