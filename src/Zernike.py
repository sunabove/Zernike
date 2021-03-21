# -*- coding: utf-8 -*-

import warnings
#from gevent.libev.corecext import NONE

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import logging as log
log.basicConfig( format='%(levelname)s %(filename)s:%(lineno)04d %(message)s', datefmt='%Y-%m-%d:%H:%M:%S', level=log.INFO )

from os.path import join
from glob import glob

import sqlite3

import numpy as np
import math, cmath
from math import factorial, perm, atan2, pi, sqrt, log10
from time import time, sleep

from Profiler import *

class Zernike :
    def __init__(self, **kwargs) :
        self.debug = False 
        
        self.conn = sqlite3.connect("c:/temp/zernike.db")
        #self.conn.set_trace_callback(print)
        self.cursor = self.conn.cursor()

        self.create_table( kwargs )
    pass

    def __del__(self) :
        self.conn.commit()

        log.info( "Commit" )
    pass

    def create_table(self, kwargs):
        cursor = self.cursor

        log.info( "Create tables ...\n " )

        dropTable = "dropTable" in kwargs and kwargs["dropTable"] 
        if dropTable :
            tables = [ "zernike_function", "zernike_polynomial", ]
            for table in tables :
                sql = f"DROP TABLE IF EXISTS {table}"
                cursor.execute( sql )
            pass
        pass

        sql = """
            CREATE TABLE IF NOT EXISTS zernike_polynomial
            ( n INTEGER, m INTEGER, rho DOUBLE
              , value DOUBLE
              , calc_time DOUBLE NOT NULL DEFAULT 0
              , PRIMARY KEY ( n, m, rho )
            )
            """
        cursor.execute( sql )

        sql = """
               CREATE TABLE IF NOT EXISTS zernike_function
               ( n INTEGER, m INTEGER,  rho DOUBLE, theta DOUBLE
                 , vx DOUBLE, vy DOUBLE
                 , calc_time DOUBLE NOT NULL DEFAULT 0
                 , PRIMARY KEY ( n, m, rho, theta )
               )
               """
        cursor.execute(sql)

    pass # -- create_table

    @profile
    def calc_polynomial(self, n, m, rho):
        # -------------------------------------------------------------------------
        #   n = the order of Zernike polynomial
        #   m = the repetition of Zernike moment
        #   r = radius
        # -------------------------------------------------------------------------
        R = 0

        m = abs( m )
            
        for k in range( 0, n - m + 1 ) :
            r = 1
            
            if n - k == 0 :
                r = (-1) ** (k % 4)
                r *= factorial(2*n + 1 - k)/factorial(k)/factorial(n + m + 1 - k)/factorial(n - m - k)
            elif n - k != 0 :
                if rho == 0 :
                    r = 0
                elif rho != 0 :
                    r = (-1) ** (k % 4)
                    r *= rho**(n - k)
                    r *= factorial(2*n + 1 - k)/factorial(k)/factorial(n + m + 1 - k)/factorial(n - m - k)
                pass
            pass

            R += r
        pass

        return R
    pass # -- polynomial

    @profile
    def np_numeric_polynomial(self, x, n, m ):
        y = np.array( [ self.select_polynomial(n, m, rho) for rho in x ] )
         
        return y
    pass

    @profile
    def np_analytic_polynomial(self, x, n, m ):
        y = np.array( [ self.analytic_polynomial(n, m, rho) for rho in x ] )
         
        return y
    pass

    @profile
    def analytic_polynomial(self, n, m, rho):
        m = abs( m )
        
        R = 0 
        r = rho
        
        if n == 0 :
            if m == 0 :
                R = 1
            pass
        elif n == 1 :
            if m == 0 :
                R = -2 + 3*r
            elif m == 1 :
                R = r
            pass            
        elif n == 2 :
            if m == 0 :
                R = 3 + 10*(r**2) - 12*r
            elif m == 1 :
                R = 5*(r**2) - 4*r
            elif m == 2 :
                R = r**2
            pass
        elif n == 3 :
            if m == 0 :
                R = -4 + 35*(r**3) - 60*(r**2) + 30*r
            elif m == 1 :
                R = 21*(r**3) - 30*(r**2) + 10*r
            elif m == 2 :
                R = 7*(r**3) -6*(r**2)
            elif m == 3 :
                R = r**3
            pass
        elif n == 4 :
            if m == 0 :
                R = 5 + 126*(r**4) - 280*(r**3) + 210*(r**2) - 60*r
            elif m == 1 :
                R = 84*(r**4) - 168*(r**3) + 105*(r**2) - 20*r
            elif m == 2 :
                R = 36*(r**4) - 56*(r**3) + 21*(r**2)
            elif m == 3 :
                R = 94*(r**4) - 8*(r**3)
            elif m == 4 :
                R = r**4
            pass
        elif n == 5 :
            if m == 0 :
                R = -6 + 462*(r**5) - 1260*(r**4) + 1260*(r**3) - 560*(r**2) + 105*r
            elif m == 1 :
                R = 330*(r**5) - 840*(r**4) + 756*(r**3) - 280*(r**2) + 35*r
            elif m == 2 :
                R = 165*(r**5) - 360*(r**4) + 252*(r**3) - 56*(r**2)
            elif m == 3 :
                R = 55*(r**5) - 90*(r**4) + 36*(r**3)
            elif m == 4 :
                R = 11*(r**5) - 10*(r**4)
            elif m == 5 :
                R = r**5
            pass
        pass
    
        return R
    pass

    @profile
    def select_polynomial(self, n, m, rho):
        
        R = 0
                 
        cursor = self.cursor

        rows = cursor.execute(
            "SELECT value FROM zernike_polynomial WHERE n = ? and m = ? and rho = ?",
            [n, m, rho],
        ).fetchall()

        cnt = len( rows )
        
        if cnt > 1 :
            log.info( "Invalid polynomial count." )
            import sys
            sys.exit( 1 )
        elif cnt == 1 :
            for row in rows:
                R = row[ 0 ]
            pass
        elif cnt < 1 :
            then = time()
            R = self.calc_polynomial(n, m, rho)
            now = time()

            calc_time = now - then

            sql = '''
                INSERT INTO zernike_polynomial( n, m, rho, value, calc_time )
                VALUES ( ?, ?, ?, ?, ? )
                '''
            cursor.execute(sql, [n, m, rho, R, calc_time]) 
        pass 

        return R
    pass # -- select

    @profile
    def zernike_function(self, n, m, x, y):
        debug = self.debug 
        
        rho = math.sqrt( x*x + y*y )

        v = 1.0
        
        theta = atan2(y, x)
        theta = (theta*m) % pi
        calc_time = -1        

        if rho == 0 : 
            v = 0 
        elif rho != 0 :
            cursor = self.cursor
            
            rows = cursor.execute(
                "SELECT vx, vy FROM zernike_function WHERE n = ? and m = ? and rho = ? and theta = ?",
                [n, m, rho, theta],
            ).fetchall()

            cnt = len( rows )
            
            if cnt > 1 : 
                log.info( "Invalid count for zernike function")
                
                import sys
                sys.exit(1)     
            elif cnt == 1 :
                vx = 0
                vy = 0 
                for row in rows : 
                    vx = row[0]
                    vy = row[1]
                pass
            
                v = vx + 1j*vy
            elif cnt < 1 :
                then = time()
                r = self.select_polynomial(n, m, rho)
    
                debug and log.info(f"R(n={n}, m={m}, rho={rho:.4f}, x={x:.4f}, y={y:.4f}) = {r}")
                
                e = cmath.exp( 1j*theta )
            
                v = r*e
                
                calc_time = time() - then
                
                sql = '''
                    INSERT INTO zernike_function( n, m, rho, theta, vx, vy, calc_time )
                    VALUES ( ?, ?, ?, ?, ?, ?, ? )
                    '''
                cursor.execute(sql, [n, m, rho, theta, v.real, v.imag, calc_time])
            pass
        pass
    
        debug and log.info(f"V(n={n}, m={m}, rho={rho:.4f}, theta={theta:.4f}, x={x:.4f}, y={y:.4f}) = {v}, calc_time={calc_time}")

        return v
    pass # -- zernike_function

    @profile
    def zernike_moment(self, img, n, m, k=1 ):
        '''
        Parameters
        ----------
        T : 
        k : numerical scheme        
        '''
        debug = self.debug 
        
        log.info( f"n={n}, m={m}, k={k}" )
        
        then = time()
        
        if k < 1 :
            k = 1
        pass
    
        ns = np.arange( 0, 1, 1/k)
        ns_count = len( ns )
        
        h = img.shape[0]
        w = img.shape[1]
        
        radius = max( h, w )/sqrt(2)
        debug and log.info( f"Radius = {radius}" )
        
        moments = np.zeros([h, w], dtype=np.complex) 
        
        ds = 1/radius/radius
        
        for y0, row in enumerate(img) :
            for x0, pixel in enumerate( row ) :
                a = 0
                for dy in ns :
                    y = y0 + dy
                         
                    for dx in ns :
                        x = x0 + dx
                        
                        # convert coordinate into unit circle coordinate system
                        ry = (y - h/2)/radius
                        rx = (x - w/2)/radius
                        
                        zf = self.zernike_function(n, m, rx, ry)
                        zf = zf.conjugate()
                        
                        a += zf
                    pass
                pass 
                
                moments[y0, x0] = pixel*a/ns_count/ns_count
            pass
        pass
    
        moments = moments*ds
    
        moment = np.sum( moments ) 
        
        elapsed = time() - then
        
        log.info( f"Elapsed time to calculate zernike moment = {elapsed}" )
        log.info( f"zernike moment = {moment}" )    
    
        return moment
                
    pass # -- moment

    @profile
    def image_reconstruct(self, img, t = 20, k = 1 ):
        then = time()
        
        h = img.shape[0]
        w = img.shape[1]
        
        radius = max( h, w )/sqrt(2)
        
        img_recon = np.zeros([h, w], dtype=np.complex)
        
        moments = {}
        
        for y in range(h) :
            for x in range(w) :
                pixel = 0
                for n in range(t + 1 ):
                    for m in range( - n, n+1) :
                        ry = (y - h/2)/radius
                        rx = (x - w/2)/radius
                        
                        key = f"{n}:{m}"
                        moment = None 
                        
                        if key in moments : 
                            moment = moments[ key ]
                        else :
                            moment = None 
                        pass
                        
                        if not moment :
                            moment = self.zernike_moment(img, n, m, k)
                            moments[ key ] = moment 
                        pass
                        
                        pixel += (n+1)/pi*moment*self.zernike_function(n, m, rx, ry)
                    pass
                pass
            
                img_recon[y, x] = pixel 
            pass
        pass
    
        elapsed = time() - then
        
        log.info( f"Elapsed time to reconstruct an image = {elapsed}" )
        
        return img_recon
    pass # -- image_reconstruct    

pass # -- class zernike moment

if __name__ == '__main__':
    log.info( "Hello ...\n" )
    
    from skimage import data
    from skimage import color
    from skimage.transform import rescale 
    img = data.camera()
    
    import mahotas
    img = mahotas.demos.load('lena')
    
    img = color.rgb2gray( img )
    
    log.info( f"image shape = {img.shape}" )
    
    rescale_width = 50 
    
    if rescale_width :
        img = rescale(img, rescale_width/img.shape[1], anti_aliasing=True)
        
        log.info( f"image shape = {img.shape}" )
    pass
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(nrows=2, ncols=1)
    axes = axes.ravel()
    
    ax_idx = 0     
    ax = axes[ ax_idx ]
    ax.imshow( img , cmap='gray')
    ax.set_xlabel( 'original image')    

    zernike = Zernike()
    
    Ts = [ 20, 40, 60, 80, 100, 120 ]
    Ks = [ 1, 3, 5, 7 ]
    
    t = Ts[0] 
    k = Ks[0]
    img_reconst = zernike.image_reconstruct(img, t=t, k=k)
    
    img_reconst = img_reconst.real
    
    ax_idx += 1     
    ax = axes[ ax_idx ]
    
    img_diff = img - img_reconst
    
    gmax = np.max( img_reconst ) # 복원된 이미지의 회색조 최대값 
    
    mse = np.sum( np.square( img_diff ) )/(img_diff.shape[0]*img_diff.shape[1])
    
    psnr = 10*log10(gmax*gmax/mse)
    
    log.info( f"t={t}, k={k}, psnr = {psnr:.2f}" )
    
    title = f"t={t}, k={k}, psnr = {psnr:.2f}"
    ax.imshow( img_reconst, cmap='gray' )
    ax.set_xlabel( title )
    
    #moment = zernike.zernike_moment(img, 10, 10, k=4)
    
    print_profile()
    
    plt.tight_layout()
    plt.show()

    log.info( "\nGood bye!" )
pass