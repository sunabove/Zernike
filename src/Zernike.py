# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import logging as log
log.basicConfig( format='%(levelname)s %(filename)s:%(lineno)04d %(message)s', datefmt='%Y-%m-%d:%H:%M:%S', level=log.INFO )

from os.path import join
from glob import glob

import sqlite3

import numpy as np
import math, cmath
from math import factorial, perm, atan2, pi, sqrt
from time import time, sleep

from Profiler import *

class Zernike :
    def __init__(self) :
        self.conn = sqlite3.connect("c:/temp/zernike.db")
        self.cursor = self.conn.cursor()

        self.create_table()
    pass

    def __del__(self) :
        self.conn.commit()

        log.info( "Commit" )
    pass

    def create_table(self):
        cursor = self.cursor

        log.info( "Create tables ...\n " )

        dropAnyway = False 
        if dropAnyway :
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
        R = 1.0

        if rho == 0 :
            R = 0
        else :
            for k in range( 0, n - abs(m) + 1 ) :
                #R *= factorial(2*n + 1 - k)/factorial(k)/factorial(n + m + 1 - k)/factorial(n - m - k)
                #R *= factorial(2 * n + 1 - k) / factorial(k) / perm(n + m + 1 - k, n - m - k)
                r = (-1) ** (k % 4)
                r *= perm(2*n + 1 - k, k) / factorial(n + abs(m) + 1 - k) / factorial(n - abs(m) - k)
                r *= rho**(n - k)

                R += r
            pass
        pass

        return R
    pass # -- polynomial

    @profile
    def select_polynomial(self, n, m, rho):
        R = 1.0

        if rho == 0 :
            R = 0
        else :
            cursor = self.cursor

            rows = cursor.execute(
                "SELECT value FROM zernike_polynomial WHERE n = ? and m = ? and rho = ?",
                [n, m, rho],
            ).fetchall()

            cnt = len( rows )

            if cnt < 1 :
                then = time()
                R = self.calc_polynomial(n, m, rho)
                now = time()

                calc_time = now - then

                sql = '''
                    INSERT INTO zernike_polynomial( n, m, rho, value, calc_time )
                    VALUES ( ?, ?, ?, ?, ? )
                    '''
                cursor.execute(sql, [n, m, rho, R, calc_time])
            else :
                for row in rows:
                    R = row[ 0 ]
                pass
            pass
        pass

        return R
    pass # -- select

    @profile
    def zernike_function(self, n, m, x, y ):
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
            
            if cnt > 0 :
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
    
                log.info(f"R(n={n}, m={m}, rho={rho:.4f}, x={x:.4f}, y={y:.4f}) = {r}")
                
                e = cmath.exp( 1j*theta )
            
                v = r*e
                
                now = time()
                
                calc_time = now - then
                
                sql = '''
                    INSERT INTO zernike_function( n, m, rho, theta, vx, vy, calc_time )
                    VALUES ( ?, ?, ?, ?, ?, ?, ? )
                    '''
                cursor.execute(sql, [n, m, rho, theta, v.real, v.imag, calc_time])
            pass
        pass
    
        log.info(f"V(n={n}, m={m}, rho={rho:.4f}, theta={theta:.4f}, x={x:.4f}, y={y:.4f}) = {v}, calc_time={calc_time}")

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
        if k < 1 :
            k = 1
        pass
    
        ns = np.arange( 0, 1, 1/k)
        ns_count = len( ns )
        
        h = img.shape[0]
        w = img.shape[1]
        
        radius = max( h, w )/sqrt(2)
        log.info( f"Radius = {radius}" )
        
        moments = np.zeros([w*ns_count, h*ns_count]).astype(complex)
        
        for y0, row in enumerate(img) :
            for x0, pixel in enumerate( row ) :
                for r, dy in enumerate( ns ) : 
                    for c, dx in enumerate( ns ) :
                        y = y0 + dy
                        x = x0 + dx
                        
                        # convert coordinate into unit circle coordinate system
                        y = (y - h/2)/radius
                        x = (x - w/2)/radius
                        
                        zf = self.zernike_function(n, m, x, y)
                        a = pixel*zf
                    pass
                pass
            pass
        pass
    
        moment = np.sum( moments ) 
    
        return moment
                
    pass # -- moment

pass # -- class zernike moment

if __name__ == '__main__':
    log.info( "Hello ...\n" )
    
    from skimage import data
    from skimage import color
    img = data.camera()
    
    import mahotas
    img = mahotas.demos.load('lena')
    
    img = color.rgb2gray( img )
    
    import matplotlib.pyplot as plt
    
    print( f"image shape = {img.shape}" )
    
    plt.imshow( img , cmap='gray')    

    zernike = Zernike()
    
    Ts = [ 20, 40, 60, 80, 100, 120 ]
    Ks = [ 1, 3, 5, 7 ]
    
    moment = zernike.zernike_moment(img, 10, 10, k=1)
    
    print( f"zernike moment = {moment}" )

    print_profile()
    
    plt.show()

    log.info( "\nGood bye!" )
pass