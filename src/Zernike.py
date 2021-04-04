# -*- coding: utf-8 -*-

import warnings
from _ast import Pass
#from gevent.libev.corecext import NONE

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import sys

import logging as log
log.basicConfig(stream=sys.stdout, format='%(levelname)s %(filename)s:%(lineno)04d %(message)s', datefmt='%Y-%m-%d:%H:%M:%S', level=log.INFO )

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
        
        db_name = 'c:/temp/zernike.db'
        db_name = ':memory:'
        
        self.conn = sqlite3.connect(db_name)
        #self.conn.set_trace_callback(print)
        self.cursor = self.conn.cursor()

        self.create_table( kwargs )
        
        self.poly_factors = {}
    pass

    def __del__(self) :
        self.conn.commit()

        log.info( "Commit" )
    pass

    def create_table(self, kwargs):
        cursor = self.cursor

        log.info( "Checking tables ..." )

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
        
        log.info( "Done. checking tables ..." )

    pass # -- create_table

    def poly_factor(self, n, m, k):
        key = f"{2*n + 1 - k}:{k}:{n + m + 1 - k}:{n - m - k}"
        
        poly_factors = self.poly_factors
        
        factor = None 
        
        if key in poly_factors :
            factor = poly_factors[ key ]
        else :
            factor = factorial(2*n + 1 - k)/factorial(k)/factorial(n + m + 1 - k)/factorial(n - m - k)
            poly_factors[ key ] = factor
        pass
        
        return factor
    pass

    @profile
    def calc_polynomial(self, n, m, rho):        
        m = abs( m )
        
        rs = np.zeros([ n - m + 1 ], dtype=np.double )
            
        for k in range( len( rs ) ) :            
            r = (-1) ** (k % 4)
            
            if n - k :
                r *= rho**(n - k)
            pass
        
            r *= self.poly_factor(n, m, k)
            
            rs[k] = r
        pass

        return np.sum( rs )
    pass # -- polynomial

    @profile
    def np_numeric_polynomial(self, x, n, m ):
        y = np.array( [ self.select_polynomial(n, m, rho) for rho in x ] )
         
        return y
    pass

    def polynomial(self, n, m, rho):
        return self.select_polynomial(n, m, rho)
    pass

    @profile
    def select_polynomial(self, n, m, rho):
        
        R = 0
                 
        if rho == 1 :
            R = 1
        else : 
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
        pass

        return R
    pass # -- select

    def function(self, n, m, x, y):
        return self.zernike_function(n, m, x, y)
    pass

    @profile
    def zernike_function(self, n, m, x, y):
        debug = self.debug 
        
        rho = math.sqrt( x*x + y*y )

        v = 1.0
        
        theta = m*atan2(y, x)        
        
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

    def moment(self, img, n, m, k=1 ):
        return self.zernike_moment(img, n, m, k)
    pass # -- moment

    def img_radius(self, img):
        h = img.shape[0]
        w = img.shape[1]
        
        radius = max( h/2, w/2 )*sqrt(2)
        
        #radius = max( h, w )/sqrt(2)
        
        return radius
    pass # -- img_radius

    @profile
    def zernike_moment(self, img, n, m, k=1 ):
        debug = self.debug 
        
        log.info( f"n={n}, m={m}, k={k}" )
        
        then = time()
        
        if k < 1 :
            k = 1
        pass
    
        h = img.shape[0]
        w = img.shape[1]
        
        radius = self.img_radius(img)
        
        debug and log.info( f"Radius = {radius}" )
        
        dx = 1/k/radius
        dy = dx
            
        moments = np.zeros([h*k*w*k], dtype=np.complex) 
        
        idx = 0 
        for y0, row in enumerate( img ) :
            for x0, pixel in enumerate( row ) :
                for y in np.arange(y0, y0 + 1, 1/k) :
                    # convert coordinate into unit circle coordinate system
                    ry = (y - h/2)/radius
                        
                    for x in np.arange(x0, x0 + 1, 1/k) :                        
                        # convert coordinate into unit circle coordinate system
                        rx = (x - w/2)/radius
                        
                        if 0 and ry > 1 or rx > 1 or ry*ry + rx*rx > 1 :
                            log.info( f"rx = {rx}, ry = {ry}")                        
                        
                            log.info( "invalid coordinate conversion." )
                            
                            import sys
                            sys.exit( -1 )
                        pass
                        
                        zf = self.function(n, m, rx, ry)
                        zf = zf.conjugate()
                        
                        moments[ idx ] = pixel*dx*dy*zf
                        
                        idx += 1
                    pass
                pass    
            pass
        pass
    
        moment = np.sum( moments ) 
        
        elapsed = time() - then
        
        log.info( f"Elapsed time to calculate zernike moment = {elapsed}" )
        log.info( f"Zernike moment = {moment}" )    
    
        return moment                
    pass # -- moment

    @profile
    def image_reconstruct(self, img, t = 20, k = 1 ):
        then = time()
        
        h = img.shape[0]
        w = img.shape[1]
        
        radius = self.img_radius(img)
        
        img_recon = np.zeros([h, w], dtype=np.complex)
        
        moments = {}
        
        for y in range(h) :
            ry = (y - h/2)/radius
                                        
            for x in range(w) :
                pixel = 0                                
                rx = (x - w/2)/radius
                                
                for n in range(t + 1):
                    p = np.zeros([2*n + 1], dtype=np.complex)                    
                    idx = 0 
                    
                    for m in range( - n, n + 1) :
                        key = f"{n}:{m}:{k}"
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
                        
                        p[idx] = moment*self.zernike_function(n, m, rx, ry)
                        idx += 1
                    pass
                
                    pixel += (n+1)/pi*np.sum(p)
                pass
            
                img_recon[y, x] = pixel 
            pass
        pass
    
        elapsed = time() - then
        
        log.info( f"Elapsed time to reconstruct an image = {elapsed}" )
        
        return img_recon
    pass # -- image_reconstruct    

pass # -- class zernike moment
