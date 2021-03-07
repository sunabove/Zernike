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
from random import random
from math import factorial, perm, atan2, pi
#import time
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

        dropAnyway = True
        if dropAnyway :
            tables = [ "fun", "polynomial", ]
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
                     , x DOUBLE, y DOUBLE
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
            m = abs( m )

            for s in range( n - m + 1 ) :
                #R *= factorial(2*n + 1 - s)/factorial(s)/factorial(n + m + 1 - s)/factorial(n - m - s)
                #R *= factorial(2 * n + 1 - s) / factorial(s) / perm(n + m + 1 - s, n - m - s)
                r = 1.0
                r *= perm(2 * n + 1 - s, s) / factorial(n + m + 1 - s) / factorial(n - m - s)
                r *= (-1) ** (s % 4)
                r *= rho**(n - s)

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
        theta = 0

        if rho == 0 : 
            v = 0 
        elif rho != 0 :
            r = self.select_polynomial(n, m, rho)

            log.info(f"R(n={n}, m={m}, rho={rho:.4f}, x={x:.4f}, y={y:.4f}) = {r}")
            
            theta = atan2(y, x)
            theta = (theta*m) % pi 
            
            e = cmath.exp( 1j*theta )
            
            v = r*e
        pass
    
        log.info(f"V(n={n}, m={m}, rho={rho:.4f}, theta={theta:.4f}, x={x:.4f}, y={y:.4f}) = {v}")

        return v
    pass # -- zernike_function

pass # -- class Zernike

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
    plt.show()

    zernike = Zernike()

    for n in range( 10 ) :
        for m in range( n + 1 ) :
            x = random()
            y = random()
            x = 1
            y = 0
            if x*x + y*y <= 1 : 
                zernike.zernike_function(n, m, x, y )
            pass
        pass
    pass

    print_profile()

    log.info( "\nGood bye!" )
pass