# -*- coding: utf-8 -*-

import warnings, sys, logging as log
import sqlite3, numpy as np, math, cmath, xlsxwriter

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

log.basicConfig(stream=sys.stdout, format='%(levelname)s %(filename)s:%(lineno)04d %(message)s', datefmt='%Y-%m-%d:%H:%M:%S', level=log.INFO )

from _ast import Pass
from os.path import join
from glob import glob

from math import factorial, perm, atan2, pi, sqrt, log10
from time import time, sleep 
from Profiler import *

class Zernike :
    def __init__(self, **kwargs) :
        self.debug = False 
        
        self.poly_factors = {}
        self.momentsDict = {}
        self.two_pi = 2*pi
        
        db_name = 'c:/temp/zernike.db'
        db_name = ':memory:'
        
        self.conn = sqlite3.connect(db_name)
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
        debug = self.debug
        
        debug and log.info( f"n={n}, m={m}, n - m + 1 = {n - m + 1}" )
        
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
            
            if cnt :
                for row in rows:
                    R = row[ 0 ]
                    
                    break
                pass
            else :
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
        
        if m < 0 :
            v = self.zernike_function(n, -m, x, y)
            return v.conjugate()
        pass
        
        rho = math.sqrt( x*x + y*y )

        v = 1.0
        
        theta = m*atan2(y, x)
        theta = theta % self.two_pi
        
        calc_time = -1        

        cursor = self.cursor
            
        rows = cursor.execute(
            "SELECT vx, vy FROM zernike_function WHERE n = ? and m = ? and rho = ? and theta = ?",
            [n, m, rho, theta],
        ).fetchall()

        cnt = len( rows )
        
        if cnt :
            vx = 0
            vy = 0 
            
            for row in rows : 
                vx = row[0]
                vy = row[1]
                
                break
            pass
        
            v = vx + 1j*vy
        else :
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
    
        debug and log.info(f"V(n={n}, m={m}, rho={rho:.4f}, theta={theta:.4f}, x={x:.4f}, y={y:.4f}) = {v}, calc_time={calc_time}")

        return v
    pass # -- zernike_function

    def moment(self, img, n, m, k=1 ):
        return self.zernike_moment(img, n, m, k)
    pass # -- moment

    def img_radius(self, img):
        h = img.shape[0]
        w = img.shape[1]
        
        radius = 0
        include_all = 1
        if include_all :
            radius = max( h/2, w/2 )*sqrt(2)
        else : 
            radius = max( h/2, w/2 )
        pass
         
        return radius
    pass # -- img_radius

    @profile
    def zernike_moment(self, img, n, m, k=1, T=-1 ):
        debug = self.debug 
        
        debug and log.info( f"T={T}, n={n}, m={m}, k={k}" )
        
        if m < 0 :
            moment = self.zernike_moment(img, n, -m, k, T)
            
            return moment.conjugate()
        pass
        
        then = time()
        
        momentsDict = self.momentsDict
        
        if k < 1 :
            k = 1
        pass
    
        key = self.moment_key(n, m, k)
        
        moment = None 
        
        if key in momentsDict : 
            moment = momentsDict[ key ] 
        else :
            h = img.shape[0]
            w = img.shape[1]
            
            radius = self.img_radius(img)
            
            debug and log.info( f"Radius = {radius}" )
            
            dx = 1/(k*2*radius)
            dy = dx
            ds = dx*dy
                
            moments = np.zeros([np.count_nonzero(img)*k*k], dtype=np.complex) 
            
            idx = 0 
            for y0, row in enumerate( img ) :
                for x0, pixel in enumerate( row ) :
                    if pixel : 
                        for y in np.arange(y0, y0 + 1, 1/k) :
                            # convert coordinate into unit circle coordinate system
                            ry = (y - h/2)/radius
                                
                            for x in np.arange(x0, x0 + 1, 1/k) :                        
                                # convert coordinate into unit circle coordinate system
                                rx = (x - w/2)/radius
                                
                                if ry*ry + rx*rx <= 1 :                                
                                    zf = self.function(n, m, rx, ry)
                                    zf = zf.conjugate()
                                    
                                    moments[ idx ] = pixel*zf*ds
                                    
                                    idx += 1
                                pass
                            pass
                        pass
                    pass
                pass
            pass    
        
            moment = np.sum( moments )            
            moment = moment
                            
            momentsDict[ key ] = moment 
        pass
        
        elapsed = time() - then
        
        debug and log.info( f"Elapsed time to calculate zernike moment = {elapsed}" )
        debug and log.info( f"Zernike moment = {moment}" )    
    
        return moment                
    pass # -- moment

    def moment_key(self, n, m, k):
        key = f"{n:3d}:{m:3d}:{k}"
        
        return key
    pass

    @profile
    def moments_list(self, img, t, k=1):
        funName = "Calculating moments ...."
        log.info( funName )
        
        then = time()
        
        moments = {}
        
        for n in range( 0, t + 1 ) :
            for m in range( -n, n + 1 ) :
                key = self.moment_key(n, m, k)                
                moment = self.zernike_moment(img, n, m, k=k, T=t)
                
                moments[ key ] = moment      
            pass
        pass
    
        elapsed = time() - then
        
        self.print_moments_list(moments, t, k)
        
        log.info( f"Elapsed time = {elapsed:.2f}" )
        log.info( "Done." + funName )     
        
        return moments 
    pass # -- moments list

    def print_moments_list(self, moments, t, k):
        keys = moments.keys()
        keys = sorted( keys )
        
        # 엑셀 파일(workbook)을 만들고, 엑셀 시트를 하나 추가함.
        path = f'C:\\Temp\\zernike_{t:03d}_{k}.xlsx'
        
        workbook = xlsxwriter.Workbook( path )
        worksheet = workbook.add_worksheet()
        
        row = 0
        
        if 1 :
            col = 0
                
            worksheet.write(row, col, "k" ); col += 1
            worksheet.write(row, col, "n" ); col += 1
            worksheet.write(row, col, "m" ); col += 1
            worksheet.write(row, col, "real" ); col += 1
            worksheet.write(row, col, "imag" ); col += 1
            
            row += 1
        pass
        
        print( "Moments")
        for n in range( t + 1 ) :
            for m in range( -n, n + 1 ) :
                key = self.moment_key(n, m, k)
                moment = moments[key]
                print( f"{key} = {moment}" )
                
                col = 0
                
                worksheet.write(row, col, k ); col += 1
                worksheet.write(row, col, n ); col += 1
                worksheet.write(row, col, m ); col += 1
                worksheet.write(row, col, moment.real ); col += 1
                worksheet.write(row, col, moment.imag ); col += 1
                
                row += 1
            pass
        pass
    
        workbook.close()
    pass

    @profile
    def image_reconstruct(self, img, moments, t = 20, k = 1 ):
        log.info( f"t={t}, k={k}. Image reconstruccting ..." )
        
        then = time()
        
        h = img.shape[0]
        w = img.shape[1]
        
        radius = self.img_radius(img)
        
        img_recon = np.zeros([h, w], dtype=np.complex)
        
        for y in range(h) :
            ry = (y - h/2)/radius
            
            entered = False 
                               
            for x in range(w) :
                pixel = 0                                
                rx = (x - w/2)/radius
                
                if ry*ry + rx*rx > 1 :
                    if entered :
                        break
                    pass
                else :
                    entered = True 
                              
                    for n in range(t + 1):
                        p = np.zeros([2*n + 1], dtype=np.complex)                    
                        idx = 0 
                        
                        for m in range( - n, n + 1) :
                            v = 0 
                            zf = self.zernike_function(n, m, rx, ry)
                            
                            if zf : 
                                key = self.moment_key(n, m, k)
                                moment = moments[ key ]
                                v = moment*zf
                            pass
                            
                            p[idx] = v
                            idx += 1
                        pass
                    
                        pixel += (n+1)/pi*np.sum(p)
                    pass
                
                    img_recon[y, x] = pixel
                pass 
            pass
        pass
    
        elapsed = time() - then
        
        log.info( f"t={t}, k={k}. Elapsed time to reconstruct an image = {elapsed:.2f}" )
        
        return img_recon
    pass # -- image_reconstruct    

pass # -- class zernike moment

if __name__ == '__main__':
    import TestZernike
    TestZernike.test_zernike_image_restore( is_jupyter = 0)    
pass
