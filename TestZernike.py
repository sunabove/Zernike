# -*- coding: utf-8 -*-

from Zernike import *

import os
import matplotlib.pyplot as plt
from click._compat import _is_jupyter_kernel_output

image_save_cnt = 0 

def save_image(img, fileName):
    # C:/temp 폴더에 결과 파일을 저정합니다.

    directory = "C:/temp"

    if os.path.exists(directory):
        if not os.path.isdir(directory):
            os.remove(directory)
            os.mkdir(directory)
        pass
    else:
        os.mkdir(directory)
    pass

    global image_save_cnt

    fn = os.path.join( directory, f"{image_save_cnt:03d}_{fileName}" )

    fn = fn.replace("\\", "/")

    log.info( f"fn={fn}")
    
    plt.imsave(fn, img, cmap="gray")
    
    image_save_cnt += 1

    return fn
pass  # -- img_file_name

def load_image_by_skimage():
    from skimage import data
    from skimage import color
    from skimage.transform import rescale 
    
    import mahotas
    img_name = "lena"
    img = mahotas.demos.load( img_name )
    
    img = color.rgb2gray( img )
    
    save_image(img, f"{img_name}_org.png")
    
    log.info( f"image shape = {img.shape}" )
    
    rescale_width = 50 
    
    if rescale_width :
        scale = rescale_width/img.shape[1]
        img = rescale(img, scale, anti_aliasing=True)
        
        save_image(img, f"{img_name}_size_{rescale_width}.png")
        
        log.info( f"image shape = {img.shape}" )
    pass

    return img, img_name
pass

def load_image():
    from skimage import data
    
    import cv2, cv2 as cv 
    
    import mahotas
    img_name = "lena"
    img = mahotas.demos.load( img_name )
    
    save_image(img, f"{img_name}_org.png")
    
    img = img.astype(np.uint8)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    log.info( f"image shape = {img.shape}" )
    
    rescale_width = img.shape[1]/4
    
    if rescale_width :
        height = img.shape[0]
        width = img.shape[1]
        
        scale = rescale_width/img.shape[1]
        
        dim = ( int( width*scale ), int( height*scale ) )
        img = cv2.resize(img, dim, interpolation = cv2.INTER_LINEAR)
        
        save_image(img, f"{img_name}_size_{rescale_width}.png")
        
        log.info( f"image shape = {img.shape}" )
    pass

    return img, img_name
pass

def test_zernike_image_restore(is_jupyter = 1) :
    print( "Hello ..." )
    
    line = "*"*100
    
    img, img_name = load_image()

    zernike = Zernike()
    
    if True : 
        h = img.shape[0]
        w = img.shape[1]
        radius = zernike.img_radius(img)
        
        for y, row in enumerate( img ) :
            ry = (y - h/2)/radius
            for x, pixel in enumerate( row ) :
                rx = (x - h/2)/radius
                if ry*ry + rx*rx > 1 :
                    img[y, x] = 0
                pass
            pass
        pass
    
        save_image(img, f"{img_name}_unit_circle.png")
    pass
    
    Ts = [ 10 ]
    Ks = [ 1, 3 ]
    
    for t in Ts :
        for k in Ks :
            print(line)
            
            moments = zernike.moments_list(img, t, k )
            
            img_zernike = zernike.image_reconstruct(img, moments, t=t, k=k)
            
            img_reconst = np.absolute( img_zernike )
            
            img_diff = img - img_reconst
            
            omax = np.max( img )
            gmax = np.max( img_diff ) # 복원된 이미지의 회색조 최대값 
            
            log.info( f"img shape = {img.shape}" )
            log.info( f"img_diff shape = {img_diff.shape}, size = {img_diff.size}" )
            
            mse = np.sum( np.square( img_diff )/img_diff.size ) 
            
            psnr = 10*math.log10(gmax*gmax/mse)
            
            log.info( f"omax = {omax}, gmax = {gmax}, mse = {mse}, psnr = {psnr:.2f}") 
            
            title = f"t={t}, k={k}, psnr={psnr:.2f}"
            fileName = f"{title}.png"
            
            save_image(img_reconst, fileName)
            
            if is_jupyter :
                plt.imshow( img )
                plt.show()
            pass
        
            print(line)
            print()          
        pass        
    pass 
    
    print_profile()

    print( "Good bye!" )
pass # --test_zernike

if __name__ == '__main__':
    test_zernike_image_restore( is_jupyter = 0)    
pass