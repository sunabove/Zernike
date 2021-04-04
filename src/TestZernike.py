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

def test_zernike_image_restore(is_jupyter = 1) :
    print( "Hello ..." )
    
    line = "*"*100
    
    from skimage import data
    from skimage import color
    from skimage.transform import rescale 
    img = data.camera()
    
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
    
    zernike = Zernike()
    
    Ts = [ 10, 20, 40 ]
    Ks = [ 1, 3, 5, 7 ]    
    
    for t in Ts :
        for k in Ks :
            print(line)
            
            img_reconst = zernike.image_reconstruct(img, t=t, k=k)
            
            img_reconst = img_reconst.real
            
            img_diff = img - img_reconst
            
            gmax = np.max( img_reconst ) # 복원된 이미지의 회색조 최대값 
            
            mse = np.sum( np.square( img_diff ) )/(img_diff.shape[0]*img_diff.shape[1])
            
            psnr = 10*log10(gmax*gmax/mse)
            
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