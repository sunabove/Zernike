# -*- coding: utf-8 -*-

from Zernike import *

def test_zernike_image_restore() :
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
    
    zernike = Zernike()
    
    Ts = [ 20, 40 ]
    Ks = [ 1, 3 ]    
    
    import matplotlib.pyplot as plt
    nrows = len(Ts)
    ncols = len(Ks)
    
    plt.rcParams['figure.figsize'] = [14, 14]
    fig, axes = plt.subplots( nrows=nrows, ncols=ncols)
    axes = axes.ravel()    
    ax_idx = -1
    
    for t in Ts :
        for k in Ks :
            img_reconst = zernike.image_reconstruct(img, t=t, k=k)
            
            img_reconst = img_reconst.real
            
            ax_idx += 1     
            ax = axes[ ax_idx ]
            
            img_diff = img - img_reconst
            
            gmax = np.max( img_reconst ) # 복원된 이미지의 회색조 최대값 
            
            mse = np.sum( np.square( img_diff ) )/(img_diff.shape[0]*img_diff.shape[1])
            
            psnr = 10*log10(gmax*gmax/mse)
            
            title = f"t={t}, k={k}, psnr = {psnr:.2f}"
            
            ax.imshow( img_reconst, cmap='gray' )
            ax.set_title( title + "\n" )            
        pass        
    pass 
    
    plt.get_current_fig_manager().canvas.set_window_title('Pseudo-Zernike Moment Image Restoration')
    plt.tight_layout()
    plt.show()
    
    print_profile()

    log.info( "\nGood bye!" )
pass # --test_zernike

if __name__ == '__main__':
    test_zernike_image_restore()    
pass