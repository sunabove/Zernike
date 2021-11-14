from MyZernike import *

if __name__ == '__main__':
    print( "Hello..." )
    import cv2
    import pylab as pl
    from matplotlib import cm

    d = 12

    img = cv2.imread('./image/gl5un.png', 0)
    #img = cv2.imread('./image/messi.png', 0)

    rows, cols = img.shape
    radius = cols//2 if rows > cols else rows//2

    print( "Shape: ", img.shape )

    reconst = zernike_reconstruct(img, radius, d, (rows/2., cols/2.))

    reconst = reconst.reshape(img.shape)

    pl.figure(1)
    pl.imshow(img, cmap=cm.jet, origin = 'upper')
    pl.figure(2)
    pl.imshow(reconst.real, cmap=cm.jet, origin = 'upper')
    pl.show()

    print( "Good bye!" )
pass