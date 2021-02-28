# coding: utf-8

import cv2
import matplotlib.pylab as plt
from Zernike import *

if __name__ == '__main__':
    n = 4
    m = 2

    print( '-' * 40 )
    print( 'Calculating Zernike moments ..., n = %d, m = %d' % (n, m) )

    fig, axes = plt.subplots(2, 3)

    pathss = [['Oval_H.png', 'Oval_45.png', 'Oval_V.png'] ,
              ['Shape_0.png', 'Shape_90.png', 'Rectangular_H.png'],
              ]

    zernike = Zernike();

    for r, paths in enumerate(pathss) :
        for c, path in enumerate(paths) :
            img = cv2.imread(path, 0)

            z, amp, phase = zernike.zernike(img, n, m)

            axes[r, c].imshow(img, cmap="gray" )
            axes[r, c].axis('off')

            title = f"A = {amp:.4f}\nPhase = {phase:.2f}"

            axes[r, c].set_title(title)
        pass
    pass

    plt.show()

    print( 'Calculation is complete' )
    print('-' * 40)

pass # -- main
