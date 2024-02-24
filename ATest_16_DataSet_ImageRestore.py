import os
import torch
import matplotlib.pyplot as plt

import skimage as sk
import cv2 as cv

def plot_dataset_image_restore() :

    src_dir = os.path.dirname( os.path.abspath(__file__) )

    images = [ ]

    images.append( cv.imread( f'{src_dir}/image/lenna.png', 0 ) )
    images.append( sk.data.astronaut() )
    images.append( sk.data.camera() )
    images.append( sk.data.brick() )
    images.append( sk.data.moon() )
    images.append( sk.data.grass() )

    classes = [ "lena", "astronaut", "camera", "brick", "moon", "grass" ]

    fs = fontsize = 16
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = fs

    col_cnt = 3
    row_cnt = int( float(len( images ))/col_cnt + 0.5 )
    fig, charts = plt.subplots( row_cnt, col_cnt, figsize=( 3*col_cnt, 3*row_cnt), tight_layout=1 )
    charts = charts.flatten() if row_cnt*col_cnt > 1 else [charts]

    for idx, ( img, klass ) in enumerate( zip( images, classes ) ) :
        chart = charts[idx]

        shape = img.shape
        channel = shape[ -1 ]
        if channel == 3 : 
            img = sk.color.rgb2gray( img )
        pass

        chart.imshow( img, cmap=plt.cm.gray )
        chart.set_title( klass )
    pass 

    #plt.tight_layout()
    result_figure_file = f"./result/dataset_overview_image_restore.png"
    print( f"Result figure file = {result_figure_file}" )
    plt.savefig( result_figure_file )
    plt.show()

pass

if __name__ == "__main__" :
    plot_dataset_image_restore()
pass