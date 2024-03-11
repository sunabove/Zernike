import os
import math
import torch
import matplotlib.pyplot as plt

import skimage
import cv2 as cv

from pathlib import Path

def plot_dataset_image_restore() :

    src_dir = os.path.dirname( os.path.abspath(__file__) )

    img_lbls = [ ]

    img_lbls.append( [ cv.imread( f'{src_dir}/image/lenna.png', 0 ), "lenna" ] )
    img_lbls.append( [ skimage.color.rgb2gray( skimage.data.astronaut() ), "astronaut" ] )
    img_lbls.append( [ skimage.data.camera(), "camera" ] )
    img_lbls.append( [ skimage.data.brick(), "brick" ] )
    img_lbls.append( [ skimage.data.moon(), "moon" ] )
    img_lbls.append( [ skimage.data.grass(), "grass" ] )

    fs = fontsize = 16
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = fs

    col_cnt = 3
    row_cnt = int( float(len( img_lbls ))/col_cnt + 0.5 )
    fig, charts = plt.subplots( row_cnt, col_cnt, figsize=( 3*col_cnt, 3*row_cnt), tight_layout=1 )
    charts = charts.flatten() if row_cnt*col_cnt > 1 else [charts]

    for idx, [ img, lbl ] in enumerate( img_lbls ) :
        chart = charts[idx]

        shape = img.shape
        channel = shape[ -1 ] if len( shape ) > 2 else 1
        h = shape[ 0 ]
        w = shape[ 1 ]

        print( f"[{idx:03}] name = {lbl}, channel = {channel}, shape={shape}", flush=1 )

        if channel == 3 : 
            img = sk.color.rgb2gray( img ) 
        pass

        xticks = torch.arange( 0, w + 1, pow( 10, int( math.log10( w ) ) ) )
        xtick_labels = [ f"{int(t)}" for t in xticks ]
        yticks = torch.arange( 0, h + 1, pow( 10, int( math.log10( h ) ) ) )
        ytick_labels = [ f"{int(t)}" for t in yticks ]

        chart.imshow( img, cmap=plt.cm.gray )
        
        chart.set_title( lbl.capitalize(), fontsize=fs+4 )

        chart.set_xticks( xticks )
        chart.set_xticklabels( xtick_labels, fontsize=fs-4 )
        chart.set_yticks( yticks )
        chart.set_yticklabels( ytick_labels, fontsize=fs-4 )
    pass 

    plt.show()
    
    src_dir = os.path.dirname( os.path.abspath(__file__) )
    file_stem = Path( __file__ ).stem 
    result_figure_file = f"{src_dir}/result/{file_stem.lower()}.png"
    print()
    print( f"Result figure file = {result_figure_file}" )
    plt.savefig( result_figure_file )    

pass

if __name__ == "__main__" :
    plot_dataset_image_restore()
pass