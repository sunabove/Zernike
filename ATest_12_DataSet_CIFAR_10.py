import os
import torch
import tarfile
import matplotlib.pyplot as plt

from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

def plot_dataset_cifar_10() :
    if not os.path.isfile( ".download/cifar10.tgz" ) : 
        dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
        download_url(dataset_url, './download/' )

        with tarfile.open( './download/cifar10.tgz', 'r:gz') as tar:
            tar.extractall( path='./data' )
    pass

    data_dir = './data/cifar10'
    dataset = ImageFolder( data_dir + '/train', transform=ToTensor())
    classes = dataset.classes

    print( "classes = ", classes )

    print( type( dataset ), len( dataset ) )

    fs = fontsize = 16
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = fs

    row_cnt = 10 ; col_cnt = 5
    fig, charts = plt.subplots( row_cnt, col_cnt, figsize=( 2*col_cnt, 2*row_cnt) )
    charts = charts.flatten() if row_cnt*col_cnt > 1 else [charts]

    def show_example(r, c, img, label):
        idx = r*col_cnt + c

        klass = classes[ label ]
        size = img.size()
        chart = charts[idx]

        chart.imshow( img.permute(1, 2, 0) )

        if idx == 4 :
            ticks = torch.arange( 0, size[1] + 1, 10 )
            ticklabels = [ f"{int(tick)}" for tick in ticks ]

            chart.set_xticks( ticks )
            chart.set_xticklabels( ticklabels, fontsize=10 )
            chart.set_yticks( ticks )
            chart.set_yticklabels( ticklabels, fontsize=10 )
            #chart.set_yticks( torch.arange( 0, size[2] + 1, 10 ) )
        else : 
            chart.set_xticks( [] )
            chart.set_yticks( [] )
        pass

        if r == 0 : chart.set_title( f"[{c + 1}]" )
        if c == 0 : chart.set_ylabel( f"{klass.upper()}", fontsize=fs ) 

    for r in range( row_cnt ) :
        for c in range( col_cnt ) :
            show_example( r, c , *dataset[ len(dataset)//len(classes)*r + c ] )
        pass
    pass 

    #plt.tight_layout()
    result_figure_file = f"./result/dataset_overview_cifar-10_{row_cnt}_{col_cnt}.png"
    print( f"Result figure file = {result_figure_file}" )
    plt.savefig( result_figure_file )
    plt.show()

pass