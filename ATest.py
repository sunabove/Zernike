if __name__ == "__main__" : 

    import numpy
    import cupy
    import torch

    if 1 :
        if 1 : 
            n = numpy.zeros( (10), dtype=numpy.complex_ )
            print( f"n size = { n.size }" )
            print( f"n item size = {n.itemsize}" )
            print( f"n memory size = { n.size*n.itemsize}" )
        pass

        if 1 : 
            c = cupy.zeros( (10), dtype=cupy.complex_ )
            print( f"c size = { c.size }" )
            print( f"c item size = {c.itemsize}" )
            print( f"c memory size = { c.size*c.itemsize}" )
        pass

        if 1 : 
            t = torch.zeros( (10), dtype=torch.complex64 )
            print( f"t size = { int( t.numel() ) }" )
            print( f"t item size = { t.element_size() }" )
            print( f"t memory size = { t.numel()*t.element_size() }" )
        pass

    elif 1 : 
        print( "int(0.5) = ", int(0.5) )

        # pytorch imaginary number test

        

        # 배열 여부 체크 
        def is_array( data ) :
            if isinstance( data, torch.Tensor ) :
                return ( data.ndim > 0 )
            else :
                return isinstance( data, (list, tuple, numpy.ndarray ) )
            pass
        pass # is_array

        # 스칼라 여부 체크 
        def is_scalar( data ) :
            return not is_array( data, )
        pass # is_scalar

        print()

        m = torch.tensor( [1, 2] )
        n = torch.tensor( [3, 4] )

        print( f"pi = {torch.pi}")
        print( f"dot(m, n) = {torch.dot(m,n)}" )
        print( f"m*n = {m*n}" )
        print()

        Ks = torch.arange( 1,  6 + 1, 1 )
        print( f"device = {Ks.get_device()}" )
        print( f"dtype = {Ks.dtype}" )
        print( f"type Ks[0] = {type(Ks[0])}" )
        print( f"is_scalar = {is_scalar(Ks[0])}" )

        print( [""] * 4 )

        import torch

        use_gpu = 1
        device_name = "cuda" if use_gpu else "cpu"
        device = torch.device( device_name)

        print( "device type = ", type( device ) )

        a = torch.rand( (2, 2), dtype=torch.cdouble, device=device )
        b = a.to( device )
        c = a.cpu()

        print( a )
        print( a.data_ptr() == b.data_ptr() )
        print( a.data_ptr() == c.data_ptr() )

        import scipy 
        print( "scipy version = ", scipy.__version__ )
        from scipy.special import factorial

        print( "factorial(0) = ", factorial(0) )

        import os

        src_dir = os.path.dirname( os.path.abspath(__file__) )

        print( "os.path.basename = ", os.path.basename( __file__ ) )
        print( "os.path.basename = ", os.path.basename( __file__ ) )


        from pathlib import Path
        print( "Path( __file__ ).stem = ", Path( __file__ ).stem) 
        print( "Path( __file__ ).name = ", Path( __file__ ).name)
    pass

pass
