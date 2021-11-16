# -*- coding: utf-8 -*-

import logging as log
log.basicConfig( format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)04d] %(message)s', datefmt='%Y-%m-%d:%H:%M:%S', level=log.INFO )

import os, glob, inspect, numpy as np

from Profiler import *

class Common :

    def __init__(self):
        pass
    pass

    def print_profile( self ):
        print_profile()
    pass

    def show_versions(self):
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        import sys

        log.info(f"Python version : {sys.version}")

        log.info(f"Numpy version : {np.__version__}")

        try:
            import cv2
            log.info( f"OpenCV version : {cv2.__version__}" )
        except Exception as e:
            log.info( f"{e}" )
            log.info("OpenCV is not installed on this machine.")
        pass

        showTensorFlow = 0
        if showTensorFlow :
            # TensorFlow and tf.keras
            import tensorflow as tf
            from tensorflow import keras

            log.info( f"TensorFlow version : {tf.__version__}" )
            log.info( f"Keras version : {keras.__version__}" )

            # print gpu spec

            from tensorflow.python.client import device_lib
            log.info(device_lib.list_local_devices())

            if tf.test.gpu_device_name():
                log.info( f'Default GPU Device: {tf.test.gpu_device_name()}' )
            else:
                log.info("Please install GPU version of TF")
            pass
            # -- print gpu spec
        pass
    pass # -- show versions

    def open_file_or_folder(self, path) :
        ''' open file or folder by an explorer'''
        import webbrowser as web
        web.open( path )
    pass # -- open_file_or_folder

    def file_name_except_path_ext(self, path):
        # 확장자와 파일 패스를 제외한 파일 이름 구하기.
        head, file_name = os.path.split(path)

        dot_idx = file_name.rfind(".")
        file_name = file_name[: dot_idx]

        return file_name
    pass # file_name_except_path_ext

    def is_writable(self, file):
        # 파일 쓰기 가능 여부 체크
        if os.path.exists(file):
            try:
                os.rename(file, file)

                return True
            except OSError as e:
                log.info( f"is is not writable. {file}" )
            pass
        else :
            return True
        pass

        return False
    pass # -- is_writable

    def to_excel_letter(self, col):
        excel_column = ""

        AZ_len = ord('Z') - ord('A') + 1

        def to_alhapet(num):
            c = chr(ord('A') + int(num))
            return c

        pass  # -- to_alhapet

        while col > 0:
            col, remainder = divmod(col - 1, AZ_len)

            excel_column = to_alhapet(remainder) + excel_column
        pass

        if not excel_column:
            excel_column = "A"
        pass

        return excel_column
    pass  # -- to_excel_letter

    def remove_space_except_first(self, s):
        # 첫 글자를 제외한 나머지 모음을 삭제한다.
        import re
        reg_exp = r'[aeiou]'
        s = s[0] + re.reg_str(reg_exp, '', s[1:])

        return s
    pass # -- remove_space_except_first

    def chdir_to_curr_file(self) :
        # 현재 파일의 폴더로 실행 폴더를 이동함.
        cwd = os.getcwd()

        log.info( f"Pwd 1: {cwd}" )

        dir_name = os.path.dirname(__file__) # change working dir to current file

        if dir_name and cwd != dir_name :
            os.chdir( dir_name )
            log.info(f"Pwd 2: {os.getcwd()}")
        pass
    pass # -- chdir_to_curr_file

    def next_file(self, fileName , step = 1, debug = False ) :
        directory = os.path.dirname(fileName)
        log.info(f"dir = {directory}")

        _, ext = os.path.splitext(fileName)
        ext = ext.lower()

        find_files = f"{directory}/*{ext}"
        find_files = find_files.replace( "\\", "/" )
        log.info(f"find_files={find_files}")

        files = glob.glob(find_files)

        idx = -1

        file_next = None

        fileBaseOrg = os.path.basename(fileName)

        for i, file in enumerate( files ):
            fileBase = os.path.basename( file )
            debug and log.info(f"fileBase = {fileBase}")

            if fileBase == fileBaseOrg :
                idx = i + step
                break
            pass
        pass

        if idx < len( files ) :
            file_next = files[ idx ]
        pass

        if file_next is not None :
            file_next = file_next.replace("\\", "/")
        pass

        log.info(f"file_next = {file_next}")

        return file_next
    pass # -- next_file

    def prev_file(self, fileName , debug = False ) :
        directory = os.path.dirname(fileName)
        log.info(f"dir = {directory}")

        _, ext = os.path.splitext(fileName)
        ext = ext.lower()

        find_files = f"{directory}/*{ext}"
        find_files = find_files.replace("\\", "/")
        log.info(f"find_files={find_files}")

        files = glob.glob(find_files)

        fileBaseOrg = os.path.basename(fileName)

        file_prev = None

        for file in files:
            fileBase = os.path.basename(file)
            debug and log.info(f"fileBase = {fileBase}")

            if fileBase == fileBaseOrg or fileBase > fileBaseOrg :
                break
            elif fileBase < fileBaseOrg:
                file_prev = file
            else :
                break
            pass
        pass

        if file_prev is not None :
            file_prev = file_prev.replace("\\", "/")
        pass

        log.info(f"file_prev = {file_prev}")

        return file_prev
    pass # -- next_file

    def save_recent_file(self, settings, fileName ) :
        recent_file_list = settings.value('recent_file_list', [], str)

        if fileName in recent_file_list:
            recent_file_list.remove(fileName)
            recent_file_list.insert(0, fileName)
        else:
            recent_file_list.insert(0, fileName)
        pass

        if len(recent_file_list) > 9:
            recent_file_list.pop(len(recent_file_list) - 1)
        pass

        settings.setValue("recent_file_list", recent_file_list)
    pass # save_recent_file

pass