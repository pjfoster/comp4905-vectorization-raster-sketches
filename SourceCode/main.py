from __future__ import division
import cv2
import numpy as np
import util
import pixel_clustering_2 as pc
import topology_2 as tplg
import gaussian_smoothing_2 as gs
import time
import matlab_file_creator as mfc


artists = ['koudelka','levesque','fiala','vidal','koudelka','koudelka','fiala','levesque','foster']
images = ['koudelka5', 'levesque1', 'fiala_014', 'vidal6', 'koudelka4', 'koudelka3', 'fiala_021','levesque3', 'test3']

artists = ['couture', 'koudelka']
images = ['couture1', 'koudelka6']

for i in xrange(len(images)):

    artist =  artists[i]
    img_name = images[i]

    print("Vectorizing %s" % img_name)

    # Load a color image in grayscale
    img = cv2.imread("samples/%s/%s.jpg" % (artist, img_name), 0)
    util.display_img(img, "Initial Image", False)

    img_name += "_170421_FINAL"

    # ensure that required directories are created
    util.DW("pixelClustering/%s/%s" % (artist, img_name))
    util.DW("pointSet/%s" % artist)
    util.DW("stats/%s" % artist)
    util.DW("curves/%s" % artist)  # I think you only need this in a next step...
    util.DW("topology/%s/%s" % (artist, img_name))
    util.DW("buddyPoints/%s" % artist)
    util.DW("smoothing/%s/%s" % (artist, img_name))
    util.DW("vectorized/%s" % artist)

    f = open("%s/%s/%s_FULL.txt" % ("stats", artist, img_name), 'w')

    """ VECTORIZATION PROCESS """

    start_time = time.time()

    # PIXEL CLUSTERING
    pc.get_cluster_points(f, img, img.shape, img_name, artist)

    #TOPOLOGY EXTRACTION
    moving_points, point_map = util.load_point_map("%s.txt" % img_name, artist, "pointSet", img.shape)

    corners = cv2.goodFeaturesToTrack(img, 100, 0.01, 10)
    corners = np.int0(corners)

    tplg.extract_topology(f, moving_points, point_map, img, corners, artist, img_name)

    # GAUSSIAN SMOOTHING
    curves = util.load_curves("%s.txt" % img_name, artist, "curves")
    mfc.save_matlab_suite("%s_UNSMOOTHED" % img_name, artist, "curves", curves, img.shape)

    gs.smooth_image(f, curves, img, artist, img_name)

    end_time = time.time()

    f.write("\nTOTAL TIME (s): %s" % (end_time-start_time))
    f.close()