from __future__ import division
import cv2
import numpy as np
import math
import os
import random

""" CONTAINS COMMON UTILITY METHODS FOR ALL PHASES OF THE VECTORIZATION ALGORITHM """

MOVING = 0
STOPPED = 1
NBH_STOP = 2

""" Used to store information about moving points """
class Pixel:
    def __init__(self, x, y, gradx, grady):
        self.x = x
        self.y = y
        self.init_x = x
        self.init_y = y
        self.gradx = gradx
        self.grady = grady
        self.stopped = MOVING
        self.lsw = 0 # local stroke width
        self.isCorner = False

    def __str__(self):
        return "Point (%s, %s): %s, %s, %s, %s, %s, %s" % (self.x, self.y, self.gradx, self.grady, self.init_x, self.init_y, self.lsw, self.stopped)

    def get_local_radius(self):
        return math.sqrt((self.x-self.init_x)**2 + (self.y-self.init_y)**2)

    @staticmethod
    def pixel2coords(pixel_list):
        coords = []
        for p in pixel_list:
            coords.append([np.float32(p.x), np.float32(p.y)])
        return coords

    @staticmethod
    def pixel2coords_SEP(pixel_list):
        x = []
        y =[]
        for p in pixel_list:
            x.append(np.float32(p.x))
            y.append(np.float32(p.y))
        return x, y

    @staticmethod
    def pixel2coords_FLIP(pixel_list):
        coords = []
        for p in pixel_list:
            coords.append([np.float32(p.y), np.float32(p.x)])
        return coords

    @staticmethod
    def getPixelCoordsDict(pixel_list):
        d = dict([])
        for p in pixel_list:
            d[(p.x, p.y)] = p
        return d

""" Converts pixel to a String representation"""
def pix2str(pix):
    return "(%s,%s,%s,%s,%s,%s,%s)" % (pix.x, pix.y, pix.gradx, pix.grady, pix.init_x, pix.init_y, pix.stopped)

""" Parses a string and returns a Pixel Object"""
def str2pix(str):
    str = str[1:len(str)-1]
    attr = str.split(",")
    pix = Pixel((float)(attr[0]), (float)(attr[1]), (float)(attr[2]), (float)(attr[3]))
    pix.init_x = (float)(attr[4])
    pix.init_y = (float)(attr[5])
    if attr[6] == "0":
        pix.stopped = MOVING
    elif attr[6] == "2":
        pix.stopped = NBH_STOP
    else:
        pix.stopped = STOPPED
    return pix

""" Compares 2 pixel objects using their x and y coordinates """
def cmpPixels(p1, p2):
    if cmp(p1.y, p2.y) == 0:
        return cmp(p1.x, p2.x)
    return cmp(p1.y, p2.y)

""" Given a hashed grid, located all the neighbours of a particular points"""
def get_neighbours(point_map, p, shape):

    i = R(p.x)
    j = R(p.y)

    if i < 0 or j < 0 or i >= shape[0] or j >= shape[1]: return []

    neighbours = []
    neighbours.extend(point_map[(i, j)])
    if j != shape[1]-1: neighbours.extend(point_map[(i, j+1)])
    if j != 0: neighbours.extend(point_map[(i, j-1)])

    if i != shape[0]-1: neighbours.extend(point_map[(i + 1, j)])
    if i != shape[0]-1 and j != shape[1]-1: neighbours.extend(point_map[(i + 1, j + 1)])
    if i != shape[0]-1 and j != 0: neighbours.extend(point_map[(i + 1, j - 1)])

    if i != 0: neighbours.extend(point_map[(i - 1, j)])
    if i != 0 and j != shape[1]-1: neighbours.extend(point_map[(i - 1, j + 1)])
    if i != 0 and j != 0: neighbours.extend(point_map[(i - 1, j - 1)])

    return neighbours

""" A more general method that finds all points in an arbitrary-sized neighbourhood """
def get_neighbours_x(point_map, p, shape, x):

    i = R(p.x)
    j = R(p.y)

    if i < 0 or j < 0 or i >= shape[0] or j >= shape[1]: return []

    neighbours = []
    neighbours.extend(point_map[(i, j)])

    for n in xrange(-x, x+1):
        for m in xrange(-x, x+1):
            if (i+n) < 0 or (i+n) >= shape[0]: continue
            if (j+m) < 0 or (j+m) >= shape[1]: continue

            neighbours.extend(point_map[(i+n,j+m)])

    return neighbours

""" Prints a matrix object to the console """
def print_matrix(mat):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            print("%d, " % mat.item(i,j)),
        print("")

def display_img(img, title, wait):
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(title, img)
    if wait:
        cv2.waitKey(0)
        #cv2.destroyAllWindows()

""" Creates a hashed grid given a list of point objects"""
def getPointSet(img, thresh):
    pointset = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img.item(i,j) > thresh:
                pointset.append([i,j])
    return pointset

""" Maps a list of continuous 2D points to a matrix object"""
def mapToImage(pointSet, shape):
    img = np.zeros(shape, dtype=np.uint8)
    for p in pointSet:
        if round(p.x) < 0 or round(p.x) >= shape[0]: continue
        if round(p.y) < 0 or round(p.y) >= shape[1]: continue
        img.itemset(((int)(round(p.x)), (int)(round(p.y))), 255)
    img_inv = cv2.bitwise_not(img)
    return img_inv

""" Maps a list of curves to a matrix object """
def mapCurvesToImg(curves, shape):
    img = np.zeros(shape, dtype=np.uint8)
    for curve in curves:
        for p in curve:
            if round(p.x) < 0 or round(p.x) >= shape[0]: continue
            if round(p.y) < 0 or round(p.y) >= shape[1]: continue
            img.itemset(((int)(round(p.x)), (int)(round(p.y))), 255)
    img_inv = cv2.bitwise_not(img)
    return img_inv

""" Maps a list of curves to a matrix object using circles """
def mapCurvesToImg_circles(curves, shape):
    img = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    for curve in curves:
        for p in curve:
            if round(p.x) < 0 or round(p.x) >= shape[0]: continue
            if round(p.y) < 0 or round(p.y) >= shape[1]: continue
            cv2.circle(img, (R(p.y), R(p.x)), 1, (0, 255, 255), -1)
    img_inv = cv2.bitwise_not(img)
    return img_inv

""" Maps a list of moving points to a matrix object, using colours to distinguish types of points """
def mapToImage_moving(pointSet, shape):
    img = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    for p in pointSet:
        if round(p.x) < 0 or round(p.x) >= shape[0]: continue
        if round(p.y) < 0 or round(p.y) >= shape[1]: continue
        if p.stopped == STOPPED: # black
            img.itemset(((int)(round(p.x)), (int)(round(p.y)), 0), 255)
            img.itemset(((int)(round(p.x)), (int)(round(p.y)), 1), 255)
            img.itemset(((int)(round(p.x)), (int)(round(p.y)), 2), 255)
        elif p.stopped == NBH_STOP: # red?
            img.itemset(((int)(round(p.x)), (int)(round(p.y)), 1), 255)
            img.itemset(((int)(round(p.x)), (int)(round(p.y)), 2), 255)
        else: # MOVING (green)
            img.itemset(((int)(round(p.x)), (int)(round(p.y)), 0), 255)
            img.itemset(((int)(round(p.x)), (int)(round(p.y)), 2), 255)
    img_inv = cv2.bitwise_not(img)
    return img_inv

def mapToImage_corners(pointSet, shape):
    img = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    for p in pointSet:
        if round(p.x) < 0 or round(p.x) >= shape[0]: continue
        if round(p.y) < 0 or round(p.y) >= shape[1]: continue
        if not p.isCorner:
            img.itemset(((int)(round(p.x)), (int)(round(p.y)), 0), 255)
            img.itemset(((int)(round(p.x)), (int)(round(p.y)), 1), 255)
            img.itemset(((int)(round(p.x)), (int)(round(p.y)), 2), 255)
        else:
            img.itemset(((int)(round(p.x)), (int)(round(p.y)), 0), 255)
            img.itemset(((int)(round(p.x)), (int)(round(p.y)), 1), 255)
    img_inv = cv2.bitwise_not(img)
    return img_inv

""" Maps a set of points to a file """
def save_point_set(filename, artist, dir, moving_points):
    DW("%s/%s" % (dir, artist))
    f = open("%s/%s/%s" % (dir, artist, filename), 'w')
    for p in moving_points:
        if p.stopped:
            f.write("%s,%s,%s,%s,%s,%s,%s,%s" % (p.x, p.y, p.gradx, p.grady, p.init_x, p.init_y, 1, p.lsw))
        else:
            f.write("%s,%s,%s,%s,%s,%s,%s,%s" % (p.x, p.y, p.gradx, p.grady, p.init_x, p.init_y, 0, p.lsw))
        f.write(";")
    f.close()

def save_scarce_point_set(filename, artist, dir, point_map, shape):
    DW("%s/%s" % (dir, artist))
    f = open("%s/%s/%s" % (dir, artist, filename), 'w')
    for i in range(shape[0]):
        for j in range(shape[1]):
            vals = point_map[(i, j)]
            #if len(vals) != 0: print("(%d, %d): %s" % (i, j, vals))
            n = len(vals)
            #if n!= 0: print("n: %d" % n)
            n = (int)(math.ceil(n/1))
            random.shuffle(vals)
            for i in xrange(n):
                p = vals[i]
                if p.stopped:
                    f.write("%s,%s,%s,%s,%s,%s,%s,%s" % (p.x, p.y, p.gradx, p.grady, p.init_x, p.init_y, 1, p.lsw))
                else:
                    f.write("%s,%s,%s,%s,%s,%s,%s,%s" % (p.x, p.y, p.gradx, p.grady, p.init_x, p.init_y, 0, p.lsw))
                f.write(";")
    f.close()

""" Maps a list of point objects from a given file """
def load_point_map(filename, artist, dir, shape):
    f = open("%s/%s/%s" % (dir, artist, filename), 'r')
    txt = f.read()
    moving_points = []
    point_map = dict([])

    for i in range(shape[0]):
        for j in range(shape[1]):
            point_map[(i,j)] = []

    points = txt.split(";")
    for p in points[:-1]:
        attr = p.split(",")
        pix = Pixel((float)(attr[0]), (float)(attr[1]), (float)(attr[2]), (float)(attr[3]))
        pix.init_x = (float)(attr[4])
        pix.init_y = (float)(attr[5])
        if attr[6] == "0":
            pix.stopped = MOVING
        elif attr[6] == "2":
            pix.stopped = NBH_STOP
        else:
            pix.stopped = STOPPED
        if len(attr) >= 7:
            pix.lsw = (float)(attr[7])
        moving_points.append(pix)
        point_map[(R(pix.x), R(pix.y))].append(pix)
    f.close()
    return moving_points, point_map

""" Saves a list of curves to a file """
def save_curves(filename, artist, dir, curves):
    DW("%s/%s" % (dir, artist))
    f = open("%s/%s/%s" % (dir, artist, filename), 'w')
    for curve in curves:
        for p in curve:
            if p.stopped:
                f.write("%s,%s,%s,%s,%s,%s,%s,%s" % (p.x, p.y, p.gradx, p.grady, p.init_x, p.init_y, 1, p.lsw))
            else:
                f.write("%s,%s,%s,%s,%s,%s,%s,%s" % (p.x, p.y, p.gradx, p.grady, p.init_x, p.init_y, 0, p.lsw))
            f.write(";")
        f.write("\n")
    f.close()

def save_curves_matlab(filename, artist, dir, curves, shape):
    DW("%s/%s" % (dir, artist))
    f = open("%s/%s/%s_MATLAB.m" % (dir, artist, filename), 'w')

    curve_counter = 0
    for curve in curves:
        curve_counter += 1

        x_list = []
        y_list = []
        for p in curve:
            x_list.append(p.y)
            y_list.append(-p.x)
        f.write("x%d = %s;\n" % (curve_counter, x_list))
        f.write("y%d = %s;\n" % (curve_counter, y_list))

    f.write("\nplot(")
    for i in xrange(1, curve_counter+1):
        f.write("x%d, y%d, " % (i, i))
    f.write("'LineWidth', 2),\n")
    f.write("axis([%d %d %d %d]),\n" % (0, shape[1], -shape[0], 0))
    f.write("title('%s'), grid on;\n" % filename)
    f.write("filename = '\Users\PJF\PycharmProjects\cv_projects\\vectorized\%s\%s.png';\n" % (artist, filename))
    f.write("saveas(gcf,filename)")

    f.close()

def save_curves_matlab_spline(filename, artist, dir, curves, shape):
    DW("%s/%s" % (dir, artist))
    f = open("%s/%s/%s_MATLAB.m" % (dir, artist, filename), 'w')

    curve_counter = 0
    idx = []
    for curve in curves:

        print("Curve length: %d" % len(curve))
        if len(curve) < 2: continue
        curve_counter += 1

        x_list = []
        y_list = []
        for p in curve:
            x_list.append(p.y)
            y_list.append(-p.x)

        # REMOVE DUPLICATES
        prev_x = 0
        new_x = []
        new_y = []
        for i in xrange(len(x_list)):
            if abs(x_list[i] - prev_x) > 0.001:
                new_x.append(x_list[i])
                new_y.append(y_list[i])
                prev_x = x_list[i]

        print("Previous length: %d" % len(x_list))
        print("New length: %d" % len(new_x))
        x_list = new_x
        y_list = new_y

        if len(new_x) < 2:
            curve_counter -= 1
            continue

        f.write("x%d = %s;\n" % (curve_counter, x_list))
        f.write("y%d = %s;\n" % (curve_counter, y_list))
        f.write("cs%d = csapi(x%d, y%d);\n" % (curve_counter, curve_counter, curve_counter))
        idx.append(curve_counter)

    for i in idx:
        if len(curves[i-1]) < 2: continue
        f.write("fnplt(cs%d, 2)\n" % i)
        f.write("hold on\n")
    f.write("axis([%d %d %d %d]),\n" % (0, shape[1], -shape[0], 0))
    f.write("title('%s'), grid on;\n" % filename)
    f.write("filename = '\Users\PJF\PycharmProjects\cv_projects\\vectorized\%s\%s.png';\n" % (artist, filename))
    f.write("saveas(gcf,filename)")

    f.close()

def save_buddy_points(filename, artist, dir, buddy_points):
    DW("%s/%s" % (dir, artist))
    f = open("%s/%s/%s" % (dir, artist, filename), 'w')
    for pair in buddy_points:
        if pair[0].stopped:
            f.write("%s,%s,%s,%s,%s,%s,%s,%s" % (pair[0].x, pair[0].y, pair[0].gradx, pair[0].grady, pair[0].init_x, pair[0].init_y, 1, pair[0].lsw))
        else:
            f.write("%s,%s,%s,%s,%s,%s,%s,%s" % (pair[0].x, pair[0].y, pair[0].gradx, pair[0].grady, pair[0].init_x, pair[0].init_y, 0, pair[0].lsw))
        f.write(":")
        if pair[1].stopped:
            f.write("%s,%s,%s,%s,%s,%s,%s,%s" % (pair[1].x, pair[1].y, pair[1].gradx, pair[1].grady, pair[1].init_x, pair[1].init_y, 1, pair[1].lsw))
        else:
            f.write("%s,%s,%s,%s,%s,%s,%s,%s" % (pair[1].x, pair[1].y, pair[1].gradx, pair[1].grady, pair[1].init_x, pair[1].init_y, 0, pair[1].lsw))
        f.write(";")
    f.close()

def load_point_set(filename, artist, dir):
    f = open("%s/%s/%s" % (dir, artist, filename), 'r')
    txt = f.read()
    moving_points = []
    points = txt.split(";")
    for p in points[:-1]:
        attr = p.split(",")
        pix = Pixel((float)(attr[0]), (float)(attr[1]), (float)(attr[2]), (float)(attr[3]))
        pix.init_x = (float)(attr[4])
        pix.init_y = (float)(attr[5])
        if attr[6] == "0": pix.stopped = MOVING
        elif attr[6] == "2": pix.stopped = NBH_STOP
        else: pix.stopped = STOPPED
        if len(attr) >= 7:
            pix.lsw = (float)(attr[7])
        moving_points.append(pix)
    f.close()
    return moving_points

def load_curves(filename, artist, dir):
    f = open("%s/%s/%s" % (dir, artist, filename), 'r')
    txt = f.read()
    curves_final = []
    curves = txt.split("\n")
    for curve in curves[:-1]:
        points_final = []
        points = curve.split(";")
        for p in points[:-1]:
            attr = p.split(",")
            pix = Pixel((float)(attr[0]), (float)(attr[1]), (float)(attr[2]), (float)(attr[3]))
            pix.init_x = (float)(attr[4])
            pix.init_y = (float)(attr[5])
            if attr[6] == "0": pix.stopped = MOVING
            elif attr[6] == "2": pix.stopped = NBH_STOP
            else: pix.stopped = STOPPED
            if len(attr) >= 7:
                pix.lsw = (float)(attr[7])
            points_final.append(pix)
        curves_final.append(points_final)
    f.close()
    return curves_final

def load_buddy_points(filename, artist, dir):
    f = open("%s/%s/%s" % (dir, artist, filename), 'r')
    txt = f.read()
    buddy_points = []
    pairs = txt.split(";")
    for pair in pairs[:-1]:
        points = pair.split[":"]
        duo = []
        for p in points:
            attr = p.split(",")
            pix = Pixel((float)(attr[0]), (float)(attr[1]), (float)(attr[2]), (float)(attr[3]))
            pix.init_x = (float)(attr[4])
            pix.init_y = (float)(attr[5])
            if attr[6] == "0": pix.stopped = MOVING
            elif attr[6] == "2": pix.stopped = NBH_STOP
            else: pix.stopped = STOPPED
            if len(attr) >= 7:
                pix.lsw = (float)(attr[7])
            duo.append(pix)
        buddy_points.append(duo)
    f.close()
    return buddy_points

""" If a directory doesn't exist, create it"""
def DW(dirName): # short for dirWrapper
    dir_path = "/Users/PJF/PycharmProjects/cv_projects/%s" %dirName
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dirName

""" Rounds a number """
def R(num):
    return (int)(round(num))

""" Returns the Euclidean distance between two points """
def pointDist(p1, p2):
    return math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2)

def thicken_line(mat):
    for i in range(1,mat.shape[0]-1):
        for j in range(1, mat.shape[1]-1):
            # if cell is white
            if mat.item(i,j) == 255:
                if mat.item(i-1, j) == 0 and mat.item(i+1, j) == 0:
                    mat.itemset((i,j), 0)
                elif mat.item(i, j-1) == 0 and mat.item(i, j+1) == 0:
                    mat.itemset((i, j), 0)