from __future__ import division
import numpy as np
import cv2
import math
import util
import time

DISPLAY_IMG = False

# DECLARE CONSTANTS
SPEED_FACTOR = 0.0001
STOPPING_POINT = 0.95 # percentage
MIN_SPEED = 0.5
MIN_NEIGHBOURS = 2
NBHD_RADIUS = math.sqrt(2)

""" Returns list of all pixels whose gradient is larger than the threshold """
def get_moving_pixels(f, img, gradx, grady):

    pointset = []
    point_map = dict([])

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            point_map[(i,j)] = []

    # STEP 1: MEASURE GRADIENT MAGNITUDES + GET THRESHOLD
    grad_mag = np.sqrt(np.square(gradx) + np.square(grady))
    max = grad_mag.max()
    thresh = max / 10
    print("Threshold value: %s" % thresh)
    f.write("Threshold value: %s\n" % thresh)

    # STEP 2: ITERATE THROUGH PIXELS
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):

            # THRESHOLD TEST
            if grad_mag.item(i, j) > thresh:
                # ADD POINT TO POINTSET
                p = util.Pixel(i, j, grady.item(i, j), gradx.item(i, j))
                pointset.append(p)
                point_map[(i,j)].append(p)

    return pointset, point_map

def get_neighbours(point_map, p, shape):

    i = util.R(p.x)
    j = util.R(p.y)

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

""" Define minimum speed for moving pixels"""
def getMinSpeed(speed):
    if speed >= 0:
        return max(speed, MIN_SPEED)
    else:
        return min(speed, -MIN_SPEED)

""" Returns True if the pixel meets any of the stopping conditions, i.e. they have moved passed a pixel
    of the opposite band or they have less than 2 neighbours """
def stop_pixel(point_map, pixel, shape):
    neighbours = get_neighbours(point_map, pixel, shape)
    if len(neighbours) < MIN_NEIGHBOURS:
        return util.NBH_STOP
    opposing_band = 0
    for p2 in neighbours:
        if util.pointDist(pixel, p2) > NBHD_RADIUS: continue
        prod = (p2.gradx) * (pixel.gradx) + (p2.grady) * (pixel.grady)
        if prod < 0:
            opposing_band += 1
            stop_cond = (p2.x - pixel.x) * pixel.gradx + (p2.y - pixel.y) * pixel.grady
            if stop_cond < 0: return util.STOPPED
    return util.MOVING

""" Returns percentage of moving pixels that are stopped """
def get_percent_stopped_pixels(moving_points):
    stp = 0.0
    for p in moving_points:
        if p.stopped:
            stp += 1
    return (stp/len(moving_points))

""" Removes all moving points that haven't stopped
    Likely, most of these pixel are unwanted noise"""
def clean_up(moving_points):
    new_points = []
    for p in moving_points:
        if p.stopped == util.STOPPED: new_points.append(p)
    return new_points

""" Calculate the local stroke thickness at each point using the points in the pixel's nbhd """
def calculate_local_stroke_thickness(f, point_map, moving_points, shape):
    radii = []

    for pixel in moving_points:
        neighbours = get_neighbours(point_map, pixel, shape)
        max_rad = 0
        for p2 in neighbours:
            if util.pointDist(p2, pixel) > NBHD_RADIUS: continue
            if p2.get_local_radius() > max_rad:
                max_rad = p2.get_local_radius()
        pixel.lsw = max_rad
        radii.append(pixel.lsw)

    print("MIN: %.4f" % min(radii))
    print("MAX: %.4f" % max(radii))
    print("AVERAGE: %.4f" % (sum(radii) / len(radii)))
    f.write("MIN Local Stroke Thickness: %.4f\n" % min(radii))
    f.write("MAX Local Stroke Thickness: %.4f\n" % max(radii))
    f.write("AVERAGE Local Stroke Thickness: %.4f\n" % (sum(radii) / len(radii)))

""" --- MAIN FUNCTION ---
    Given a greyscale input image, returns a set of clustered moving pixels """
def get_cluster_points(f, img, shape, img_name, artist, dir="pixelClustering"):

    print("IMAGE SHAPE: (%s, %s)" % (shape[0], shape[1]))

    f.write("Speed factor: %f\n" % SPEED_FACTOR)
    f.write("Stopping Point: %f\n" % STOPPING_POINT)
    f.write("Min Speed: %f\n" % MIN_SPEED)
    f.write("Min Neighbours: %f\n" % MIN_NEIGHBOURS)
    f.write("Neighbourhood Radius: %f\n\n" % NBHD_RADIUS)

    blur = cv2.GaussianBlur(img, (7,7), 1)
    util.display_img(blur, "Blurred", DISPLAY_IMG)
    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)

    scaled_x = cv2.convertScaleAbs(sobelx)
    scaled_y = cv2.convertScaleAbs(sobely)
    cv2.imwrite('%s/%s/%s/%s_sobelx.png' % (dir, artist, img_name, img_name), scaled_x)
    cv2.imwrite('%s/%s/%s/%s_sobely.png' % (dir, artist, img_name, img_name), scaled_y)

    moving_points, point_map = get_moving_pixels(f, img, sobelx, sobely)
    print("Number of moving points: %s" % len(moving_points))
    f.write("Initial number of moving points: %d\n" % len(moving_points))

    newImg = util.mapToImage(moving_points, img.shape)
    cv2.imwrite('%s/%s/%s/%s_COLOR_%d.png' % (dir, artist, img_name, img_name, 1), newImg)
    #util.display_img(newImg, "Mapped from Point Set!", DISPLAY_IMG)

    # MOVEMENT PHASE

    start_time = time.time()
    movement_phase = True
    counter = 0
    while movement_phase:
        counter += 1

        bad_pix = []
        # movement phase
        for i in xrange(len(moving_points)):

            idx = (util.R(moving_points[i].x), util.R(moving_points[i].y))

            x_speed = getMinSpeed(SPEED_FACTOR * moving_points[i].gradx)
            y_speed = getMinSpeed(SPEED_FACTOR * moving_points[i].grady)

            if moving_points[i].stopped: continue
            moving_points[i].x = moving_points[i].x - x_speed
            moving_points[i].y = moving_points[i].y - y_speed

            idx2 = (util.R(moving_points[i].x), util.R(moving_points[i].y))

            # update space in bin
            if idx !=  idx2:
                point_map[idx].remove(moving_points[i])
                if idx2 in point_map:
                    point_map[idx2].append(moving_points[i])
                else:
                    print("BAD POINT FOUND: (%.3f, %.3f)" % (moving_points[i].x, moving_points[i].y))
                    bad_pix.append(moving_points[i])

        for p in bad_pix:
            moving_points.remove(p)

        # evaluation phase
        for p in moving_points:
            stop = stop_pixel(point_map, p, shape)
            if stop:
                p.stopped = stop
        print("PHASE COMPLETE.")

        newImg2 = util.mapToImage_moving(moving_points, img.shape)
        cv2.imwrite('%s/%s/%s/%s_COLOR_%d.png' % (dir, artist, img_name, img_name, (counter+1)), newImg2)
        #util.display_img(newImg2, "Mapped from Point Set %s!" % (counter+1), DISPLAY_IMG

        # Determine whether to break the loop
        stp = get_percent_stopped_pixels(moving_points)
        print("Percentage stopped: %f" % stp)
        f.write("Rep %d. Percentage stopped: %f\n" % (counter, stp))
        if stp > STOPPING_POINT:
            print("STOPPING THE LOOP")
            movement_phase = False

    end_time = time.time()
    moving_points = clean_up(moving_points)
    f.write("\nTOTAL Number of iterations: %d\n" % counter)
    f.write("Time taken (in seconds): %f\n" % (end_time - start_time))
    f.write("Final number of moving points: %d\n" % len(moving_points))

    newImg2 = util.mapToImage(moving_points, img.shape)
    util.display_img(newImg2, "THE FINAL RESULT", DISPLAY_IMG)
    cv2.imwrite('%s/%s/%s/%s_COLOR_%s.png' % (dir, artist, img_name, img_name, "FINAL"), newImg2)

    cv2.destroyAllWindows()

    calculate_local_stroke_thickness(f, point_map, moving_points, shape)

    util.save_point_set("%s.txt" % img_name, artist, "pointSet", moving_points)

    return moving_points

""" Run the script independantly """

if __name__ == "__main__":

    artists = ['foster', 'levesque', 'koudelka', 'vidal', 'fiala', 'fiala', 'levesque', 'vidal', 'fiala', 'foster']
    images = ['test2', 'levesque2', 'koudelka5', 'vidal2', 'fiala_001', 'fiala_010', 'levesque1', 'vidal6', 'fiala_014', 'test1']

    artists = ['fiala']
    images = ['fiala_001']

    for i in xrange(len(images)):
        artist = artists[i]
        img_name = images[i]
        orig_name = img_name

        # Load a color image in grayscale
        img = cv2.imread("samples/%s/%s.jpg" % (artist, img_name), 0)
        util.display_img(img, "Initial Image", DISPLAY_IMG)

        img_name = orig_name + "_SCARCE"

        print("Clustering %s, %s" % (artist, img_name))
        util.DW("pixelClustering/%s/%s" % (artist, img_name))
        util.DW("pointSet/%s" % artist)
        util.DW("stats/%s" % artist)

        f = open("%s/%s/%s_pixelClustering_NEW.txt" % ("stats", artist, img_name), 'w')
        moving_points = get_cluster_points(f, img, img.shape, img_name, artist)
        f.close()


