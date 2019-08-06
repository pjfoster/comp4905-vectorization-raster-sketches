import util

""" Contains functions used to automatically generate MATLAB scripts """

def save_matlab_suite(filename, artist, dir, curves, shape):
    save_curves_matlab(filename, artist, dir, curves, shape)
    save_curves_matlab_basic_spline(filename, artist, dir, curves, shape)
    save_curves_matlab_natural_spline(filename, artist, dir, curves, shape)
    save_curves_matlab_smoothing_spline(filename, artist, dir, curves, shape)

def save_curves_matlab(filename, artist, dir, curves, shape):
    util.DW("%s/%s" % (dir, artist))
    f = open("%s/%s/%s_MATLAB_POINTS.m" % (dir, artist, filename), 'w')

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
    f.write("filename = '\Users\PJF\PycharmProjects\cv_projects\\vectorized\%s\%s_POINTS.png';\n" % (artist, filename))
    f.write("saveas(gcf,filename)")

    f.close()

def save_curves_matlab_basic_spline(filename, artist, dir, curves, shape):
    util.DW("%s/%s" % (dir, artist))
    f = open("%s/%s/%s_MATLAB_SPLINE.m" % (dir, artist, filename), 'w')

    curve_counter = 0
    idx = []
    for curve in curves:

        #print("Curve length: %d" % len(curve))
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
    f.write("filename = '\Users\PJF\PycharmProjects\cv_projects\\vectorized\%s\%s_SPLINE.png';\n" % (artist, filename))
    f.write("saveas(gcf,filename)")

    f.close()

def save_curves_matlab_smoothing_spline(filename, artist, dir, curves, shape):
    util.DW("%s/%s" % (dir, artist))
    f = open("%s/%s/%s_MATLAB_SMOOTHING.m" % (dir, artist, filename), 'w')

    curve_counter = 0
    idx = []
    for curve in curves:

        #print("Curve length: %d" % len(curve))
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

        x_list = new_x
        y_list = new_y

        if len(new_x) < 2:
            curve_counter -= 1
            continue

        f.write("x%d = %s;\n" % (curve_counter, x_list))
        f.write("y%d = %s;\n" % (curve_counter, y_list))
        f.write("cs%d = csaps(x%d, y%d, 0.5);\n" % (curve_counter, curve_counter, curve_counter))
        idx.append(curve_counter)

    for i in idx:
        if len(curves[i-1]) < 2: continue
        f.write("fnplt(cs%d, 2)\n" % i)
        f.write("hold on\n")
    f.write("axis([%d %d %d %d]),\n" % (0, shape[1], -shape[0], 0))
    f.write("title('%s'), grid on;\n" % filename)
    f.write("filename = '\Users\PJF\PycharmProjects\cv_projects\\vectorized\%s\%s_SMOOTHING.png';\n" % (artist, filename))
    f.write("saveas(gcf,filename)")

    f.close()

def save_curves_matlab_natural_spline(filename, artist, dir, curves, shape):
    util.DW("%s/%s" % (dir, artist))
    f = open("%s/%s/%s_MATLAB_NATURAL.m" % (dir, artist, filename), 'w')

    curve_counter = 0
    idx = []
    for curve in curves:

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

        x_list = new_x
        y_list = new_y

        if len(new_x) < 2:
            curve_counter -= 1
            continue

        f.write("pts%d = [" % curve_counter)
        for x in x_list[:-1]:
            f.write("%f," % x)
        f.write("%f; " % x_list[-1])

        for y in y_list[:-1]:
            f.write("%f," % y)
        f.write("%f];\n " % y_list[-1])

        f.write("cs%d = cscvn(pts%d);\n" % (curve_counter, curve_counter))
        idx.append(curve_counter)

    for i in idx:
        if len(curves[i-1]) < 2: continue
        f.write("fnplt(cs%d, 2)\n" % i)
        f.write("hold on\n")
    f.write("axis([%d %d %d %d]),\n" % (0, shape[1], -shape[0], 0))
    f.write("title('%s'), grid on;\n" % filename)
    f.write("filename = '\Users\PJF\PycharmProjects\cv_projects\\vectorized\%s\%s_NATURAL.png';\n" % (artist, filename))
    f.write("saveas(gcf,filename)")

    f.close()