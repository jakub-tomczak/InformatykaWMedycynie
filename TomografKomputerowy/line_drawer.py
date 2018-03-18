import numpy as np

def bresenham(image, start, end):
    x1,y1 = start
    x2,y2 = end

    xi = 1
    yi = 1
    if x2 - x1 < 0:
        xi = -1
    if y2 - y1 < 0:
        y1 = -1

    dx = np.abs(x2 - x1)
    dy = np.abs(y2 - y1)
    e = dx / 2
    image[y1,x1] = 1

    if dx == 0 or dy == 0:
        draw_straight_line(image, start, end)
    else:
        if dx > dy:
            #OX priority
            for i in range(0, dx):
                x1 = x1 + xi
                e = e - dy
                if e < 0:
                    y1 = y1 + yi
                    e = e + dx
                image[y1, x1] = 1
        else:
            #OY priority
            for i in range(0, dy):
                y1 = y1 + yi
                e = e - dx
                if e < 0:
                    x1 = x1 + xi
                    e = e + dy
                image[y1, x1] = 1

def draw_straight_line(image, start, end):
    if start[0] == end[0]:
        image[:,start[0]] = 1
    if end[1] == start[1]:
        image[start[1],:] = 1

def calculatePosition(angle, radius, circle_middle = (0,0)):
    rad = angle / 180 * np.pi
    return (circle_middle[0] + (int)(radius*np.cos(rad)), circle_middle[1] - (int)(radius*np.sin(rad)))

'''Calculates position x1, y1 and changes these values according to the available space'''
def calculatePositionSafe(angle, diameter, circle_middle = (0,0)):
    x1, y1 = calculatePosition(angle, (int)(diameter / 2) - 1, circle_middle)
    return check_borders((x1, y1), diameter)

def check_value(val, size):
    if val <= 0:
        val = 1
    if val >=size:
        val = size -1
    return val

def check_borders(point, size):
    return (check_value(point[0], size),check_value(point[1], size))


def draw_line(image, angle = None, diameter = None, center = None, start_point = None, end_point = None):
    if start_point != None and end_point != None:
        bresenham(image, start_point, end_point)
        return

    if diameter == None:
        diameter = len(image)
    radius = (int)(diameter / 2) - 1
    if center == None:
        center = (radius, radius)

    x1, y1 = calculatePosition(angle, radius, center)
    x1, y1 = check_borders((x1,y1), diameter)

    print(x1,y1)

    x2 = diameter - x1
    y2 = diameter - y1
    bresenham(image, (x1, y1), (x2, y2))
