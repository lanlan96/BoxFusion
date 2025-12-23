# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Helper functions for calculating 2D and 3D bounding box IoU.

Collected and written by Charles R. Qi
Last modified: Jul 2019
"""
from __future__ import print_function

import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
def polygon_clip(subjectPolygon, clipPolygon):
   """ Clip a polygon with another polygon.

   Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python

   Args:
     subjectPolygon: a list of (x,y) 2d points, any polygon.
     clipPolygon: a list of (x,y) 2d points, has to be *convex*
   Note:
     **points have to be counter-clockwise ordered**

   Return:
     a list of (x,y) vertex point for the intersection polygon.
   """
   def inside(p):
      return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])
 
   def computeIntersection():
      dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
      dp = [ s[0] - e[0], s[1] - e[1] ]
      n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
      n2 = s[0] * e[1] - s[1] * e[0] 
      n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
      return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]
 
   outputList = subjectPolygon
   cp1 = clipPolygon[-1]
 
   for clipVertex in clipPolygon:
      cp2 = clipVertex
      inputList = outputList
      outputList = []
      s = inputList[-1]
 
      for subjectVertex in inputList:
         e = subjectVertex
         if inside(e):
            if not inside(s):
               outputList.append(computeIntersection())
            outputList.append(e)
         elif inside(s):
            outputList.append(computeIntersection())
         s = e
      cp1 = cp2
      if len(outputList) == 0:
          return None
   return(outputList)

def poly_area(x,y):
    """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


def check_duplicate_tuples(inter_p):
    """
    Check whether the second and third tuples in the inter_p list are exactly the same.

    Args:
        inter_p: A list that contains a mix of lists and tuples.

    Returns:
        bool: True if the second and third tuples are exactly the same, False otherwise.
    """
    if len(inter_p) < 3:
        # The list is less than 3 elements, cannot compare
        return False

    # Get the second and third elements
    second = inter_p[1]
    third = inter_p[2]

    # Compare their first and second elements for equality within a small tolerance
    second_0, second_1, third_0, third_1 = second[0], second[1], third[0], third[1]
    if abs(second_0 - third_0) > 1e-6 or abs(second_1 - third_1) > 1e-6:
        return False
    else:
        return True


def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1,p2)

    if inter_p is not None and not check_duplicate_tuples(inter_p):
        # print("inter_p", inter_p)
        hull_inter = ConvexHull(inter_p) #, qhull_options="QJ"
        return inter_p, hull_inter.volume
    else:
        return None, 0.0 


def box3d_vol_ca1m(area,corners):
    ''' corners: (8,3) no assumption on axis direction '''
    v = area*(np.max(corners[:,2]-np.min(corners[:,2])))


    return  v 

def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
    b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
    c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
    return a*b*c

def is_clockwise(p):
    x = p[:,0]
    y = p[:,1]
    return np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)) > 0

def box3d_iou(corners1, corners2):
    ''' Compute 3D bounding box IoU.

    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU

    todo (rqi): add more description on corner points' orders.
    '''
    # corner points are in counter clockwise order
    rect1 = [(corners1[i,0], corners1[i,2]) for i in range(3,-1,-1)]
    rect2 = [(corners2[i,0], corners2[i,2]) for i in range(3,-1,-1)] 
    area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
    area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])
    # print("rect1",rect1,"rect2",rect2)
    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area/(area1+area2-inter_area)
    ymax = min(corners1[0,1], corners2[0,1])
    ymin = max(corners1[4,1], corners2[4,1])
    inter_vol = inter_area * max(0.0, ymax-ymin)
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou, iou_2d


def augment_vertices(corners):
    # List of edges, each defined by a pair of vertex indices
    edges = [
        [0, 1], [0, 4], [1, 5], [4, 5],
        [2, 3], [2, 6], [6, 7], [3, 7],
        [0, 3], [4, 7], [1, 2], [5, 6]
    ]

    # Compute several midpoints for each edge
    midpoints = []
    for edge in edges:
        v1 = corners[edge[0]]
        v2 = corners[edge[1]]
        midpoint = (v1 + v2) / 2        # Regular midpoint (50%)
        midpoint2 = ((v1 + v2) / 3) * 2 # 2/3 of the way from v1 to v2
        midpoint3 = ((v1 + v2) / 3) * 1 # 1/3 of the way from v1 to v2
        midpoints.append(midpoint)
        midpoints.append(midpoint2)
        midpoints.append(midpoint3)

    # Combine the original vertices with the computed midpoints
    combined = np.vstack([corners, midpoints])
    
    return combined


def check_intersection(corners1, corners2):
    """
    Check whether two 3D boxes (represented by their convex hull vertices) intersect.

    Args:
        corners1: ndarray of shape (8, 3), vertices of box 1
        corners2: ndarray of shape (8, 3), vertices of box 2

    Returns:
        bool: True if there is intersection, False otherwise
    """
    # Compute convex hulls for both sets of vertices
    hull1 = ConvexHull(corners1)
    hull2 = ConvexHull(corners2)

    # Add extra points to increase robustness for intersection test
    corners1 = augment_vertices(corners1)
    corners2 = augment_vertices(corners2)

    # Each "equations" is shape [K, 4], representing planes as ax + by + cz + d = 0
    equations1 = hull1.equations
    equations2 = hull2.equations

    # Check if any point in corners1 lies inside the convex hull of corners2
    dot_products1 = np.dot(corners1, equations2[:, :3].T) + equations2[:, 3]  # shape [N, K]
    mask1 = np.all(dot_products1 <= 1e-6, axis=1)  # True if inside for all planes, shape [N,]

    # Check if any point in corners2 lies inside the convex hull of corners1
    dot_products2 = np.dot(corners2, equations1[:, :3].T) + equations1[:, 3]  # shape [N, K]
    mask2 = np.all(dot_products2 <= 1e-6, axis=1)  # True if inside for all planes, shape [N,]

    sum_of_mask = np.sum(mask1) + np.sum(mask2)

    # If any points from either set are contained in the other's convex hull, boxes intersect
    if sum_of_mask > 0:
        return True
    else:
        return False
    


def batch_in_convex_hull_3d(points, corners):
    """
    Batch check if points are inside a 3D convex hull.

    Args:
        points (np.ndarray): Points to check, shape (N, 3).
        corners (np.ndarray): Vertices of the convex hull, shape (M, 3).

    Returns:
        np.ndarray: Boolean mask of shape (N,), True if point is inside the hull, False otherwise.
    """
    # Compute equations for the convex hull faces:
    # Each row is [a, b, c, d] corresponding to a plane a*x + b*y + c*z + d = 0.
    hull = ConvexHull(corners)
    equations = hull.equations  # Shape: [K, 4]

    # Vectorized: For all points, compute signed distance to each hull plane
    dot_products = np.dot(points, equations[:, :3].T) + equations[:, 3]  # Shape: [N, K]

    # A point is inside the convex hull if it lies on the inner side of all planes.
    mask = np.all(dot_products <= 1e-6, axis=1)  # Shape: [N,]

    return mask


def convex_hull_area_2d(points):
    """
    Calculate the area of the convex hull formed by a set of 2D points.
    
    Args:
        points (list): A list of points in the format [[x1, y1], [x2, y2], ...]
        
    Returns:
        float: The area of the convex hull.
    """
    # Step 1: Find the vertices of the convex hull
    hull = ConvexHull(points)
    hull_points = [points[i] for i in hull.vertices]  # Convex hull vertex coordinates
    
    # Step 2: Use the shoelace formula to calculate area
    n = len(hull_points)
    area = 0.0
    for i in range(n):
        x1, y1 = hull_points[i]
        x2, y2 = hull_points[(i + 1) % n]  # Wrap around to form a closed loop
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0

def convex_hull_intersection_area(points1, points2):
    """
    Calculate the intersection area of two convex hulls.

    Args:
        points1 (list or np.ndarray): List of (x, y) tuples representing the first convex hull.
        points2 (list or np.ndarray): List of (x, y) tuples representing the second convex hull.

    Returns:
        tuple:
            - intersection_area (float): The area of the intersection between the two convex hulls.
            - area1 (float): Area of the first convex hull.
            - area2 (float): Area of the second convex hull.
    """
    # Create convex hull polygons from the input point sets
    poly1 = Polygon(points1)
    poly2 = Polygon(points2)
    
    # Calculate the intersection polygon between the two convex polygons
    intersection = poly1.intersection(poly2)
    
    # Return the area of the intersection and the areas of the original two convex hulls
    return intersection.area, poly1.area, poly2.area

def box3d_iou_v2(corners1, corners2):
    ''' Compute 3D bounding box IoU.

    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU

    todo (rqi): add more description on corner points' orders.
    '''
    # corner points are in counter clockwise order
    xmin1,xmax1 = np.min(corners1[:,0]), np.max(corners1[:,0])
    ymin1,ymax1 = np.min(corners1[:,1]), np.max(corners1[:,1])
    xmin2,xmax2 = np.min(corners2[:,0]), np.max(corners2[:,0])
    ymin2,ymax2 = np.min(corners2[:,1]), np.max(corners2[:,1])
    rect1 = [(xmin1, ymin1),(xmin1, ymax1),(xmax1, ymax1),(xmax1, ymin1)]
    rect2 = [(xmin2, ymin2),(xmin2, ymax2),(xmax2, ymax2),(xmax2, ymin2)]

    inter_area,area1,area2 = convex_hull_intersection_area(rect1,rect2)

    # inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area/(area1+area2-inter_area)
    zmax = min(np.max(corners1[:,2]), np.max(corners2[:,2]))
    zmin = max(np.min(corners1[:,2]), np.min(corners2[:,2]))
    inter_vol = inter_area * max(0.0, zmax-zmin)
    # vol1 = box3d_vol(corners1)
    # vol2 = box3d_vol(corners2)
    vol1 = area1 * (np.max(corners1[:,2])-np.min(corners1[:,2]))
    vol2 = area2 * (np.max(corners2[:,2])-np.min(corners2[:,2]))

    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou, iou_2d





def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two 2D bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def box2d_iou(box1, box2):
    ''' Compute 2D bounding box IoU.

    Input:
        box1: tuple of (xmin,ymin,xmax,ymax)
        box2: tuple of (xmin,ymin,xmax,ymax)
    Output:
        iou: 2D IoU scalar
    '''
    return get_iou({'x1':box1[0], 'y1':box1[1], 'x2':box1[2], 'y2':box1[3]}, \
        {'x1':box2[0], 'y1':box2[1], 'x2':box2[2], 'y2':box2[3]})

# -----------------------------------------------------------
# Convert from box parameters to 
# -----------------------------------------------------------
def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                    [0,  1,  0],
                    [-s, 0,  c]])

def roty_batch(t):
    """Rotation about the y-axis.
    t: (x1,x2,...xn)
    return: (x1,x2,...,xn,3,3)
    """
    input_shape = t.shape
    output = np.zeros(tuple(list(input_shape)+[3,3]))
    c = np.cos(t)
    s = np.sin(t)
    output[...,0,0] = c
    output[...,0,2] = s
    output[...,1,1] = 1
    output[...,2,0] = -s
    output[...,2,2] = c
    return output


def get_3d_box(box_size, heading_angle, center):
    ''' box_size is array(l,w,h), heading_angle is radius clockwise from pos x axis, center is xyz of box center
        output (8,3) array for 3D box cornders
        Similar to utils/compute_orientation_3d
    '''
    R = roty(heading_angle)
    l,w,h = box_size
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
    y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2];
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    # corners_3d = np.vstack([x_corners,y_corners,z_corners])
    corners_3d[0,:] = corners_3d[0,:] + center[0];
    corners_3d[1,:] = corners_3d[1,:] + center[1];
    corners_3d[2,:] = corners_3d[2,:] + center[2];
    corners_3d = np.transpose(corners_3d)
    return corners_3d

def get_3d_box_batch(box_size, heading_angle, center):
    ''' box_size: [x1,x2,...,xn,3]
        heading_angle: [x1,x2,...,xn]
        center: [x1,x2,...,xn,3]
    Return:
        [x1,x3,...,xn,8,3]
    '''
    input_shape = heading_angle.shape
    R = roty_batch(heading_angle)
    l = np.expand_dims(box_size[...,0], -1) # [x1,...,xn,1]
    w = np.expand_dims(box_size[...,1], -1)
    h = np.expand_dims(box_size[...,2], -1)
    corners_3d = np.zeros(tuple(list(input_shape)+[8,3]))
    corners_3d[...,:,0] = np.concatenate((l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2), -1)
    corners_3d[...,:,1] = np.concatenate((h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2), -1)
    corners_3d[...,:,2] = np.concatenate((w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2), -1)
    tlist = [i for i in range(len(input_shape))]
    tlist += [len(input_shape)+1, len(input_shape)]
    corners_3d = np.matmul(corners_3d, np.transpose(R, tuple(tlist)))
    corners_3d += np.expand_dims(center, -2)
    return corners_3d
