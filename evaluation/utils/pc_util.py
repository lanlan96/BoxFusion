# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Utility functions for processing point clouds.

Author: Charles R. Qi and Or Litany
"""

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Point cloud IO
import numpy as np
try:
    from plyfile import PlyData, PlyElement
except:
    print("Please install the module 'plyfile' for PLY i/o, e.g.")
    print("pip install plyfile")
    sys.exit(-1)


# Mesh IO
import trimesh

import matplotlib.pyplot as pyplot

# ----------------------------------------
# Point Cloud Sampling
# ----------------------------------------

def random_sampling(pc, num_sample, replace=None, return_choices=False):
    """ Input is NxC, output is num_samplexC
    """
    if replace is None: replace = (pc.shape[0]<num_sample)
    choices = np.random.choice(pc.shape[0], num_sample, replace=replace)
    if return_choices:
        return pc[choices], choices
    else:
        return pc[choices]

# ----------------------------------------
# Point Cloud/Volume Conversions
# ----------------------------------------




def point_cloud_to_volume(points, vsize, radius=1.0):
    """ input is Nx3 points.
        output is vsize*vsize*vsize
        assumes points are in range [-radius, radius]
    """
    vol = np.zeros((vsize,vsize,vsize))
    voxel = 2*radius/float(vsize)
    locations = (points + radius)/voxel
    locations = locations.astype(int)
    vol[locations[:,0],locations[:,1],locations[:,2]] = 1.0
    return vol

def volume_to_point_cloud(vol):
    """ vol is occupancy grid (value = 0 or 1) of size vsize*vsize*vsize
        return Nx3 numpy array.
    """
    vsize = vol.shape[0]
    assert(vol.shape[1] == vsize and vol.shape[1] == vsize)
    points = []
    for a in range(vsize):
        for b in range(vsize):
            for c in range(vsize):
                if vol[a,b,c] == 1:
                    points.append(np.array([a,b,c]))
    if len(points) == 0:
        return np.zeros((0,3))
    points = np.vstack(points)
    return points


# ----------------------------------------
# Point cloud IO
# ----------------------------------------


def write_ply(points, filename, text=True):
    """ input: Nx3, write points to filename as PLY format. """
    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)

def write_ply_color(points, labels, filename, num_classes=None, colormap=pyplot.cm.jet):
    """ Color (N,3) points with labels (N) within range 0 ~ num_classes-1 as OBJ file """
    labels = labels.astype(int)
    N = points.shape[0]
    if num_classes is None:
        num_classes = np.max(labels)+1
    else:
        assert(num_classes>np.max(labels))
    
    vertex = []
    #colors = [pyplot.cm.jet(i/float(num_classes)) for i in range(num_classes)]    
    colors = [colormap(i/float(num_classes)) for i in range(num_classes)]    
    for i in range(N):
        c = colors[labels[i]]
        c = [int(x*255) for x in c]
        vertex.append( (points[i,0],points[i,1],points[i,2],c[0],c[1],c[2]) )
    vertex = np.array(vertex, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4'),('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=True).write(filename)
   
def write_ply_rgb(points, colors, out_filename, num_classes=None):
    """ Color (N,3) points with RGB colors (N,3) within range [0,255] as OBJ file """
    colors = colors.astype(int)
    N = points.shape[0]
    fout = open(out_filename, 'w')
    for i in range(N):
        c = colors[i,:]
        fout.write('v %f %f %f %d %d %d\n' % (points[i,0],points[i,1],points[i,2],c[0],c[1],c[2]))
    fout.close()

# ----------------------------------------
# Simple Point cloud and Volume Renderers
# ----------------------------------------

def pyplot_draw_point_cloud(points, output_filename):
    """ points is a Nx3 numpy array """
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    #savefig(output_filename)

def pyplot_draw_volume(vol, output_filename):
    """ vol is of size vsize*vsize*vsize
        output an image to output_filename
    """
    points = volume_to_point_cloud(vol)
    pyplot_draw_point_cloud(points, output_filename)

# ----------------------------------------
# Simple Point manipulations
# ----------------------------------------
def rotate_point_cloud(points, rotation_matrix=None):
    """ Input: (n,3), Output: (n,3) """
    # Rotate in-place around Z axis.
    if rotation_matrix is None:
        rotation_angle = np.random.uniform() * 2 * np.pi
        sinval, cosval = np.sin(rotation_angle), np.cos(rotation_angle)     
        rotation_matrix = np.array([[cosval, sinval, 0],
                                    [-sinval, cosval, 0],
                                    [0, 0, 1]])
    ctr = points.mean(axis=0)
    rotated_data = np.dot(points-ctr, rotation_matrix) + ctr
    return rotated_data, rotation_matrix

def rotate_pc_along_y(pc, rot_angle):
    ''' Input ps is NxC points with first 3 channels as XYZ
        z is facing forward, x is left ward, y is downward
    '''
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, -sinval],[sinval, cosval]])
    pc[:,[0,2]] = np.dot(pc[:,[0,2]], np.transpose(rotmat))
    return pc

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

def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])


# ----------------------------------------
# BBox
# ----------------------------------------
def bbox_corner_dist_measure(crnr1, crnr2):
    """ compute distance between box corners to replace iou
    Args:
        crnr1, crnr2: Nx3 points of box corners in camera axis (y points down)
        output is a scalar between 0 and 1        
    """
    
    dist = sys.maxsize
    for y in range(4):
        rows = ([(x+y)%4 for x in range(4)] + [4+(x+y)%4 for x in range(4)])
        d_ = np.linalg.norm(crnr2[rows, :] - crnr1, axis=1).sum() / 8.0            
        if d_ < dist:
            dist = d_

    u = sum([np.linalg.norm(x[0,:] - x[6,:]) for x in [crnr1, crnr2]])/2.0

    measure = max(1.0 - dist/u, 0)
    print(measure)
    
    
    return measure


def point_cloud_to_bbox(points):
    """ Extract the axis aligned box from a pcl or batch of pcls
    Args:
        points: Nx3 points or BxNx3
        output is 6 dim: xyz pos of center and 3 lengths        
    """
    which_dim = len(points.shape) - 2 # first dim if a single cloud and second if batch
    mn, mx = points.min(which_dim), points.max(which_dim)
    lengths = mx - mn
    cntr = 0.5*(mn + mx)
    return np.concatenate([cntr, lengths], axis=which_dim)

def write_bbox(scene_bbox, out_filename):
    """Export scene bbox to meshes
    Args:
        scene_bbox: (N x 6 numpy array): xyz pos of center and 3 lengths
        out_filename: (string) filename

    Note:
        To visualize the boxes in MeshLab.
        1. Select the objects (the boxes)
        2. Filters -> Polygon and Quad Mesh -> Turn into Quad-Dominant Mesh
        3. Select Wireframe view.
    """
    def convert_box_to_trimesh_fmt(box):
        ctr = box[:3]
        lengths = box[3:]
        trns = np.eye(4)
        trns[0:3, 3] = ctr
        trns[3,3] = 1.0            
        box_trimesh_fmt = trimesh.creation.box(lengths, trns)
        return box_trimesh_fmt

    scene = trimesh.scene.Scene()
    for box in scene_bbox:
        scene.add_geometry(convert_box_to_trimesh_fmt(box))        
    
    mesh_list = trimesh.util.concatenate(scene.dump())
    # save to ply file    
    trimesh.io.export.export_mesh(mesh_list, out_filename, file_type='ply')
    
    return

def write_oriented_bbox(scene_bbox, out_filename):
    """Export oriented (around Z axis) scene bbox to meshes
    Args:
        scene_bbox: (N x 7 numpy array): xyz pos of center and 3 lengths (dx,dy,dz)
            and heading angle around Z axis.
            Y forward, X right, Z upward. heading angle of positive X is 0,
            heading angle of positive Y is 90 degrees.
        out_filename: (string) filename
    """
    def heading2rotmat(heading_angle):
        pass
        rotmat = np.zeros((3,3))
        rotmat[2,2] = 1
        cosval = np.cos(heading_angle)
        sinval = np.sin(heading_angle)
        rotmat[0:2,0:2] = np.array([[cosval, -sinval],[sinval, cosval]])
        return rotmat

    def convert_oriented_box_to_trimesh_fmt(box):
        ctr = box[:3]
        lengths = box[3:6]
        trns = np.eye(4)
        trns[0:3, 3] = ctr
        trns[3,3] = 1.0            
        trns[0:3,0:3] = heading2rotmat(box[6])
        box_trimesh_fmt = trimesh.creation.box(lengths, trns)
        return box_trimesh_fmt

    scene = trimesh.scene.Scene()
    for box in scene_bbox:
        scene.add_geometry(convert_oriented_box_to_trimesh_fmt(box))        
    
    mesh_list = trimesh.util.concatenate(scene.dump())
    # save to ply file    
    combined_mesh = trimesh.util.concatenate(mesh_list)
    print("combined_mesh",out_filename)
    out_filename = os.path.join('/home/lyq/myprojects/ARM3D/test/',out_filename)
    combined_mesh.export(out_filename)
    # trimesh.io.export.export_mesh(mesh_list, out_filename, file_type='ply')
    
    return

def write_oriented_bbox_camera_coord(scene_bbox, out_filename):
    """Export oriented (around Y axis) scene bbox to meshes
    Args:
        scene_bbox: (N x 7 numpy array): xyz pos of center and 3 lengths (dx,dy,dz)
            and heading angle around Y axis.
            Z forward, X rightward, Y downward. heading angle of positive X is 0,
            heading angle of negative Z is 90 degrees.
        out_filename: (string) filename
    """
    def heading2rotmat(heading_angle):
        pass
        rotmat = np.zeros((3,3))
        rotmat[1,1] = 1
        cosval = np.cos(heading_angle)
        sinval = np.sin(heading_angle)
        rotmat[0,:] = np.array([cosval, 0, sinval])
        rotmat[2,:] = np.array([-sinval, 0, cosval])
        return rotmat

    def convert_oriented_box_to_trimesh_fmt(box):
        ctr = box[:3]
        lengths = box[3:6]
        trns = np.eye(4)
        trns[0:3, 3] = ctr
        trns[3,3] = 1.0            
        trns[0:3,0:3] = heading2rotmat(box[6])
        box_trimesh_fmt = trimesh.creation.box(lengths, trns)
        return box_trimesh_fmt

    scene = trimesh.scene.Scene()
    for box in scene_bbox:
        scene.add_geometry(convert_oriented_box_to_trimesh_fmt(box))        
    
    mesh_list = trimesh.util.concatenate(scene.dump())
    # save to ply file    
    trimesh.io.export.export_mesh(mesh_list, out_filename, file_type='ply')
    
    return

def write_lines_as_cylinders(pcl, filename, rad=0.005, res=64):
    """Create lines represented as cylinders connecting pairs of 3D points
    Args:
        pcl: (N x 2 x 3 numpy array): N pairs of xyz pos             
        filename: (string) filename for the output mesh (ply) file
        rad: radius for the cylinder
        res: number of sections used to create the cylinder
    """
    scene = trimesh.scene.Scene()
    for src,tgt in pcl:
        # compute line
        vec = tgt - src
        M = trimesh.geometry.align_vectors([0,0,1],vec, False)
        vec = tgt - src # compute again since align_vectors modifies vec in-place!
        M[:3,3] = 0.5*src + 0.5*tgt
        height = np.sqrt(np.dot(vec, vec))
        scene.add_geometry(trimesh.creation.cylinder(radius=rad, height=height, sections=res, transform=M))
    mesh_list = trimesh.util.concatenate(scene.dump())
    trimesh.io.export.export_mesh(mesh_list, '%s.ply'%(filename), file_type='ply')

# ----------------------------------------
# Testing
# ----------------------------------------
if __name__ == '__main__':
    print('running some tests')
    
    ############
    ## Test "write_lines_as_cylinders"
    ############
    pcl = np.random.rand(32, 2, 3)
    write_lines_as_cylinders(pcl, 'point_connectors')
    input()
    
   
    scene_bbox = np.zeros((1,7))
    scene_bbox[0,3:6] = np.array([1,2,3]) # dx,dy,dz
    scene_bbox[0,6] = np.pi/4 # 45 degrees 
    write_oriented_bbox(scene_bbox, 'single_obb_45degree.ply')
    ############
    ## Test point_cloud_to_bbox 
    ############
    pcl = np.random.rand(32, 16, 3)
    pcl_bbox = point_cloud_to_bbox(pcl)
    assert pcl_bbox.shape == (32, 6)
    
    pcl = np.random.rand(16, 3)
    pcl_bbox = point_cloud_to_bbox(pcl)    
    assert pcl_bbox.shape == (6,)
    
    ############
    ## Test corner distance
    ############
    crnr1 = np.array([[2.59038660e+00, 8.96107932e-01, 4.73305349e+00],
 [4.12281644e-01, 8.96107932e-01, 4.48046631e+00],
 [2.97129656e-01, 8.96107932e-01, 5.47344275e+00],
 [2.47523462e+00, 8.96107932e-01, 5.72602993e+00],
 [2.59038660e+00, 4.41155793e-03, 4.73305349e+00],
 [4.12281644e-01, 4.41155793e-03, 4.48046631e+00],
 [2.97129656e-01, 4.41155793e-03, 5.47344275e+00],
 [2.47523462e+00, 4.41155793e-03, 5.72602993e+00]])
    crnr2 = crnr1

    print(bbox_corner_dist_measure(crnr1, crnr2))
    
    
    
    print('tests PASSED')
    
    

def read_mesh(filename):
    """ read XYZ for each vertex.
    """
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 7], dtype=np.float32)
        print(plydata['vertex'].data['x'].shape)
        vertices[:, 0] = plydata['vertex'].data['x']
        vertices[:, 1] = plydata['vertex'].data['y']
        vertices[:, 2] = plydata['vertex'].data['z']
        vertices[:, 3] = plydata['vertex'].data['red']
        vertices[:, 4] = plydata['vertex'].data['green']
        vertices[:, 5] = plydata['vertex'].data['blue']
        vertices[:, 6] = plydata['vertex'].data['alpha']

    return vertices, plydata['face']

def write_mesh(vertices, faces, outname):
    new_vertices = []
    for i in range(vertices.shape[0]):
        new_vertices.append(
            (
                vertices[i][0],
                vertices[i][1],
                vertices[i][2],
                vertices[i][3],
                vertices[i][4],
                vertices[i][5],
                vertices[i][6],
            )
        )

    vertices = np.array(
        new_vertices,
        dtype=[
            ("x", np.dtype("float32")),
            ("y", np.dtype("float32")),
            ("z", np.dtype("float32")),
            ("red", np.dtype("uint8")),
            ("green", np.dtype("uint8")),
            ("blue", np.dtype("uint8")),
            ("alpha", np.dtype("uint8"))
        ]
    )

    vertices = PlyElement.describe(vertices, "vertex")
    mesh = PlyData([vertices, faces])
    mesh.write(outname)

def export_aligned_mesh(filename, meta_file, out_name):
    # 1. read pc
    # filename = os.path.join(ROOT, "scene0000_00_vh_clean_2.ply")
    vertices, faces = read_mesh(filename)

    # 2. 
    # meta_file = os.path.join(ROOT, "scene0000_00.txt")
    lines = open(meta_file).readlines()
    axis_align_matrix = None
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]

    # 3. 
    if axis_align_matrix != None:
        axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
        pts = np.ones((vertices.shape[0], 4))
        pts[:, 0:3] = vertices[:, :3]
        pts = np.dot(pts, axis_align_matrix.transpose())
        aligned_vertices = np.copy(vertices)
        aligned_vertices[:, 0:3] = pts[:, 0:3]

    write_mesh(aligned_vertices, faces, out_name)
