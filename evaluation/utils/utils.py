import numpy as np
import random
import open3d as o3d
import pickle


def boxes3d_to_ply(corners, output_path):
    """
    Convert 3D cube corners to a PLY file with random colors.

    Args:
        corners: numpy array of shape [N,8,3], where N is the number of cubes,
                 and each cube has 8 corner 3D coordinates.
        output_path: Path to save the output PLY file.
    """
    # Pre-allocate arrays for vertices, faces, and vertex colors
    vertices = np.zeros((len(corners) * 8, 3), dtype=np.float32)
    vertex_colors = np.zeros((len(corners) * 8, 3), dtype=np.float32)
    
    # Face template for a cube: 12 triangle faces, defined by corner indices
    # Order: bottom, top, front, right, back, left (2 triangles per face)
    face_template = np.array([
        [0, 1, 2], [0, 2, 3],      # bottom face
        [4, 5, 6], [4, 6, 7],      # top face
        [0, 1, 5], [0, 5, 4],      # front face
        [1, 2, 6], [1, 6, 5],      # right face
        [2, 3, 7], [2, 7, 6],      # back face
        [3, 0, 4], [3, 4, 7]       # left face
    ], dtype=np.int32)
    
    faces = np.zeros((len(corners) * 12, 3), dtype=np.int32)
    
    for i in range(len(corners)):
        # Index range for the current cube's vertices in the global arrays
        start_idx = i * 8
        end_idx = (i + 1) * 8

        # Store the 8 corner coordinates
        vertices[start_idx:end_idx] = corners[i]
        
        # Generate a random RGB color (values in [0, 1]) for all 8 vertices
        color = [random.random(), random.random(), random.random()]
        vertex_colors[start_idx:end_idx] = color
        
        # Generate face indices for this cube, offset by the global start index
        faces[i * 12:(i + 1) * 12] = face_template + start_idx

    # Create an Open3D TriangleMesh object
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    
    # Save to a PLY file using Open3D
    o3d.io.write_triangle_mesh(output_path, mesh)
    print(f"Successfully saved {len(corners)} colored cubes to: {output_path}")
    
    
    
def obb_to_aabb_corners(obb_data):
    """
    Convert OBB data to AABB corner coordinates.

    Args:
        obb_data (np.ndarray): Array of OBB corners, shape [N,8,3].

    Returns:
        np.ndarray: Array of AABB corners, shape [N,8,3].
    """
    # 1. Compute min and max along each axis (X, Y, Z) for every OBB [N, 3]
    min_vals = np.min(obb_data, axis=1)  # Shape: [N, 3]
    max_vals = np.max(obb_data, axis=1)  # Shape: [N, 3]

    # 2. Allocate output array for AABB corners [N, 8, 3]
    corners = np.zeros_like(obb_data)

    for i in range(len(obb_data)):
        # Extract min and max coordinates for current box
        x_min, y_min, z_min = min_vals[i]
        x_max, y_max, z_max = max_vals[i]

        # Generate 8 corners for the AABB in fixed order
        corners[i] = np.array([
            [x_min, y_min, z_min],  # 0: front-left-bottom
            [x_max, y_min, z_min],  # 1: front-right-bottom
            [x_max, y_max, z_min],  # 2: back-right-bottom
            [x_min, y_max, z_min],  # 3: back-left-bottom
            [x_min, y_min, z_max],  # 4: front-left-top
            [x_max, y_min, z_max],  # 5: front-right-top
            [x_max, y_max, z_max],  # 6: back-right-top
            [x_min, y_max, z_max]   # 7: back-left-top
        ])

    return corners


def reorganize_obb_to_aabb(obb_array):
    """
    Reorganize OBB (Oriented Bounding Box) array into AABB (Axis-Aligned Bounding Box) array with a specified corner order.

    Args:
        obb_array: numpy array of shape [N, 8, 3], representing N OBBs with their 8 corner coordinates.

    Returns:
        aabb_array: numpy array of shape [N, 8, 3], where each 8 corners are ordered as:
            0: x_min, y_max, z_min
            1: x_max, y_max, z_min
            2: x_max, y_max, z_max
            3: x_min, y_max, z_max
            4: x_min, y_min, z_min
            5: x_max, y_min, z_min
            6: x_max, y_min, z_max
            7: x_min, y_min, z_max
    """
    # Compute the min and max values along each axis for every OBB
    x_min = np.min(obb_array[:, :, 0], axis=1, keepdims=True)
    x_max = np.max(obb_array[:, :, 0], axis=1, keepdims=True)
    y_min = np.min(obb_array[:, :, 1], axis=1, keepdims=True)
    y_max = np.max(obb_array[:, :, 1], axis=1, keepdims=True)
    z_min = np.min(obb_array[:, :, 2], axis=1, keepdims=True)
    z_max = np.max(obb_array[:, :, 2], axis=1, keepdims=True)
    
    # Create a new array to hold AABB corners
    aabb_array = np.empty_like(obb_array)
    
    # Fill in the corners according to the required order
    # 0: x_min, y_max, z_min
    aabb_array[:, 0, :] = np.concatenate([x_min, y_max, z_min], axis=1)
    # 1: x_max, y_max, z_min
    aabb_array[:, 1, :] = np.concatenate([x_max, y_max, z_min], axis=1)
    # 2: x_max, y_max, z_max
    aabb_array[:, 2, :] = np.concatenate([x_max, y_max, z_max], axis=1)
    # 3: x_min, y_max, z_max
    aabb_array[:, 3, :] = np.concatenate([x_min, y_max, z_max], axis=1)
    # 4: x_min, y_min, z_min
    aabb_array[:, 4, :] = np.concatenate([x_min, y_min, z_min], axis=1)
    # 5: x_max, y_min, z_min
    aabb_array[:, 5, :] = np.concatenate([x_max, y_min, z_min], axis=1)
    # 6: x_max, y_min, z_max
    aabb_array[:, 6, :] = np.concatenate([x_max, y_min, z_max], axis=1)
    # 7: x_min, y_min, z_max
    aabb_array[:, 7, :] = np.concatenate([x_min, y_min, z_max], axis=1)
    
    return aabb_array


# 2. Load data from file
def load_data(filename):
    """Load data from a pickle file.
    
    Args:
        filename: The name of the file to read.
        
    Returns:
        The loaded data (list or object, depending on file content).
    """
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    print(f"Successfully loaded data from {filename}")
    return data



def flip_axis_to_camera(pc):
    ''' Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward
    Input and output are both (N,3) array
    '''
    pc2 = np.copy(pc)
    pc2[...,[0,1,2]] = pc2[...,[0,2,1]] # cam X,Y,Z = depth X,-Z,Y
    pc2[...,1] *= -1
    return pc2

