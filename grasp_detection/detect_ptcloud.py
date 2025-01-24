import os
import argparse
import torch
import numpy as np

# Add compatibility for older np.float usage
if not hasattr(np, 'float'):
    np.float = float  # or np.float64

import open3d as o3d
from PIL import Image

from gsnet import AnyGrasp
from graspnetAPI import GraspGroup
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--max_gripper_width', type=float, default=0.1, help='Maximum gripper width (<=0.1m)')
parser.add_argument('--gripper_height', type=float, default=0.03, help='Gripper height')
parser.add_argument('--top_down_grasp', action='store_true', help='Output top-down grasps.')
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
# parser.add_argument('--from_point_cloud', action='store_true', help='Load point cloud directly from PLY file')

cfgs = parser.parse_args()
# cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))

# point_cloud_path = "example_data/splat_csv_minibot_v1.ply"
point_cloud_path = "example_data/ex1_cloud.ply"

FROM_POINT_CLOUD = True

def load_point_cloud(point_cloud_path):
    cloud = o3d.io.read_point_cloud(point_cloud_path)
    points = np.asarray(cloud.points).astype(np.float32)
    colors = np.asarray(cloud.colors).astype(np.float32)

    return points, colors, cloud

def load_rgbd_image(color_filename="", depth_filename=""):
    colors = np.array(Image.open(color_filename), dtype=np.float32) / 255.0
    depths = np.array(Image.open(depth_filename))
    # get camera intrinsics
    fx, fy = 927.17, 927.37
    cx, cy = 651.32, 349.62
    scale = 1000.0
    # set workspace to filter output grasps
    xmin, xmax = -0.19, 0.12
    ymin, ymax = 0.02, 0.15
    zmin, zmax = 0.0, 1.0
    lims = [xmin, xmax, ymin, ymax, zmin, zmax]
    # print("\n\n\nb\n\n\n")
    # get point cloud
    xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depths / scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    # set your workspace to crop point cloud
    mask = (points_z > 0) & (points_z < 1)
    points = np.stack([points_x, points_y, points_z], axis=-1)
    points = points[mask].astype(np.float32)
    colors = colors[mask].astype(np.float32)
    # print("\n\n\nc\n\n\n")
    # Create Open3D point cloud object
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)

    return points, colors, cloud



def detect_grasp(point_cloud_path="", color_filename="", depth_filename=""):
    # Initialize and load AnyGrasp model
    anygrasp = AnyGrasp(cfgs)
    anygrasp.load_net()

    # Define workspace limits
    xmin, xmax = -0.19, 0.12
    ymin, ymax = 0.02, 0.15
    zmin, zmax = 0.0, 1.0
    lims = [xmin, xmax, ymin, ymax, zmin, zmax]

    # Load and process point cloud
    if FROM_POINT_CLOUD:
        points, colors, cloud = load_point_cloud(point_cloud_path)
    else:
        points, colors, cloud = load_rgbd_image(color_filename, depth_filename)

    # Get grasp predictions
    gg, _ = anygrasp.get_grasp(points, colors, lims=lims, 
                              apply_object_mask=True, 
                              dense_grasp=False, 
                              collision_detection=True)
    
    # Post-process grasps
    gg = gg.nms().sort_by_score()

    # Get top grasp pose
    grasp_translation = gg.translations[0]  # Get translation of highest scoring grasp
    grasp_rotation = gg.rotation_matrices[0]  # Get rotation matrix of highest scoring grasp

    if cfgs.debug:
        visualize_grasps_3d(gg, cloud)
        
    return grasp_translation, grasp_rotation


def save_point_cloud_npy(cloud, filename=""):
    # Create output directory if it doesn't exist
    output_dir = "visualization_output" 
    os.makedirs(output_dir, exist_ok=True)

    o3d.io.write_point_cloud(os.path.join(output_dir, f"{filename}cloud.ply"), cloud)

def visualize_grasps_3d(gg, cloud):
    """Visualize the top 3 grasps in 3D using Open3D's legacy off-screen rendering
    
    Args:
        gg (GraspGroup): Group of grasps to visualize
        data_dir (str): Directory containing the data
        cloud (o3d.geometry.PointCloud): Point cloud of the scene
    """
    output_dir = "visualization_output"
    os.makedirs(output_dir, exist_ok=True)
    
    trans_mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
    # cloud.transform(trans_mat)
    # for gripper in grippers:
    #     gripper.transform(trans_mat)

    # gg.transform(trans_mat)

    # Save point cloud first
    o3d.io.write_point_cloud(os.path.join(output_dir, "cloud.ply"), cloud)

    # Get top 3 grasps
    top_grasps = gg[:10]  # GraspGroup is already sorted by score
    
    # Create a combined line set for all grippers
    all_vertices = []
    all_lines = []
    all_colors = []
    vertex_offset = 0

    for grasp_idx, grasp in enumerate(top_grasps):
        # Get grasp parameters
        score = grasp.score
        translation = grasp.translation
        width = grasp.width
        height = grasp.height
        depth = grasp.depth
        
        # print("depth: ", depth)
        rotation = grasp.rotation_matrix

        # Create box vertices representing the grasp
        vertices = []
        for x in [-height/2, height/2]:
            for y in [-width/2, width/2]:
                for z in [-depth/2, depth/2]:
                    point = np.array([x, y, z])
                    # Rotate point
                    point = rotation @ point
                    # Translate point
                    point = point + translation
                    vertices.append(point)
        
        all_vertices.extend(vertices)

        # Define lines connecting vertices to form gripper shape
        lines = [
            # Height lines (blue)
            [0, 1], [2, 3], [4, 5], [6, 7],
            # Width lines (red) 
            [0, 2], [1, 3], [4, 6], [5, 7],
            # Depth lines (green)
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]
        
        # Offset lines based on current vertex count
        offset_lines = [[a + vertex_offset, b + vertex_offset] for a, b in lines]
        all_lines.extend(offset_lines)
        
        # Use fixed colors for each line type
        colors = [
            # Height lines (blue)
            [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],
            # Width lines (red)
            [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], 
            # Depth lines (green)
            [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]
        ]
        
        all_colors.extend(colors)
        
        vertex_offset += 8  # Increment offset for next gripper
        
        print(f"Grasp {grasp_idx + 1} score: {score}")

    # Create line set for all grippers
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(all_vertices)
    line_set.lines = o3d.utility.Vector2iVector(all_lines)
    line_set.colors = o3d.utility.Vector3dVector(all_colors)
    
    # Save combined gripper geometry
    o3d.io.write_line_set(os.path.join(output_dir, "top_grippers.ply"), line_set)

    print(f"Visualization data saved to {output_dir}/")
    print("Point cloud and top 3 gripper geometries have been saved as PLY files.")

if __name__ == '__main__':
    detect_grasp('./example_data/ex1_color.png', './example_data/ex1_depth.png')