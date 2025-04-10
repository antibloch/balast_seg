import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import torch
import gc
import os
import requests
import cv2
import time
import copy
from scipy import ndimage
from skimage.filters import sato
from skimage import io






def orthogonal_image_to_cloud(orthoimage, pcd_np, resolution):
    """
    Projects an orthogonal RGB image back to the 3D point cloud,
    creating a new point cloud with only the colors from the image.
    Args:
        orthoimage: The processed RGB image
        pcd_np: Original point cloud with coordinates and colors
        resolution: The resolution used when creating the image
    Returns:
        Modified point cloud with new colors from the image (original colors removed)
    """
    # Create a new point cloud with only the coordinates from the original
    # Initialize with zeros for colors
    modified_pcd = np.zeros_like(pcd_np)
    modified_pcd[:, :3] = pcd_np[:, :3]  # Copy only the XYZ coordinates
    ref_pcd = np.copy(modified_pcd)
    # Calculate the bounds of the point cloud
    minx = np.min(pcd_np[:, 0])
    maxx = np.max(pcd_np[:, 0])
    miny = np.min(pcd_np[:, 1])
    maxy = np.max(pcd_np[:, 1])
    # For each point in the point cloud, find its corresponding pixel in the image
    for i, point in enumerate(pcd_np):
        x, y, *_ = point
        # Convert 3D coordinates to pixel coordinates
        pixel_x = int((x - minx) / resolution)
        pixel_y = int((maxy - y) / resolution)  # Flipped to match image coordinate system
        # Get the height and width of the orthoimage
        height, width = orthoimage.shape[:2]
        # Check if the pixel coordinates are within the image bounds
        if 0 <= pixel_x < width and 0 <= pixel_y < height:
            # Get the color from the image
            r, g, b = orthoimage[pixel_y, pixel_x]
            # Update the color in the modified point cloud (scale from 0-255 to 0-1)
            modified_pcd[i, 3:6] = [r/255.0, g/255.0, b/255.0]
        else:
            # Points outside image bounds get zero color
            modified_pcd[i, 3:6] = [0, 0, 0]
    return modified_pcd, ref_pcd





if __name__ == '__main__':
    rotate = True
    resolution = 0.01  # Adjust resolution for better output
    ref_width = np.load("ref_width.npy")
    ref_height = np.load("ref_height.npy")

    ref_width = int(ref_width)
    ref_height = int(ref_height)


    pcd_np = np.load("pcd_np.npy")


    # Project the image back to the point cloud
    orthoimage = cv2.imread("segmented.png")
    if ref_width != orthoimage.shape[1] or ref_height != orthoimage.shape[0]:
        orthoimage = cv2.resize(orthoimage, (ref_width, ref_height))
        
    projected_pcd_np, ref_pcd = orthogonal_image_to_cloud(orthoimage, pcd_np, resolution)

    # Create Open3D point clouds for visualization
    original_pcd = o3d.geometry.PointCloud()
    original_pcd.points = o3d.utility.Vector3dVector(pcd_np[:, :3])
    original_pcd.colors = o3d.utility.Vector3dVector(pcd_np[:, 3:6])

    projected_pcd = o3d.geometry.PointCloud()
    projected_pcd.points = o3d.utility.Vector3dVector(projected_pcd_np[:, :3])
    projected_pcd.colors = o3d.utility.Vector3dVector(projected_pcd_np[:, 3:6])

    before_pcd = o3d.geometry.PointCloud()
    before_pcd.points = o3d.utility.Vector3dVector(ref_pcd[:, :3])
    before_pcd.colors = o3d.utility.Vector3dVector(ref_pcd[:, 3:6])

    # Save the point clouds
    o3d.io.write_point_cloud("original_point_cloud.ply", original_pcd)
    o3d.io.write_point_cloud("projected_point_cloud.ply", projected_pcd)
    o3d.io.write_point_cloud("before_point_cloud.ply", before_pcd)

    # visualize side by side
    o3d.visualization.draw_geometries([projected_pcd])


    # compare the heights 
    plane_model, inliers = projected_pcd.segment_plane(distance_threshold=0.2,  # Adjust based on your data scale
                                            ransac_n=3,
                                            num_iterations=1000)
    a, b, c, d = plane_model
    print(f"Detected ground plane: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")
    # Compute the normalization factor for the plane normal
    norm_factor = np.sqrt(a**2 + b**2 + c**2)
    # Convert the point cloud to a NumPy array of points
    pcd_np = np.asarray(projected_pcd.points)
    pcd_np_ref = np.copy(pcd_np)
    # Compute the perpendicular distance from each point to the ground plane
    distances = np.abs(a * pcd_np[:, 0] + b * pcd_np[:, 1] + c * pcd_np[:, 2] + d) / norm_factor

    min_dist = np.min(distances)
    max_dist = np.max(distances)

    normalized_dist= (distances - min_dist) / (max_dist - min_dist)

    colors_np = np.asarray(projected_pcd.colors)
    colors_np_ref = np.asarray(original_pcd.colors)

    for i in range(len(normalized_dist)):
        if normalized_dist[i] > 0.2:
            colors_np[i] = colors_np_ref[i]



    projected_pcd.colors = o3d.utility.Vector3dVector(colors_np)

    o3d.io.write_point_cloud("final_projected_point_cloud.ply", projected_pcd)
    
    # visualize side by side
    o3d.visualization.draw_geometries([projected_pcd])
