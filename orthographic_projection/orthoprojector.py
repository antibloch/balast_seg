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




def cloud_to_image_with_splats(pcd_np, resolution, splat_size=3, use_post_processing=True):
    """
    Projects a 3D point cloud to a 2D orthographic image using splats with optimized performance.
    
    Args:
        pcd_np: Point cloud numpy array with coordinates and colors
        resolution: Resolution for projection
        splat_size: Size of the splat in pixels (odd number recommended)
        use_post_processing: Whether to apply post-processing to fill gaps
        
    Returns:
        RGB image representation of the point cloud
    """
    minx = np.min(pcd_np[:, 0])
    maxx = np.max(pcd_np[:, 0])
    miny = np.min(pcd_np[:, 1])
    maxy = np.max(pcd_np[:, 1])
    print(f"Point Cloud Bounds: X[{minx}, {maxx}], Y[{miny}, {maxy}]")
    
    # Add padding for splats that might extend beyond the image boundaries
    padding = splat_size // 2
    width = int((maxx - minx) / resolution) + 1 + 2 * padding
    height = int((maxy - miny) / resolution) + 1 + 2 * padding
    print(f"Computed Image Dimensions: Width={width}, Height={height}")
    
    # Initialize image and depth buffer
    image = np.zeros((height, width, 3), dtype=np.uint8)
    # Depth buffer to handle occlusions (use z-coordinate)
    depth = np.ones((height, width)) * float('inf')
    
    # Create a circular splat kernel (precompute once)
    y, x = np.ogrid[-padding:padding+1, -padding:padding+1]
    dist_grid = np.sqrt(x**2 + y**2)
    circular_mask = dist_grid <= padding
    
    # Skip sorting for better performance, just process points 
    # with depth testing at each pixel
    for point in pcd_np:
        x, y, z = point[:3]
        r, g, b = point[3:6]
        
        # Convert 3D coordinates to pixel coordinates
        pixel_x = int((x - minx) / resolution) + padding
        pixel_y = int((maxy - y) / resolution) + padding
        
        # Skip if the center is outside the image bounds
        if not (0 <= pixel_x < width and 0 <= pixel_y < height):
            continue
            
        # Calculate the bounding box for this splat
        x_min = max(0, pixel_x - padding)
        x_max = min(width - 1, pixel_x + padding)
        y_min = max(0, pixel_y - padding)
        y_max = min(height - 1, pixel_y + padding)
        
        # Process the splat region
        for dy in range(y_min - pixel_y + padding, y_max - pixel_y + padding + 1):
            for dx in range(x_min - pixel_x + padding, x_max - pixel_x + padding + 1):
                # Only process pixels within the circular mask
                if circular_mask[dy, dx]:
                    py = pixel_y + dy - padding
                    px = pixel_x + dx - padding
                    
                    # Only update if this point is closer than what's already there
                    if z < depth[py, px]:
                        image[py, px] = [int(b * 255), int(g * 255), int(r * 255)]  # OpenCV uses BGR
                        depth[py, px] = z
    
    # Optional post-processing to fill small gaps
    if use_post_processing:
        # Apply a lighter post-processing for better performance
        # 1. Dilate slightly to fill small gaps
        kernel = np.ones((3, 3), np.uint8)
        mask = np.all(image == [0, 0, 0], axis=2).astype(np.uint8)
        
        # Apply a faster gap-filling approach
        # Dilate the non-empty pixels to fill small gaps
        filled_region = cv2.dilate(1 - mask, kernel, iterations=1)
        # Create a mask of regions to fill (small gaps only)
        fill_mask = np.logical_and(mask, filled_region).astype(np.uint8) * 255
        
        # Quick inpainting only on small gaps
        if np.any(fill_mask > 0):
            image = cv2.inpaint(image, fill_mask, 3, cv2.INPAINT_TELEA)
    
    return image


def get_components(output_seg_np, p = 0.55):

    _,greyscale_otsu = cv2.threshold(output_seg_np,int(p*255),255,cv2.THRESH_BINARY)
    

    binary_image = (greyscale_otsu > 0).astype(np.uint8)  # Convert 255 to 1
    num_labels, labels = cv2.connectedComponents(binary_image)
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    original_labelled = labeled_img.copy()

    uniques= np.unique(labeled_img)

    sizes = []
    for i in range(1, num_labels):  # start from 1 to ignore the background
        size = np.sum(labels == i)
        sizes.append(size)
    idx_max = np.argmax(sizes) + 1  # add 1 because sizes index starts from 0
    max_label = idx_max

    labeled_img[labels != max_label] = 0
    labeled_img[labels == max_label] = 255

    labelled_img = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2GRAY)

    return greyscale_otsu, labelled_img, original_labelled



def get_tubes(ae_image):
    ae_image_grey = cv2.cvtColor(ae_image, cv2.COLOR_RGB2GRAY)

    g_otsu, _, l_img = get_components(ae_image_grey, p = 0.15)
    g_otsu = (g_otsu > 0).astype(np.uint8)
    tube_image = sato(g_otsu, sigmas=range(1, 5, 2), black_ridges=False)

    tube_image_norm = cv2.normalize(tube_image, None, 0, 255, cv2.NORM_MINMAX)

    # Convert to uint8 (OpenCV requires uint8 for thresholding)
    tube_image_uint8 = np.uint8(tube_image_norm)
    
    p = 0.55
    _ , ae_tube = cv2.threshold(tube_image_uint8,int(p*255),255,cv2.THRESH_BINARY)

    return ae_tube



def clahe_thing(orthoimage):
    lab = cv2.cvtColor(orthoimage, cv2.COLOR_BGR2LAB)

    # Split LAB channels
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Apply CLAHE to the L (lightness) channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_channel = clahe.apply(l_channel)

    # Merge back the LAB channels
    lab_clahe = cv2.merge((l_channel, a_channel, b_channel))

    # Convert back to BGR color space
    orthoimage = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    return orthoimage

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



def cloud_to_image(pcd_np, resolution):
    minx = np.min(pcd_np[:, 0])
    maxx = np.max(pcd_np[:, 0])
    miny = np.min(pcd_np[:, 1])
    maxy = np.max(pcd_np[:, 1])
    print(f"Point Cloud Bounds: X[{minx}, {maxx}], Y[{miny}, {maxy}]")
    width = int((maxx - minx) / resolution) + 1
    height = int((maxy - miny) / resolution) + 1
    print(f"Computed Image Dimensions: Width={width}, Height={height}")
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for point in pcd_np:
        x, y, *_ = point
        r, g, b = point[-3:]
        pixel_x = int((x - minx) / resolution)
        pixel_y = int((maxy - y) / resolution)
        # Ensure the pixel coordinates are within image bounds
        if 0 <= pixel_x < width and 0 <= pixel_y < height:
            image[pixel_y, pixel_x] = [int(r * 255), int(g * 255), int(b * 255)]
    return image





def read_data(file_path):
    if file_path.endswith('.ptx'):
        with open(file_path, 'r') as file:
            for _ in range(12):
                file.readline()
            points = []
            colors = []
            for line in file:
                parts = line.strip().split()
                if len(parts) == 7:
                    x, y, z, intensity, r, g, b = map(float, parts)
                    points.append([x, y, z])
                    colors.append([r / 255.0, g / 255.0, b / 255.0])
        points, colors = np.array(points), np.array(colors)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd_np = np.concatenate((points, colors), axis=1)
    elif file_path.endswith('.ply'):
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        pcd_np = np.concatenate((points, colors), axis=1)
    else:
        raise ValueError('Unsupported file format')
    return pcd_np




def fill_black_pixels_robust(image):
    """
    Robustly fill black pixels in an RGB image with colors from the nearest non-black pixels.
    Parameters:
    -----------
    image : numpy.ndarray
        RGB image as numpy array of shape (height, width, 3)
    Returns:
    --------
    numpy.ndarray
        Image with black pixels filled
    """
    # Ensure we're working with a copy and correct data type
    img = image.copy().astype(np.uint8)
    height, width, _ = img.shape
    # Create a mask of non-black pixels (pixels with at least one non-zero channel)
    non_black_mask = np.any(img > 5, axis=2)
    # Check if there are any non-black pixels to use as sources
    if not np.any(non_black_mask):
        print("Warning: Image contains only black pixels, nothing to fill from")
        return img
    # Check if there are no black pixels to fill
    if np.all(non_black_mask):
        print("Note: No black pixels to fill")
        return img
    # Create a filled version of the image
    filled_img = np.zeros_like(img)
    # Use scipy's distance transform to find nearest non-black pixel for each position
    # This returns both the distance and the indices of the nearest non-black pixel
    distances, indices = ndimage.distance_transform_edt(
        ~non_black_mask,
        return_distances=True,
        return_indices=True
    )
    # For each pixel position
    for y in range(height):
        for x in range(width):
            if non_black_mask[y, x]:
                # If this is a non-black pixel, keep its original value
                filled_img[y, x] = img[y, x]
            else:
                # For black pixels, find the closest non-black pixel
                nearest_y = indices[0, y, x]
                nearest_x = indices[1, y, x]
                # Use the color from the nearest non-black pixel
                filled_img[y, x] = img[nearest_y, nearest_x]
    return filled_img




def filler(image):
    ker_size = 3
    kernel = np.ones((ker_size,ker_size), np.uint8)  # Adjust size based on gaps

    # Apply closing operation on each channel separately
    r, g, b = cv2.split(image)

    r_closed = cv2.morphologyEx(r, cv2.MORPH_CLOSE, kernel)
    g_closed = cv2.morphologyEx(g, cv2.MORPH_CLOSE, kernel)
    b_closed = cv2.morphologyEx(b, cv2.MORPH_CLOSE, kernel)

    # Merge the processed channels back
    result = cv2.merge([r_closed, g_closed, b_closed])

    return result






def k_mean(img, K):

    Z = img.reshape((-1,3))
    
    # convert to np.float32
    Z = np.float32(Z)
    
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    return res2




if __name__ == '__main__':
    rotate = True
    resolution = 0.01  # Adjust resolution for better output


    pcd_np = read_data('hmls_01.ply')
    pcd_np_ref = np.copy(pcd_np)
    colors_np = pcd_np[:, 3:6]

    original_pcd = o3d.geometry.PointCloud()
    original_pcd.points = o3d.utility.Vector3dVector(pcd_np[:, :3])
    original_pcd.colors = o3d.utility.Vector3dVector(pcd_np[:, 3:6])

    # Rotating point cloud
    if rotate:
        R_x = original_pcd.get_rotation_matrix_from_xyz((np.pi / 8, 0, 0))  # Rotate 90° around X-axis
        R_y = original_pcd.get_rotation_matrix_from_xyz((0, np.pi / 8, 0))  # Rotate 90° around Y-axis
        R_z = original_pcd.get_rotation_matrix_from_xyz((0, 0, np.pi / 8))  # Rotate 90° around Z-axis
        original_pcd.rotate(R_x, center=(0, 0, 0))  # Apply the X rotation


    print("Displaying original point cloud...")

    plane_model, inliers = original_pcd.segment_plane(distance_threshold=0.2,  # Adjust based on your data scale
                                            ransac_n=3,
                                            num_iterations=1000)
    a, b, c, d = plane_model
    print(f"Detected ground plane: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")
    # Compute the normalization factor for the plane normal
    norm_factor = np.sqrt(a**2 + b**2 + c**2)
    # Convert the point cloud to a NumPy array of points
    pcd_np = np.asarray(original_pcd.points)
    # Compute the perpendicular distance from each point to the ground plane
    distances = np.abs(a * pcd_np[:, 0] + b * pcd_np[:, 1] + c * pcd_np[:, 2] + d) / norm_factor
    # Normalize distances to the range [0, 1] for color mapping]
    distances = np.log1p(np.log1p(np.log1p(np.log1p(distances))))
    distances =np.max(distances) - distances
    min_dist = np.min(distances)
    max_dist = np.max(distances)

    normalized_intensity = (distances - min_dist) / (max_dist - min_dist)

    # element multuply with rgb
    colors_np = colors_np * normalized_intensity[:, None]
    colors = np.clip(colors_np, 0, 1)

    
    # Create a new point cloud that encodes distance from ground as color intensity
    pcd_with_distance = o3d.geometry.PointCloud()
    pcd_with_distance.points = o3d.utility.Vector3dVector(pcd_np)  # Use the same points
    pcd_with_distance.colors = o3d.utility.Vector3dVector(colors)   # Apply our computed colors

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd_with_distance])
    points = np.asarray(pcd_with_distance.points)
    colors = np.asarray(pcd_with_distance.colors)
    pcd_np = np.concatenate((points, colors), axis=1)
    
    # orthoimage = cloud_to_image(pcd_np, resolution)
    orthoimage = cloud_to_image_with_splats(pcd_np, resolution, splat_size=3, use_post_processing=True)
    cv2.imwrite("orthoimage.png", orthoimage)
    plt.imshow(orthoimage)
    plt.title("Cloud to SplatImage")
    plt.show()

    ref_width = orthoimage.shape[1]
    ref_height = orthoimage.shape[0]


    orthoimage = cv2.imread("orthoimage.png")
    orthoimage = cv2.resize(orthoimage, (2024, 2024))

    # orthoimage = filler(orthoimage)
    # plt.imshow(orthoimage)
    # plt.title("Closed Image")
    # plt.show()
    # cv2.imwrite("orthoimage_closed.png", orthoimage)



    # orthoimage = fill_black_pixels_robust(orthoimage)
    # plt.imshow(orthoimage)
    # plt.title("Filled Image")
    # plt.show()
    # cv2.imwrite("ortho_filled.png", orthoimage) 


    # orthoimage=cv2.fastNlMeansDenoisingColored(orthoimage,None,10,10,7,21)
    # plt.imshow(orthoimage)
    # plt.title("Means Denoised Image")
    # plt.show()

    orthoimage = clahe_thing(orthoimage)
    orthoimage_r = np.copy(orthoimage)
    cv2.imwrite("orthoimage_clahe.png", orthoimage)
    plt.imshow(orthoimage)
    plt.title("CLAHE Image")
    plt.show()


    orthoimage = cv2.medianBlur(orthoimage, 3)
    plt.imshow(orthoimage)
    plt.title("Median Blurred Image")
    plt.show()
    cv2.imwrite("orthoimage_median.png", orthoimage)

    img_src = cv2.imread("orthoimage_median.png")
    img_dst = cv2.imread("orthoimage_clahe.png")


    orthoimage = cv2.addWeighted(img_src, 0.5, img_dst, 0.5, 0.0)

    plt.imshow(orthoimage)
    plt.title("Mean Blend")
    plt.show()
    cv2.imwrite("orthoimage_blend.png", orthoimage)
    

    # orthoimage_tube = get_tubes(orthoimage)
    # plt.imshow(orthoimage_tube, cmap='gray')
    # plt.title("Tubes")
    # plt.show()
    

    #============================================================================
    # mask = np.any(orthoimage < 10, axis=2).astype("uint8") * 255
    # plt.imshow(mask, cmap='gray')
    # plt.show()

    # cv2.imwrite("orthoimage_median.png", orthoimage)

    # orthoimage = cv2.inpaint(orthoimage, mask, inpaintRadius=6, flags=cv2.INPAINT_TELEA)

    # plt.imshow(orthoimage)
    # plt.show()

    # cv2.imwrite("orthoimage_inpaint.png", orthoimage)
    #============================================================================


    # orthoimage = cv2.bilateralFilter(orthoimage, d=5, sigmaColor=75, sigmaSpace=75)
    # plt.imshow(orthoimage)
    # plt.title("Bilateral Filtered Image")

    # plt.show()
    # cv2.imwrite("ortho_bilateral.png", orthoimage)




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
