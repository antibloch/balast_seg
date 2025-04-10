import open3d as o3d
import laspy
import numpy as np


def convert_ply_2_laz(path):
    """
    Convert a .ply file to .laz format using Open3D.
    
    Args:
        path (str): Path to the .ply file.
        
    Returns:
        None
    """
    # Load the point cloud from the .ply file
    # Extract points and optionally colors from the point cloud
    pcd = o3d.io.read_point_cloud(path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None

    # Create a LAS file and write points
    laz_path = path.replace('.ply', '.laz')
    header = laspy.LasHeader(point_format=3, version="1.2")
    las = laspy.LasData(header)
    las.x, las.y, las.z = points[:, 0], points[:, 1], points[:, 2]
    if colors is not None:
        las.red, las.green, las.blue = (colors[:, 0] * 65535).astype(np.uint16), \
                                       (colors[:, 1] * 65535).astype(np.uint16), \
                                       (colors[:, 2] * 65535).astype(np.uint16)
    # Ensure a LazBackend is selected for compression
    from laspy import LazBackend
    las.write(laz_path, do_compress=True, laz_backend=LazBackend.Lazrs)
    # Save the point cloud in .laz format
    o3d.io.write_point_cloud(path.replace('.ply', '.laz'), pcd, write_ascii=False)



if __name__ == '__main__':
    # Specify the path to the .ply file
    ply_file_path = "final_projected_point_cloud.ply"
    
    # Convert the .ply file to .laz format
    convert_ply_2_laz(ply_file_path)


