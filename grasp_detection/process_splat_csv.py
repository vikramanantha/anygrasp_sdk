import pandas as pd
import open3d as o3d
import numpy as np
import os

df = pd.read_csv("example_data/splat.csv")
splat_data = df.to_numpy()

print(df.head())

xyz = splat_data[:, :3]
rgb = splat_data[:, -3:]
opacity = splat_data[:, -4]


threshold = 5.2e-04

# Create histogram of opacity values
# hist, bins = np.histogram(opacity, bins=50, range=(0, 0.001))
# print("\nOpacity histogram:")
# print("Bin edges:", bins)
# print("Counts:", hist)


# mask = opacity > threshold

# xyz = xyz[mask]
# rgb = rgb[mask]

# print(xyz)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(rgb)

pcd_downsampled = pcd

# pcd_downsampled = pcd.voxel_down_sample(0.01)  # bucket into 1cm grid

# Create directory if it doesn't exist
os.makedirs("example_data", exist_ok=True)

# Save with error handling
try:
    o3d.io.write_point_cloud("example_data/splat_csv.ply", pcd_downsampled)
    print("Saved downsampled point cloud to example_data/splat_csv_downsampled.ply")
except Exception as e:
    print(f"Error saving file: {e}")

# Visualize both clouds
# o3d.visualization.draw_geometries([pcd, pcd_downsampled])
