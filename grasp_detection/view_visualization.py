import open3d as o3d

# Load the point cloud and grippers
# cloud = o3d.io.read_point_cloud("visualization_output/cloud.ply")
cloud = o3d.io.read_point_cloud("example_data/splat_csv_downsampled.ply")
# grippers = o3d.io.read_line_set("visualization_output/top_grippers.ply")

# Create visualizer
vis = o3d.visualization.Visualizer()
vis.create_window()

# Add geometries
vis.add_geometry(cloud)
# vis.add_geometry(grippers)

# Set view control
ctr = vis.get_view_control()
ctr.set_zoom(0.3)
ctr.set_front([0.5, -0.5, -0.5])
ctr.set_up([0, -1, 0])

# Run visualization
vis.run()
vis.destroy_window()