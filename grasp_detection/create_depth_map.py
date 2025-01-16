# import numpy as np
# from PIL import Image

# # Create a black image with dimensions 4080x3072
# depth_img = np.zeros((3072, 4080), dtype=np.uint8)

# # Save as PNG
# Image.fromarray(depth_img).save('example_data/lab_view_1_depth.png')


import numpy as np
from PIL import Image

# Read the depth image
depth_img = np.array(Image.open('example_data/ex1_depth.png'))

# Print the numpy array
print("Depth image array (every 30th value):")

# Convert to 1D array and take every 30th element
sampled = depth_img[::30, ::30]
for row in sampled.tolist():
    print(row)

print("\nArray shape:", depth_img.shape)
print("Array dtype:", depth_img.dtype)
