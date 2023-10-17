from google.colab import drive
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Mount Google Drive
drive.mount('/content/drive')

# Specify the path to the image in Google Drive
# Please replace with the exact path where your image is stored
image_path = '/content/drive/MyDrive/my_image.jpg'
image = Image.open(image_path)

# Define Gaussian filter
def gaussian_filter(size=5, sigma=1.0):
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    return g / g.sum()

# Create Gaussian filter
gaussian_kernel = gaussian_filter()

# Convert image to grayscale
image_gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)

# Apply Gaussian filter
blurred_image = cv2.filter2D(image_gray, -1, gaussian_kernel)

# Display images
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(image_gray, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(blurred_image, cmap='gray')
axes[1].set_title('Blurred Image')
axes[1].axis('off')

plt.show()
