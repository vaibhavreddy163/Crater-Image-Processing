import numpy as np
import matplotlib.pyplot as plt

def bilateral_filter(image, spatial_sigma, range_sigma, window_size=5):
    half_window_size = window_size // 2
    filtered_image = np.zeros_like(image, dtype=np.float64)
    padded_image = np.pad(image, ((half_window_size, half_window_size), (half_window_size, half_window_size)), 'reflect')
    x, y = np.mgrid[-half_window_size:half_window_size+1, -half_window_size:half_window_size+1]
    spatial_gaussian = np.exp(-(x**2 + y**2) / (2 * spatial_sigma**2))
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            local_region = padded_image[i:i+window_size, j:j+window_size]
            range_gaussian = np.exp(-((local_region - padded_image[i+half_window_size, j+half_window_size])**2) / (2 * range_sigma**2))
            combined_filter = spatial_gaussian * range_gaussian
            normalization = np.sum(combined_filter)
            filtered_image[i, j] = np.sum(combined_filter * local_region) / normalization
            
    return filtered_image

image_size = 100
image = np.ones((image_size, image_size)) * 128
image[40:60, 40:60] = 255
noise = np.random.normal(0, 20, (image_size, image_size))
image_with_noise = image + noise

filtered_image = bilateral_filter(image_with_noise, spatial_sigma=1.0, range_sigma=30.0, window_size=5)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(image_with_noise, cmap='gray')
axes[1].set_title('Noisy Image')
axes[1].axis('off')

axes[2].imshow(filtered_image, cmap='gray')
axes[2].set_title('Filtered Image')
axes[2].axis('off')

plt.show()
