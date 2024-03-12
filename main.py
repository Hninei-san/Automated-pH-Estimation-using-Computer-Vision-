import cv2
import numpy as np
import detection as detect
import matplotlib.pyplot as plt
from prediction import predict_ph_value_with_distance

plt.figure(figsize=(10, 10))
url = './data/Strip13.jpg'

def create_bar(height, width, color):
    bar = np.zeros((height, width, 3), np.uint8)
    bar[:] = color
    red, green, blue = int(color[2]), int(color[1]), int(color[0])
    return bar, (red, green, blue)

img = detect.roi(url)
height, width, _ = np.shape(img)

data = np.reshape(img, (height * width, 3))
data = np.float32(data)

number_clusters = 6
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv2.KMEANS_PP_CENTERS
compactness, labels, centers = cv2.kmeans(data, number_clusters, None, criteria, 10, flags)

bars = []
allbars = []
rgb_values = []

pairwise_diff = np.zeros(len(centers))
for i, center in enumerate(centers):
    diff = np.max(center) - np.min(center)
    pairwise_diff[i] = diff

lowest_indices = np.argpartition(pairwise_diff, 2)[:2]
lowest_indices_set = set(lowest_indices)
filtered_centers = [center for i, center in enumerate(centers) if i not in lowest_indices_set]

for index, row in enumerate(centers):
    bar, rgb = create_bar(200, 200, row)
    allbars.append(bar)
img_allbar = np.hstack(allbars)

for index, row in enumerate(filtered_centers):
    bar, rgb = create_bar(200, 200, row)
    bars.append(bar)
    rgb_values.append(rgb)
img_bar = np.hstack(bars)
rgb_values = [[int(val) for val in reversed(rgb)] for rgb in rgb_values]

predicted_pH_value, nearest_neighbor_distance, predicted_color_sequence = predict_ph_value_with_distance(rgb_values, n_neighbors=3)

predicted_color_bars = [create_bar(200, 200, color) for color in predicted_color_sequence]
predicted_img_bar = np.hstack([bar for bar, _ in predicted_color_bars])



plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(cv2.imread(url), cv2.COLOR_BGR2RGB), cmap='gray')
plt.title("Original Image")
plt.axis('off') 
plt.subplot(2, 3, 2)
plt.imshow(img, cmap='gray')
plt.title("ROI")
plt.axis('off') 
plt.subplot(2, 3, 3)
plt.imshow(img_allbar, cmap='gray')
plt.title("All 6 Dominant color")
plt.axis('off') 
plt.subplot(2, 3, 4)
plt.imshow(img_bar, cmap='gray')
plt.title("Dominant color")
plt.axis('off') 
plt.subplot(2, 3, 5)
plt.imshow(predicted_img_bar, cmap='gray')
plt.title("Predicted color sequence")
plt.axis('off') 
plt.subplot(2, 3, 6)
plt.axis('off')
info_text = "Predicted pH value: {:.2f}\nNearest neighbor distance: {:.2f}".format(predicted_pH_value, nearest_neighbor_distance)
plt.text(0.5, 0.5, info_text, ha='center')

plt.tight_layout()
plt.show()

