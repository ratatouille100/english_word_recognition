import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from pathlib import Path
from sklearn.cluster import DBSCAN
from PIL import Image
import cv2
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
import os
from collections import defaultdict, deque

# Load and preprocess image
image_path = "handwriting.jpg" 
image = Image.open(image_path)
width, height = image.size

# Convert to grayscale and threshold
gray_image = image.convert("L")
gray_array = np.array(gray_image)
_, binary_image = cv2.threshold(gray_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Find dark pixels and cluster
y_coords, x_coords = np.where(binary_image == 0)
coordinates = np.column_stack((x_coords, y_coords))
db = DBSCAN(eps=2, min_samples=1).fit(coordinates)
labels = db.labels_

# Group coordinates by cluster
cluster_coordinates = {}
for label, (x, y) in zip(labels, coordinates):
    if label not in cluster_coordinates:
        cluster_coordinates[label] = []
    cluster_coordinates[label].append((x, y))

# Format clusters
formatted_clusters = [
    [(int(point[0]), int(point[1])) for point in cluster]
    for cluster in cluster_coordinates.values()
]

# Merge connected clusters
point_to_cluster = {}
cluster_points_sets = []
for idx, cluster in enumerate(formatted_clusters):
    s = set(cluster)
    cluster_points_sets.append(s)
    for pt in cluster:
        point_to_cluster[pt] = idx

adjacency = defaultdict(set)
y_threshold = 322

for idx, cluster in enumerate(formatted_clusters):
    for (x, y) in cluster:
        if y < y_threshold:
            for ny in range(y - 1, -1, -1):
                pt = (x, ny)
                if pt in point_to_cluster:
                    other_idx = point_to_cluster[pt]
                    if other_idx != idx:
                        adjacency[idx].add(other_idx)
                        adjacency[other_idx].add(idx)
                    break

visited = set()
merged_clusters_dict = {}
merged_id = 0

for idx in range(len(formatted_clusters)):
    if idx in visited:
        continue

    q = deque([idx])
    merged_indices = set()

    while q:
        current = q.popleft()
        if current in visited:
            continue
        visited.add(current)
        merged_indices.add(current)
        q.extend(adjacency[current] - visited)

    merged_points = []
    for m_idx in merged_indices:
        merged_points.extend(formatted_clusters[m_idx])

    merged_clusters_dict[np.int64(merged_id)] = np.array(merged_points)
    merged_id += 1

# Sort clusters by x-coordinate
sorted_clusters = sorted(merged_clusters_dict.items(), key=lambda item: np.min(item[1][:, 0]))
sorted_clusters_dict = {i: cluster for i, (_, cluster) in enumerate(sorted_clusters)}

# Create resized boxes for each character
resized_boxes = {}
for key, cluster in sorted_clusters_dict.items():
    cluster = np.array(cluster)
    if len(cluster) == 0:
        continue

    xs, ys = cluster[:, 0], cluster[:, 1]
    left, right = xs.min(), xs.max()
    top, bottom = ys.min(), ys.max()
    width = right - left + 1
    height = bottom - top + 1

    if width <= 0 or height <= 0:
        continue

    box_img = np.full((height, width), 255, dtype=np.uint8)
    for x, y in cluster:
        box_x = x - left
        box_y = y - top
        box_img[box_y, box_x] = 0

    image = cv2.cvtColor(box_img, cv2.COLOR_GRAY2RGB)
    target_size = 20
    h, w = image.shape[:2]
    max_dim = max(h, w)
    if max_dim == 0:
        continue

    scale = target_size / max_dim
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    gray_resized = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)
    _, binary_resized = cv2.threshold(gray_resized, 200, 255, cv2.THRESH_BINARY)
    resized_img_bw = cv2.cvtColor(binary_resized, cv2.COLOR_GRAY2RGB)
    canvas = np.full((28, 28, 3), 255, dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img_bw
    inverted_canvas = cv2.bitwise_not(canvas)
    resized_boxes[key] = inverted_canvas

# CNN Model definition
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv2_drop = nn.Dropout2d(p=0.6)
        self.flattened_size = 32 * 4 * 4
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc1_drop = nn.Dropout(p=0.6)
        self.fc2 = nn.Linear(128, 26)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.bn1(self.conv1(x)), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.bn2(self.conv2(x))), kernel_size=2))
        x = x.view(-1, self.flattened_size)
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Load model
model_path = 'models/emnist_cnn_model_nogray.pth'
loaded_model = CNN()
loaded_model.load_state_dict(torch.load(model_path))
loaded_model.eval()

# Character mapping
by_merge_map = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 
    7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 
    14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}

# Predict letters
predicted_string = ""
for key, image in resized_boxes.items():
    # Convert to grayscale and normalize
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image_array = gray_image / 255.0
    image_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        output = loaded_model(image_tensor)
        _, prediction = torch.max(output, 1)
        predicted_letter = by_merge_map.get(prediction.item(), "?")
        predicted_string += predicted_letter

# Output the final predicted word
print(predicted_string.lower().capitalize())