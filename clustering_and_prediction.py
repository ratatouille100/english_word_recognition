import numpy as np
import matplotlib.pyplot as plt
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
import shutil

# Load and process image
image_path = "handwriting.jpg" 
image = Image.open(image_path)
gray_image = image.convert("L")
gray_array = np.array(gray_image)
_, binary_image = cv2.threshold(gray_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
y_coords, x_coords = np.where(binary_image == 0)
coordinates = np.column_stack((x_coords, y_coords))
image = np.array(image)

# Cluster characters
db = DBSCAN(eps=2, min_samples=1).fit(coordinates)
labels = db.labels_
cluster_coordinates = {}
for k in set(labels):
    class_member_mask = (labels == k)
    cluster_coordinates[k] = coordinates[class_member_mask]

# Format clusters
formatted_clusters = [
    [(int(point[0]), int(point[1])) for point in cluster]
    for cluster in cluster_coordinates.values()
]

# Calculate average y position
all_points = [pt for cluster in formatted_clusters for pt in cluster]
y_counts = defaultdict(int)
for _, y in all_points:
    y_counts[y] += 1
avg_y = np.average(list(y_counts.keys()), weights=list(y_counts.values()))

# Merge clusters based on y-position
point_to_cluster = {}
cluster_points_sets = []
for idx, cluster in enumerate(formatted_clusters):
    s = set(cluster)
    cluster_points_sets.append(s)
    for pt in cluster:
        point_to_cluster[pt] = idx

adjacency = defaultdict(set)
y_threshold = avg_y

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

# Sort clusters left to right
sorted_clusters = sorted(merged_clusters_dict.items(), key=lambda item: np.min(item[1][:, 0]))
sorted_clusters_dict = {i: cluster for i, (_, cluster) in enumerate(sorted_clusters)}

# Process each character into 28x28 images
resized_boxes = {}
output_dirs = ['output_binarized', 'output_grayscale']
for dir_path in output_dirs:
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)

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
    grayscale_resized = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)
    _, binary_resized = cv2.threshold(grayscale_resized, 200, 255, cv2.THRESH_BINARY)

    canvas_binary = np.full((28, 28), 255, dtype=np.uint8)
    canvas_grayscale = np.full((28, 28), 255, dtype=np.uint8)

    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    
    canvas_binary[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = binary_resized
    canvas_grayscale[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = grayscale_resized

    inverted_binary = cv2.bitwise_not(canvas_binary)
    inverted_grayscale = cv2.bitwise_not(canvas_grayscale)

    cv2.imwrite(f'output_binarized/box_{key}.png', inverted_binary)
    cv2.imwrite(f'output_grayscale/box_{key}.png', inverted_grayscale)
    resized_boxes[key] = inverted_binary

# Load model and predict
DROPOUT = 0.3
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv2_drop = nn.Dropout2d(p=DROPOUT)
        self.flattened_size = 32 * 4 * 4
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc1_drop = nn.Dropout(p=DROPOUT)
        self.fc2 = nn.Linear(128, 26)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.bn1(self.conv1(x)), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.bn2(self.conv2(x))), kernel_size=2))
        x = x.view(-1, self.flattened_size)
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model_path = 'models/emnist_cnn_model.pth'
loaded_model = CNN()
loaded_model.load_state_dict(torch.load(model_path))
loaded_model.eval()

class_names = [chr(ord('A') + i) for i in range(26)]

def predict_letters_from_directory(directory_path, model, class_names):
    image_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.png')]
    image_files.sort()
    
    predicted_string = ""
    predictions = []
    
    for image_file in image_files:
        image_path = os.path.join(directory_path, image_file)
        image = Image.open(image_path).convert('L')
        image = image.resize((28, 28))
        image_array = np.array(image)
        image_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        model.eval()
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, prediction = torch.max(probabilities, dim=1)
            confidence = confidence.item()
            prediction = prediction.item()
        
        predicted_letter = class_names[prediction] if prediction < len(class_names) else "?"
        predicted_string += predicted_letter
        predictions.append((image_path, predicted_letter, confidence))
    
    return predicted_string, predictions

directory_path = "output_grayscale/" 
predicted_string, individual_predictions = predict_letters_from_directory(directory_path, loaded_model, class_names)

print(predicted_string.lower().capitalize())