import os
import cv2
import numpy as np
import joblib
import random
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# 设置路径和参数
dataset_dir = 'Aerial_Landscapes'
model_dir = 'models_imbalanced'
os.makedirs(model_dir, exist_ok=True)

categories = ['Agriculture', 'Airport', 'Beach', 'City', 'Desert', 'Forest', 'Grassland',
              'Highway', 'Lake', 'Mountain', 'Parking', 'Port', 'Railway', 'Residential', 'River']

resize_size = (256, 256)
n_clusters = 100
sift = cv2.SIFT_create()
category_limits = {cat: int(640 - i * 40) for i, cat in enumerate(categories)}

kmeans_path = os.path.join(model_dir, 'kmeans_bovw_imbalanced.pkl')
svm_path = os.path.join(model_dir, 'svm_classifier_imbalanced.pkl')
label_encoder_path = os.path.join(model_dir, 'label_encoder_imbalanced.pkl')

# 划分 train/test 图像路径
train_paths, train_labels, test_paths, test_labels = [], [], [], []

for i, label in enumerate(categories):
    folder_path = os.path.join(dataset_dir, label)
    img_files = os.listdir(folder_path)
    random.shuffle(img_files)

    split_idx = int(len(img_files) * 0.8)
    train_list = img_files[:split_idx]
    test_list = img_files[split_idx:]

    limit = category_limits[label]
    selected_train = train_list[:limit]
    train_paths += [os.path.join(folder_path, f) for f in selected_train]
    train_labels += [label] * len(selected_train)

    test_paths += [os.path.join(folder_path, f) for f in test_list]
    test_labels += [label] * len(test_list)

# 只从部分图像中提取描述子用于 KMeans 聚类
MAX_DESCRIPTOR_IMAGES = 1000
sample_paths = random.sample(train_paths, min(MAX_DESCRIPTOR_IMAGES, len(train_paths)))
descriptor_list = []

print("Extracting descriptors from sampled training images for clustering...")
fail_count = 0
for path in tqdm(sample_paths):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[Warning] Failed to load: {path}")
        fail_count += 1
        continue
    img = cv2.resize(img, resize_size)
    _, descriptors = sift.detectAndCompute(img, None)
    if descriptors is not None:
        descriptor_list.extend(descriptors)

print(f"Total descriptors extracted: {len(descriptor_list)}")
print(f"Total failed images: {fail_count}")

descriptor_stack = np.array(descriptor_list)

# 聚类（BoVW）
if os.path.exists(kmeans_path):
    print("Loading existing KMeans model...")
    kmeans = joblib.load(kmeans_path)
else:
    print("Clustering descriptors with MiniBatchKMeans...")
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, verbose=1, batch_size=5000)
    kmeans.fit(descriptor_stack)
    joblib.dump(kmeans, kmeans_path)

# 提取 BoVW 特征函数
def extract_bow_feature(img, kmeans, sift, n_clusters):
    _, descriptors = sift.detectAndCompute(img, None)
    if descriptors is None:
        return np.zeros(n_clusters)
    clusters = kmeans.predict(descriptors)
    hist, _ = np.histogram(clusters, bins=np.arange(n_clusters + 1), density=True)
    return hist

# 提取训练特征
X_train = []
print("Extracting training features...")
for path in tqdm(train_paths):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    img = cv2.resize(img, resize_size)
    feat = extract_bow_feature(img, kmeans, sift, n_clusters)
    X_train.append(feat)

# 提取测试特征
X_test = []
print("Extracting test features...")
for path in tqdm(test_paths):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    img = cv2.resize(img, resize_size)
    feat = extract_bow_feature(img, kmeans, sift, n_clusters)
    X_test.append(feat)

# 标签编码
le = LabelEncoder()
y_train = le.fit_transform(train_labels)
y_test = le.transform(test_labels)
joblib.dump(le, label_encoder_path)

X_train = np.array(X_train)
X_test = np.array(X_test)

# SVM 训练
print("Training SVM classifier...")
clf = SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced')
clf.fit(X_train, y_train)
joblib.dump(clf, svm_path)

# 模型评估
print("Evaluating model...")
y_pred = clf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix (SIFT-SVM)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig("confusion_matrix_imbalanced_sift.png")
plt.show()

# 每类准确率柱状图
class_accuracy = cm.diagonal() / cm.sum(axis=1)
plt.figure(figsize=(12, 5))
sns.barplot(x=le.classes_, y=class_accuracy)
plt.ylabel("Accuracy")
plt.title("Per-Class Accuracy (SIFT-SVM)")
plt.xticks(rotation=45)
plt.ylim(0, 1.0)
plt.tight_layout()
plt.savefig("per_class_accuracy_imbalanced_sift.png")
plt.show()

print("Saved confusion_matrix_imbalanced_sift.png and per_class_accuracy_imbalanced_sift.png.")
