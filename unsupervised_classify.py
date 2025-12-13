import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import skfuzzy as fuzz
import shutil
from pathlib import Path
import joblib





c = 4  
threshold = 0.60  
ambiguous_id = c  #

BINS_V = [32]
n_components_pca = 16 


src_img_pattern = "car/all_images_labels/images/*.jpg"
src_lbl_dir = Path("car/all_images_labels/labels")
dest_base_dir = Path(".") 
image_paths = sorted(glob.glob(src_img_pattern))

print(f"Found {len(image_paths)} images.")

data = []

for path in image_paths:
    img = cv2.imread(path)
    if img is None:
        continue
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    

    hist = cv2.calcHist([hsv], [2], None, BINS_V, [0, 256])
   
    
    data.append(hist.flatten())


X = np.array(data)
print(f"Data shape before PCA: {X.shape}")

pca = PCA(n_components=n_components_pca, random_state=42)
X_pca = pca.fit_transform(X)


X_fuzzy = X_pca.T 


print("Running Fuzzy C-Means...")
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    X_fuzzy, 
    c=c,  
    m=1.1, 
    error=0.005, 
    maxiter=1000, 
    init=None
)
..

max_probs = np.max(u, axis=0)

print(f"Highest confidence found in dataset: {np.max(max_probs):.4f}")
print(f"Average confidence in dataset: {np.mean(max_probs):.4f}")
print(f"Lowest confidence in dataset: {np.min(max_probs):.4f}")


cluster_labels = np.argmax(u, axis=0)


max_probs = np.max(u, axis=0)

cluster_labels[max_probs < threshold] = ambiguous_id


unique, counts = np.unique(cluster_labels, return_counts=True)



labels_names = [f'Cluster {i}' for i in range(c)] + ['Ambiguous']

plt.figure(figsize=(10, 8))
for i in range(c + 1):
    mask = (cluster_labels == i)
    
    if np.any(mask):
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c=colors[i % len(colors)], 
                   label=labels_names[i],
                   alpha=0.6)

plt.title(f"Fuzzy Clusters (Threshold {threshold*100}%)")
plt.legend()
plt.show()


print("Starting file copy operation...")

for i, label in enumerate(cluster_labels):
 
    if label == ambiguous_id:
        folder_name = "cluster_ambiguous"
    else:
        folder_name = f"cluster_{label}"
        
    
    full_img_path = Path(image_paths[i])
    img_filename = full_img_path.name
    
    
    lbl_filename = full_img_path.stem + ".txt"
    src_lbl_path = src_lbl_dir / lbl_filename
    
    
    dst_img_folder = dest_base_dir / folder_name / "images"
    dst_lbl_folder = dest_base_dir / folder_name / "labels"
    
    dst_img_folder.mkdir(parents=True, exist_ok=True)
    dst_lbl_folder.mkdir(parents=True, exist_ok=True)
    
    
    try:
        
        shutil.copy(full_img_path, dst_img_folder / img_filename)
        
        
        if src_lbl_path.exists():
            shutil.copy(src_lbl_path, dst_lbl_folder / lbl_filename)
            
    except Exception as e:
        print(f"Error processing {img_filename}: {e}")



np.save('fcm_centers.npy', cntr)


joblib.dump(pca, 'pca_model.pkl')
