# Traffic Sign Recognition for Varying Lighting Conditions

## Project Overview

This project addresses the issue of varying lighting conditions affecting Traffic Sign Recognition Models. By using Fuzzy C Means on HSV histograms to classify images into clusters based on lighting. The project automatically classifies and image the runs YOLOv8 object detection on the image using a model trained on similar images.

## Training 
 * Convert RGB Images to HSV Color Space
 * Create 32 feature Histograms Based On The V value
 * Apply PCA on the Histograms to Create 16 Feature Data
 * Use Elbow Method to Determine Number of Clusters
 * Use Fuzzy C Means to Sort Images
 * Train a Model on Each Cluster
## Predict
 *Convert RGB Image to HSV Color Space
 * Create 32 feature Histogram Based On The V value
 * Apply the Trained PCA Model
 * Calculate Which Cluster the Image Belongs to
 * Select YOLO Model
 * Run Inference

## Predict Visual
mermaid```
flowchart TD

A[Load Input Image] --> B[Convert to HSV]
B --> C[Compute V-Channel Histogram]
C --> D[Apply PCA Transform]
D --> E[Compute Fuzzy Membership Probabilities]

E --> F{Max Probability < 0.5?}

F -- Yes --> G[Label = Ambiguous / Use General YOLO Model]

F -- No --> H[Label = Cluster ID]

H --> I{Cluster ID?}

I -- 0 --> J[Select YOLO Model 0]
I -- 1 --> K[Select YOLO Model 1]
I -- 2 --> L[Select YOLO Model 2]
I -- 3 --> M[Select YOLO Model 3]

J --> N[Run Selected YOLO Model]
K --> N
L --> N
M --> N
G --> N
```
## How To Use

Install Dependences
<pre>``` pip install ultralytics open-cv numpy scikit-learn scikit-fuzzy matplotlib``` </pre>
