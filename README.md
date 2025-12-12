Traffic Sign Recognition for Varying Lighting Conditions

This repository implements a traffic sign recognition pipeline that adapts object detection performance to different lighting conditions. The system uses HSV-based preprocessing, PCA, Fuzzy C-Means clustering, and cluster-specific YOLOv8 models to route each image to the most appropriate detector.

This work is based on the accompanying paper Traffic Sign Recognition for Varying Lighting Conditions Using Unsupervised Clustering and YOLOv8.

Overview

Lighting variation (nighttime, glare, overexposure, deep shadow) significantly reduces traffic-sign detection accuracy.
This project addresses that by:

Converting images to HSV and extracting brightness histograms

Reducing histogram dimensionality with PCA

Using Fuzzy C-Means (FCM) to cluster images into lighting groups

Training a YOLOv8 model for each cluster

Selecting the correct model during inference using cluster probabilities

Method Summary
Preprocessing

RGB → HSV

V-channel histogram (32 bins → reduced to 16 with PCA)

Clustering

Fuzzy C-Means

Four lighting clusters identified (slightly dark, very dark, very bright, slightly bright)

Low-confidence samples use the general model

Due to limited data:

Cluster 1 → fallback to Cluster 0

Cluster 2 → fallback to Cluster 3

Model Training

YOLOv8n models trained for each cluster

General YOLOv8n baseline trained on full dataset

Training parameters: epochs=100, patience=5, mixup=0.1

Inference

The pipeline:

Compute HSV histogram

Apply PCA

Compute FCM probabilities

Select model

Run YOLOv8 detection

Results (Short Summary)

Cluster models generally perform below the general model due to limited data

However, clusters 0 and 3 match the general model for stop sign detection

Example: A bright-lighting image produces a correct stop sign detection using the cluster-based model, while the general model fails

This suggests potential for adaptive TSR systems with larger datasets.

Installation
git clone https://github.com/sbconnelly2000/Traffic-Sign-Recognition-for-Varying-Lighting-Conditions
cd Traffic-Sign-Recognition-for-Varying-Lighting-Conditions
pip install -r requirements.txt

Usage
Train
python training/train_cluster_models.py

Inference
python inference/predict.py --image path/to/image.jpg
