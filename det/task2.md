# Task 2: Object Detection Model

This document details the model selection, justification, and architecture for Task 2 of the project, focusing on object detection using the BDD100K dataset.

## 1. Chosen Model

*   **Model:** Faster R-CNN
*   **Backbone:** ResNet-50 with Feature Pyramid Network (R-50-FPN)
*   **Training Schedule:** 1x (Standard training schedule as defined in the BDD100K model zoo)
*   **Source:** BDD100K Model Zoo (`bdd100k-models/det/`)

This specific configuration (`faster_rcnn_r50_fpn_1x_det_bdd100k`) was confirmed via the `run_inference.sh` script.

## 2. Justification for Model Choice

*   **Proven Performance:** Faster R-CNN is a foundational and widely recognized two-stage object detection architecture known for its strong performance across various benchmarks.
*   **Availability:** Pre-trained weights specifically trained on the BDD100K dataset are readily available in the BDD100K Model Zoo, significantly reducing the need for extensive training from scratch. This allows focus on evaluation and analysis as per the project requirements.
*   **Baseline:** It serves as a solid baseline for comparison if further experiments with different models were to be conducted. The R-50-FPN backbone offers a reasonable trade-off between accuracy and computational cost compared to larger backbones.
*   **Framework Integration:** The model and its configuration are provided within the `mmdetection` framework, which is a popular and well-documented object detection toolbox.

## 3. Model Architecture Explanation

Faster R-CNN, as described in the paper "[Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)" by Ren et al. (NeurIPS 2015), is a two-stage object detection system.

*   **Backbone Network (R-50-FPN):** A standard convolutional neural network (ResNet-50 in this case) extracts feature maps from the input image. The Feature Pyramid Network (FPN) enhances the standard feature extraction pyramid with lateral connections, creating high-level semantic feature maps at all scales. This allows the model to better detect objects of varying sizes.
*   **Region Proposal Network (RPN):** Unlike earlier models that used computationally expensive external methods (like Selective Search) to generate region proposals, Faster R-CNN introduces the RPN.
    *   The RPN is a fully convolutional network that takes the feature maps from the backbone as input.
    *   It efficiently slides a small network over the convolutional feature map and predicts, at each spatial location (anchor), multiple potential object bounding boxes (region proposals) along with an "objectness" score (probability of the region containing *any* object vs. background).
    *   Crucially, the RPN shares convolutional layers with the backbone network, making region proposal generation almost computationally free.
*   **RoI Pooling (or RoI Align):** The features corresponding to the proposed regions (which can be of different sizes) are extracted from the backbone's feature maps and warped into a fixed-size feature map. RoI Align is often used as an improvement over the original RoI Pooling to handle quantization issues more accurately.
*   **Detection Head (Fast R-CNN):** The fixed-size feature maps from RoI Pooling/Align are fed into a final network (typically fully connected layers). This head performs two tasks:
    *   **Classification:** It classifies the object within the proposed region into one of the predefined categories (e.g., car, person, traffic light) plus a background class.
    *   **Bounding Box Regression:** It refines the coordinates of the proposed bounding box to better fit the actual object.

In essence, the RPN tells the Fast R-CNN component *where* to look, and the Fast R-CNN component then classifies and refines the boxes for those locations.

## 4. Code Snippets / Working Notebooks

*   The configuration file used for this model is located at `bdd100k-models/det/configs/det/faster_rcnn_r50_fpn_1x_det_bdd100k.py`.
*   The inference script used is `bdd100k-models/det/run_inference.sh`.
*   The core implementation relies on the `mmdetection` library, which is built upon PyTorch. Specific code for the Faster R-CNN architecture resides within the `mmdetection` source code.

## 5. (Bonus) Data Loader and Training Pipeline

*This section fulfills the bonus requirement of Task 2.*

*(Currently not implemented as per the base task requirement. If the bonus task were undertaken, details about the custom data loader for BDD100K within the mmdetection framework and the training script/configuration used for at least one epoch on a subset would be documented here.)* 