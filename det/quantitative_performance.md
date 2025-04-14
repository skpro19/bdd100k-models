# Quantitative Performance of Faster R-CNN R50-FPN on BDD100K

This document provides a detailed analysis of the quantitative performance of the Faster R-CNN R50-FPN 1x model on the BDD100K dataset. The analysis is based on inference results from a sample of 100 validation images.

## 1. Detection Performance Overview

| Metric | Value |
|--------|-------|
| Number of Images Evaluated | 100 |
| Total Detections (score ≥ 0.3) | 1,596 |
| Average Detections per Image | 15.96 |
| Median Detections per Image | 15.00 |
| Maximum Detections per Image | 45 |

## 2. Class Distribution

The model detected instances from 9 out of the 10 target classes in the BDD100K dataset. The distribution of detections across classes is as follows:

| Class | Count | Percentage | Average Confidence |
|-------|-------|------------|-------------------|
| car | 976 | 61.2% | 0.8029 |
| traffic sign | 291 | 18.2% | 0.7026 |
| pedestrian | 159 | 10.0% | 0.7237 |
| traffic light | 100 | 6.3% | 0.7002 |
| truck | 46 | 2.9% | 0.7053 |
| bus | 18 | 1.1% | 0.7726 |
| bicycle | 2 | 0.1% | 0.5207 |
| motorcycle | 2 | 0.1% | 0.7418 |
| rider | 2 | 0.1% | 0.6792 |
| train | 0 | 0.0% | N/A |

![Class Distribution](./inference_output/faster_rcnn_r50_fpn_1x_sample/analysis_visualizations/class_distribution.png)

## 3. Confidence Score Analysis

The average confidence score across all detections is high, with most classes having an average confidence above 0.7. The highest average confidence is for the 'car' class (0.8029), while the lowest is for the 'bicycle' class (0.5207).

![Average Confidence by Class](./inference_output/faster_rcnn_r50_fpn_1x_sample/analysis_visualizations/avg_score_by_class.png)

The distribution of confidence scores shows that most detections have high confidence (0.7 and above).

![Confidence Score Distribution](./inference_output/faster_rcnn_r50_fpn_1x_sample/analysis_visualizations/score_distribution.png)

## 4. Relationship to Class Imbalance

The detection distribution largely reflects the class imbalance observed in the data analysis:

1. **Dominant Classes**: 'Car' is the most frequently detected class (61.2% of detections), aligning with its dominance in the training dataset.
2. **Medium-Frequency Classes**: 'Traffic sign', 'pedestrian', and 'traffic light' form the next tier of detection frequency, matching their relative prominence in the dataset.
3. **Low-Frequency Classes**: 'Bicycle', 'motorcycle', and 'rider' have very few detections, corresponding to their rarity in the dataset.
4. **Missing Classes**: The 'train' class has no detections in the sample, likely due to its extreme rarity in the dataset.

This pattern indicates that the model's detection capabilities are influenced by the class distribution in the training data.

## 5. Performance Metrics Limitations

It's important to note some limitations of this analysis:

1. **No Precision/Recall Metrics**: Without a comparison to ground truth annotations, we cannot compute precision, recall, or mAP values. These metrics would provide a more comprehensive evaluation of detection performance.

2. **Sample Size**: The analysis is based on a sample of 100 validation images, which may not fully represent the entire validation set.

3. **Confidence Threshold**: All metrics are computed using a confidence threshold of 0.3. Different thresholds would yield different results.

4. **No Error Analysis**: Without ground truth, we cannot analyze false positives or false negatives to understand where the model fails.

## 6. Confidence in the Results

The high average confidence scores across most classes suggest that the model is generally confident in its predictions. However, confidence scores alone do not guarantee accuracy. The low detection counts for rare classes (bicycle, motorcycle, rider) indicate that the model may struggle with these classes, likely due to limited training examples.

## 7. Summary

The Faster R-CNN R50-FPN 1x model demonstrates strong detection performance on common objects in the BDD100K dataset, particularly cars and traffic elements. The model's performance across classes appears to correlate with the class distribution in the training data, suggesting that class imbalance has a significant impact on detection capabilities.

For a more comprehensive evaluation, future work should include:
- Computing precision/recall curves and mAP metrics using ground truth annotations
- Analyzing detection performance across different environmental conditions (weather, time of day, scene type)
- Investigating failure cases to identify patterns and potential improvements 