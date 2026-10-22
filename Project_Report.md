# Computer Vision Classification Project Report

## 1. Introduction

This project compares two computer vision approaches for image classification:

1. Classical machine learning using hand-crafted colour and texture features.
2. Deep learning using MobileNetV2 transfer learning.

The objective of the experiments is to evaluate the performance of different models on the supplied image dataset and compare their classification results.

**Dataset clarification:** The assignment describes avocado ripening-stage classification, but the supplied dataset contains the classes `blackspot`, `canker`, `fresh`, and `grenning`. These labels do not correspond to the requested avocado ripening stages. Therefore, the experiments in this report evaluate the supplied four-class dataset and must not be interpreted as a validated avocado ripeness classifier.

## 2. Dataset Description

The dataset is organised into training and test sets and contains four classes.

| Dataset split | Number of images |
|---|---:|
| Training | 991 |
| Testing | 99 |
| **Total** | **1,090** |

The four classes are:

- `blackspot`
- `canker`
- `fresh`
- `grenning`

### 2.1 Class Distribution

The class distribution is shown below.

| Class | Training images | Test images |
|---|---:|---:|
| blackspot | 184 | 22 |
| canker | 179 | 22 |
| fresh | 281 | 33 |
| grenning | 347 | 22 |
| **Total** | **991** | **99** |

The training dataset is imbalanced because the number of images differs across classes. The `grenning` class has the highest training count, with 347 images, while `canker` has the lowest, with 179 images.

The test set also has unequal class counts: `fresh` contains 33 images, while each of the other classes contains 22 images. This difference should be considered when interpreting overall accuracy and class-wise metrics.

### 2.2 Dataset Quality and Potential Outliers

A file-integrity check was performed on the dataset images.

- Corrupted or unreadable images detected: 0.
- Different image dimensions found: 150.
- Most common image dimension: 800 × 800 pixels.
- Images measuring 800 × 800 pixels: 776.

No corrupted or unreadable images were detected by this check. However, this does not prove that every image is free of visual outliers, incorrect labels, duplicates, or other data-quality issues.

The dataset contains images with different dimensions. Such variation does not automatically indicate outliers, but it may reflect differences in image sources or capture conditions. Further visual inspection would be needed to identify unusual images or possible duplicates.

### 2.3 Potential Dataset Bias

Potential sources of bias include differences in:

- Lighting and illumination.
- Image backgrounds.
- Camera quality and capture conditions.
- Image resolution and viewing angle.
- Number of examples available for each class.

These factors may influence the features learned by the models. The current checks identify class imbalance and image-dimension variation, but they do not establish that any specific source of bias is present. Further investigation using image metadata and visual inspection would be required.

**Important limitation:** The supplied class labels do not represent avocado ripening stages. A suitable avocado dataset with verified ripeness labels is required to evaluate the original assignment objective.

## 3. Approach 1: Classical Computer Vision and Machine Learning

The first approach uses hand-crafted colour and texture features extracted from images. These numerical features are then passed to conventional machine-learning classifiers.

### 3.1 Feature Extraction

**CIELAB colour histograms**

Images are resized to 128 × 128 pixels and converted from BGR colour space to CIELAB. Histograms are extracted from the three colour channels to represent colour distribution.

**Local Binary Patterns (LBP)**

LBP is used to represent local texture patterns in grayscale images. A histogram of LBP values provides a numerical description of texture.

The colour and texture features are concatenated to form the feature vector used by the classifiers.

### 3.2 Support Vector Machine (SVM)

An SVM classifier with an RBF kernel is trained on the extracted features. A standardisation step is included in the SVM pipeline, and class weighting is used to help account for class imbalance.

The SVM is evaluated on the held-out test set using accuracy, a classification report, and a confusion matrix.

**Reported test accuracy:** 95.96% (previously reported; verify against the latest SVM output in the notebook before final submission).

### 3.3 Random Forest

A Random Forest classifier with 100 trees is trained using the same hand-crafted features. Class weighting is enabled, and a fixed random state is used for reproducibility.

The model is evaluated using test accuracy, a classification report, and a confusion matrix.

**Latest reported test accuracy:** 96.97%.

The Random Forest classification report shows strong performance across the four supplied classes. However, performance on a small test set may not fully represent performance on new images.

## 4. Approach 2: MobileNetV2 Transfer Learning

The second approach uses MobileNetV2, a convolutional neural network pretrained on ImageNet.

The model's feature-extraction layers are frozen, and a modified final classification layer is used to predict the four supplied classes.

### 4.1 Training Configuration

- Model: MobileNetV2.
- Input image size: 224 × 224 pixels.
- Training duration: 5 epochs.
- Optimiser: Adam.
- Loss function: Cross-entropy.
- Training augmentation: Random horizontal flips and small rotations.
- Test set size: 99 images.

### 4.2 Evaluation Results

The latest classification report gives an accuracy of approximately 95% on the 99-image test set.

The confusion matrix shows 94 correct predictions out of 99 test images, corresponding to an accuracy of approximately 94.95%.

### 4.3 Classification Report

| Class | Precision | Recall | F1-score | Test images |
|---|---:|---:|---:|---:|
| blackspot | 0.91 | 0.91 | 0.91 | 22 |
| canker | 0.90 | 0.86 | 0.88 | 22 |
| fresh | 0.97 | 1.00 | 0.99 | 33 |
| grenning | 1.00 | 1.00 | 1.00 | 22 |
| **Overall accuracy** |  |  | **94.95%** | **99** |

The model correctly classified all 33 test images in the `fresh` class and all 22 test images in the `grenning` class. The `blackspot` and `canker` classes had some incorrect predictions.

The class-wise results indicate that the model performs differently across classes. The `canker` class has lower recall than `fresh` and `grenning`, suggesting that some canker images were classified as other classes.

## 5. Model Comparison

The following table summarises the reported test results.

| Model | Features or architecture | Test accuracy |
|---|---|---:|
| SVM | CIELAB colour histograms + LBP texture | 95.96%* |
| Random Forest | CIELAB colour histograms + LBP texture | 96.97% |
| MobileNetV2 | Transfer learning | 94.95% |

\*The SVM accuracy is a previously reported result and should be confirmed against the latest notebook output.

Based on the reported results, Random Forest has the highest test accuracy among the three models. MobileNetV2 also performs well, but its reported accuracy is slightly lower.

This comparison does not establish that classical machine learning is generally better than deep learning. Performance can vary with dataset size, class distribution, model configuration, random seed, and image conditions.

The test set contains only 99 images, so the measured differences should be interpreted cautiously.

## 6. Limitations and Potential Bias

The main limitations of this project are:

1. **Dataset mismatch:** The supplied class labels do not correspond to avocado ripening stages.
2. **Small test set:** Evaluation is based on only 99 test images.
3. **Class imbalance:** Training and test image counts differ across classes.
4. **Potential visual bias:** Differences in lighting, backgrounds, image sources, and capture conditions may affect predictions.
5. **Hand-crafted features:** Colour histograms and LBP may not capture every relevant visual characteristic.
6. **Generalisation uncertainty:** High test accuracy does not guarantee reliable performance on images from unseen environments.
7. **Limited deep-learning training:** MobileNetV2 was trained for only five epochs with its feature layers frozen.
8. **Production evaluation:** Real-world latency, memory use, robustness, and deployment performance have not been measured.
9. **Outlier checks:** The integrity check found no unreadable images, but it did not establish the absence of duplicates, incorrect labels, or unusual visual examples.

## 7. Mandatory Discussion Questions

### Question 1: How might the model perform in production?

The reported results indicate strong classification performance on the available test set. However, actual production performance may be lower if new images have different lighting, backgrounds, camera quality, viewing angles, or visual characteristics.

Before deployment, the model should be tested on a larger and more diverse dataset collected under realistic conditions. Performance should also be evaluated using class-wise metrics and representative examples.

Inference latency and resource consumption should be measured on the intended deployment hardware. These measurements have not been recorded in this project.

Because the supplied dataset does not contain verified avocado ripening-stage labels, the current results cannot establish production performance for avocado ripeness assessment.

### Question 2: What are the drawbacks of the approach?

The classical approach relies on hand-crafted colour and texture features. Colour histograms can be sensitive to illumination, and texture features may not capture all structural differences between images. Visually similar classes can therefore be confused.

MobileNetV2 can learn more complex visual representations, but it requires additional computational resources and depends on the quality and representativeness of the training data. Freezing the feature layers may also limit adaptation to the specific dataset.

Both approaches are evaluated on a relatively small test set, and neither has been validated on a suitable avocado ripeness dataset.

### Question 3: What would you improve with more time?

The following improvements would be prioritised:

- Obtain a correctly labelled avocado dataset covering the required ripening stages.
- Inspect the dataset for duplicates, incorrect labels, unusual images, and possible data leakage.
- Use a stratified training, validation, and test split.
- Collect more images across different lighting conditions, backgrounds, cameras, and viewpoints.
- Investigate appropriate image normalisation and data augmentation.
- Tune the SVM and Random Forest hyperparameters using validation data.
- Fine-tune selected MobileNetV2 layers and compare additional lightweight architectures.
- Evaluate precision, recall, F1-score, confusion matrices, and inference latency.
- Test the final model on images from sources not represented in the training dataset.

### Question 4: Where was most of the development time spent?

Complete this answer based on the work actually performed. Relevant activities include:

- Dataset extraction and inspection.
- Image preprocessing and feature engineering.
- Classical model training and evaluation.
- MobileNetV2 training and evaluation.
- Debugging, sample predictions, and report preparation.

The exact activity that consumed the most time should be stated from personal experience rather than assumed or estimated.

## 8. Conclusion

This project compares classical computer vision using CIELAB colour histograms and LBP texture features with MobileNetV2 transfer learning.

The latest reported Random Forest accuracy is 96.97%, while MobileNetV2 achieves approximately 94.95%. The previously reported SVM accuracy is 95.96% and should be verified against the latest notebook output.

The dataset integrity check detected no corrupted or unreadable images. However, class imbalance and image-dimension variation were observed, and additional analysis is needed to assess visual outliers and potential sources of bias.

Most importantly, the supplied dataset labels do not match the avocado ripening stages specified in the assignment. The results therefore demonstrate classification on the supplied four-class dataset, not validated avocado ripeness classification. Confirmation from the assignment provider or a correctly labelled avocado dataset is needed to complete the original objective.

## 9. Reproducibility and Deliverables

The project deliverables include:

- Google Colab notebook containing the experiments and evaluation outputs.
- Saved MobileNetV2 model checkpoint.
- CSV file containing sample image predictions.
- README describing the project and execution steps.
- This report documenting the dataset, methods, results, limitations, and possible improvements.

The notebook should be rerun or checked before submission to confirm that all reported accuracy values match the latest outputs.