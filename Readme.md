# Orange Image Classification Using Computer Vision

## Project Overview

This project compares two computer vision approaches for classifying images from the provided dataset:

1. **Classical Machine Learning:** CIELAB colour histograms and Local Binary Pattern (LBP) texture features, with Support Vector Machine (SVM) and Random Forest classifiers.
2. **Transfer Learning:** MobileNetV2 pretrained on ImageNet, with a modified classifier trained on the dataset classes.

**Dataset clarification:** The supplied dataset is named `Orange Dataset.zip`. Its labels are `blackspot`, `canker`, `fresh`, and `grenning`. These do not correspond to the avocado ripening stages described in the assignment. The current results therefore represent classification of the supplied dataset, not verified avocado ripeness classification.

## Dataset

The dataset contains separate training and test folders.

| Split | Number of images |
|---|---:|
| Training | 991 |
| Testing | 99 |
| Total | 1,090 |

Classes:
- blackspot
- canker
- fresh
- grenning

## Methods

### Approach 1: Classical Computer Vision

- Extract CIELAB colour histograms.
- Extract LBP texture features.
- Train SVM and Random Forest classifiers.
- Evaluate predictions on the test dataset.

Reported test accuracy: approximately **95.96%** for both classifiers.

### Approach 2: MobileNetV2 Transfer Learning

- Resize images to 224 × 224 pixels.
- Apply training augmentation, including horizontal flips and small rotations.
- Use ImageNet-pretrained MobileNetV2 features.
- Replace the final classifier with a four-class output layer.
- Train the classifier and evaluate it on the test dataset.

Reported test accuracy: **93.94%**.

## Results

| Model | Test accuracy |
|---|---:|
| SVM with colour and texture features | 95.96% |
| Random Forest with colour and texture features | 95.96% |
| MobileNetV2 transfer learning | 93.94% |

The test set contains 99 images. Results should be interpreted in the context of this dataset and its class distribution.

## Files

- `Avocado_Ripening_Approach2.ipynb` — Google Colab notebook containing the experiments.
- `avocado_mobilenetv2.pth` — saved MobileNetV2 model checkpoint.
- `avocado_image_predictions.csv` — sample image prediction results.

## Requirements

- Python 3
- OpenCV
- NumPy
- pandas
- scikit-learn
- scikit-image
- Matplotlib
- PyTorch
- torchvision
- Pillow

## Running the Project

1. Open the notebook in Google Colab.
2. Make the dataset available in the expected folder structure.
3. Run the notebook cells in order.
4. Review the model evaluation metrics and sample predictions.

## Limitation

The dataset labels do not match the avocado ripening stages listed in the assignment instructions. The dataset identity and intended target classes must be confirmed with the assignment provider before this work is presented as an avocado ripeness classifier.
