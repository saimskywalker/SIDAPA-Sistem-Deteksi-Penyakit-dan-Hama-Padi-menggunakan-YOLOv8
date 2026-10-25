# SIDAPA-Sistem Deteksi Penyakit dan Hama-Padi
This repository contains experiments on **object detection using YOLOv8** to detect diseases and pests affecting rice plants from rice leaf images.

The experiments use a rice plant image dataset collected from rice fields in **Sigi Regency, Central Sulawesi, Indonesia**. Two model configurations with different numbers of classes were developed to explore the effect of class complexity on detection performance.

## Model Configurations

### Model V1 — 5 Classes

Model V1 detects five classes:

* Brown Spot
* Blast
* Rice Leaf Folder
* Bacterial Leaf Blight
* Healthy

### Model V2 — 4 Classes

Model V2 uses four classes:

* Blast
* Rice Leaf Folder
* Bacterial Leaf Blight
* Healthy

The two configurations are **not intended as a competition to determine which model is the best**. Instead, they represent experimental variations that provide a basis for further research, particularly in addressing detection performance when dealing with a larger number of classes or visually similar categories.

## Dataset

The dataset consists of:

* **2,462 images**
* **4,182 bounding boxes**
* Annotation format: YOLO
* Annotation tool: Roboflow
* Dataset split: **80% training, 10% validation, 10% testing**

Dataset distribution:

| Dataset    | Number of Images |
| ---------- | ---------------: |
| Training   |            1,970 |
| Validation |              246 |
| Testing    |              246 |
| Total      |            2,462 |

## Experimental Results

Model performance was evaluated using precision, recall, mAP@50, mAP@50–95, and F1-score.

| Model | Classes | Precision | Recall | mAP@50 | mAP@50–95 | F1-Score |
| ----- | ------: | --------: | -----: | -----: | --------: | -------: |
| V1    |       5 |     89.2% |  76.1% |  84.7% |     63.2% |      82% |
| V2    |       4 |     93.3% |  80.8% |  88.7% |     71.4% |      86% |

These numbers come from the automatic validation of `best.pt` that Ultralytics runs at the end of training, on the **validation split** (246 images). F1-score is computed from the precision and recall above. The validation split was also used for early stopping, so a separate evaluation on the test split is still needed for a final, unbiased score.

### Per-Class Results (Validation Split)

Model V1 (5 classes, best epoch 154):

| Class (label in `data.yaml`)          | Images | Instances | Precision | Recall | mAP@50 | mAP@50-95 |
| ------------------------------------- | -----: | --------: | --------: | -----: | -----: | --------: |
| Brown Spot (`bercak_cokelat`)         |     44 |        82 |     83.4% |  43.9% |  61.8% |     31.2% |
| Blast (`blas`)                        |     32 |        85 |     87.1% |  71.6% |  83.9% |     47.5% |
| Rice Leaf Folder (`hama_putih_palsu`) |     57 |       102 |     90.5% |  87.3% |  91.9% |     67.0% |
| Bacterial Leaf Blight (`hawar_daun_bakteri`) | 83 |      90 |     86.2% |  78.9% |  86.5% |     72.1% |
| Healthy (`sehat`)                     |     75 |        75 |     98.9% |  98.7% |  99.5% |     98.3% |
| **All**                               |    246 |       434 |     89.2% |  76.1% |  84.7% |     63.2% |

Model V2 (4 classes, best epoch 142):

| Class (label in `data.yaml`)          | Images | Instances | Precision | Recall | mAP@50 | mAP@50-95 |
| ------------------------------------- | -----: | --------: | --------: | -----: | -----: | --------: |
| Blast (`blas`)                        |     32 |        85 |     88.8% |  61.2% |  75.0% |     43.2% |
| Rice Leaf Folder (`hama_putih_palsu`) |     57 |       102 |     96.7% |  85.6% |  92.3% |     69.4% |
| Bacterial Leaf Blight (`hawar_daun_bakteri`) | 83 |      90 |     88.6% |  77.8% |  88.1% |     74.3% |
| Healthy (`sehat`)                     |     75 |        75 |     99.3% |  98.7% |  99.5% |     98.8% |
| **All**                               |    246 |       352 |     93.3% |  80.8% |  88.7% |     71.4% |

Brown Spot (V1) and Blast (both models) have the lowest recall, so most missed detections come from these two classes.

## Repository Contents

| File | Description |
| ---- | ----------- |
| `Model_Deteksi_Padi_Lima_Kelas.ipynb` | Training, evaluation, and inference notebook for Model V1 (5 classes) |
| `Model_Deteksi_Padi_Empat_Kelas.ipynb` | Training, evaluation, and inference notebook for Model V2 (4 classes) |
| `results.csv` | Per-epoch training log (losses, metrics, learning rate) from the Model V2 (4 classes) run |

## Training Setup

Both models were trained in Google Colab with the same settings:

| Item | Value |
| ---- | ----- |
| GPU | NVIDIA L4 |
| Environment | Python 3.12.13, PyTorch 2.10.0 (CUDA 12.8), Ultralytics 8.4.39 |
| Base model | `yolov8m.pt` (COCO pretrained) |
| Image size / batch | 640 / 64 |
| Epochs | 300 max, early stopping with patience 50 (V1 stopped at epoch 204, V2 at epoch 192) |
| Optimizer | AdamW, `lr0=0.001`, `lrf=0.01`, cosine LR schedule |
| Augmentation | mosaic 1.0, mixup 0.1, rotation 5 degrees, HSV saturation 0.5 / value 0.4, horizontal flip 0.5 |

Model V2 uses the same dataset as V1 and is trained with `classes=[1, 2, 3, 4]`, so Brown Spot labels are ignored during training.

## Methodology

The general experimental workflow consists of:

1. Collecting rice plant images.
2. Annotating objects using Roboflow.
3. Splitting the dataset into training, validation, and testing sets.
4. Preparing the dataset in YOLO format.
5. Training the YOLOv8 model.
6. Evaluating performance using precision, recall, mAP, and F1-score.
7. Conducting experiments with different class configurations.
8. Analyzing detection results as a basis for further development.

## Technologies

* Python
* YOLOv8
* Ultralytics
* OpenCV
* NumPy
* Roboflow
* Google Colab
