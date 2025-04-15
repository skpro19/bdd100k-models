# Task 2: Object Detection Model

This document details the model selection, justification, and architecture for Task 2 of the project, focusing on object detection using the BDD100K dataset.

## 1. Chosen Model

*   **Model:** Faster R-CNN
*   **Backbone:** ResNet-50 with Feature Pyramid Network (R-50-FPN)
*   **Training Schedule:** 1x (Standard training schedule as defined in the BDD100K model zoo)
*   **Source:** BDD100K Model Zoo (`bdd100k-models/det/`)

This specific configuration (`faster_rcnn_r50_fpn_1x_det_bdd100k`) was confirmed via the [`run_inference.sh`](./bdd100k-models/det/run_inference.sh) script.

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

*   The configuration file used for this model is located at [`bdd100k-models/det/configs/det/faster_rcnn_r50_fpn_1x_det_bdd100k.py`](./bdd100k-models/det/configs/det/faster_rcnn_r50_fpn_1x_det_bdd100k.py).
*   The inference script used is [`bdd100k-models/det/run_inference.sh`](./bdd100k-models/det/run_inference.sh).
*   The core implementation relies on the `mmdetection` library, which is built upon PyTorch. Specific code for the Faster R-CNN architecture resides within the `mmdetection` source code.

## 5. (Bonus) Data Loader and Training Pipeline

*This section fulfills the bonus requirement of Task 2 by demonstrating a data loader and training pipeline for the BDD100K dataset, training a Faster R-CNN model for one epoch on a subset of the data.*

### 5.1 Data Loading

The data loading process utilizes the MMDetection framework and a custom dataset class `BDD100KDetDataset` (defined in [`datasets/bdd100k.py`](./datasets/bdd100k.py), though not shown here for brevity) which is designed to handle the BDD100K annotation format.

The core configuration for the training data loader is defined within the MMDetection configuration file ([`configs/det/minimal_faster_rcnn.py`](./bdd100k-models/det/configs/det/minimal_faster_rcnn.py)):

```python
# From: bdd100k-models/det/configs/det/minimal_faster_rcnn.py

dataset_type = 'BDD100KDetDataset'
data_root = '/media/skumar/External/bdd100k/' # Adjust if necessary

# Pipelines (Simplified)
backend_args = None # For specifying file backend (e.g., ceph, petrel), None for local fs

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1280, 720), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs') # Packs data into the required format
]

# Dataloaders
train_dataloader = dict(
    batch_size=2, # Adjust based on GPU memory
    num_workers=2, # Adjust based on CPU cores
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='data/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json', # USER PROVIDED PATH
        data_prefix=dict(img_path='data/bdd100k_images_100k/bdd100k/images/100k/train/'), # Corrected path
        pipeline=train_pipeline,
        backend_args=backend_args
        # `indices` will be set by train.py for subset training
        ))
```

Key aspects:
*   **`dataset_type`**: Specifies the custom `BDD100KDetDataset`.
*   **`data_root`**, **`ann_file`**, **`data_prefix`**: Define the paths to the BDD100K images and annotations.
*   **`train_pipeline`**: A list of data augmentation and preprocessing steps applied to each image and its annotations (loading, resizing, random flipping, packing into the format expected by the model).
*   **`batch_size`**, **`num_workers`**: Standard PyTorch DataLoader parameters for batching and parallel data loading.
*   **Subset Loading**: The training script ([`train.py`](./bdd100k-models/det/train.py)) modifies this configuration at runtime to load only a subset (the first 200 samples) for the bonus task demonstration. It achieves this by adding a `subset_size=200` parameter to the `dataset` dictionary, which is then handled by the custom `BDD100KDetDataset`.

### 5.2 Training Pipeline

The training process is orchestrated by the [`train.py`](./bdd100k-models/det/train.py) script, which leverages MMDetection's `Runner` class. This script takes a configuration file as input and handles the setup, training loop, logging, and checkpointing.

**Configuration File (`minimal_faster_rcnn.py`):**
This file ([`configs/det/minimal_faster_rcnn.py`](./bdd100k-models/det/configs/det/minimal_faster_rcnn.py)) defines the complete model architecture (Faster R-CNN with R-50-FPN), dataset configurations (as shown above), optimizer (SGD), learning rate schedule (Linear warmup + MultiStep decay), and runtime settings (hooks for logging, checkpointing, etc.).

```python
# Relevant sections from: bdd100k-models/det/configs/det/minimal_faster_rcnn.py

# Model Definition (Structure defined earlier in this document)
model = dict(
    type='FasterRCNN',
    # ... backbone, neck, rpn_head, roi_head ...
    train_cfg=dict(
        # ... rpn and rcnn training settings ...
    ),
    test_cfg=dict(
        # ... rpn and rcnn testing settings ...
    ))

# Optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
)

# Learning Rate Scheduler
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12, # Corresponds to base max_epochs (overridden later)
        by_epoch=True,
        milestones=[8, 11], # Steps to decay LR
        gamma=0.1)
]

# Default settings (Epochs overridden by train.py)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Runtime settings (hooks, logging, etc.)
default_hooks = dict(...)
env_cfg = dict(...)
log_processor = dict(...)
# ... etc ...
```

**Training Script (`train.py`):**
This script ([`train.py`](./bdd100k-models/det/train.py)) performs several key actions for the bonus task:
1.  Parses command-line arguments (config file path, work directory, number of epochs).
2.  Loads the specified configuration file ([`minimal_faster_rcnn.py`](./bdd100k-models/det/configs/det/minimal_faster_rcnn.py)).
3.  **Modifies the configuration:**
    *   Sets `max_epochs` to the value provided via the command line (defaults to 1 in [`run_train_subset.sh`](./bdd100k-models/det/run_train_subset.sh)).
    *   Adds `subset_size=200` to the `train_dataloader.dataset` configuration to limit the training data.
    *   Disables validation (`val_dataloader`, `val_evaluator`, `val_cfg` set to `None`) as we are only training on a small subset for demonstration.
4.  Initializes MMDetection's `Runner` with the modified configuration.
5.  Calls `runner.train()` to start the training loop for the specified number of epochs.

```python
# Snippet from: bdd100k-models/det/train.py (Illustrative)

def main() -> None:
    args = parse_args()
    cfg = Config.fromfile(args.config)
    # ... work_dir setup ...

    # --- Modifications for Bonus Task ---
    target_epochs = args.epochs
    # Set max_epochs in cfg.train_cfg (or cfg.runner)
    # ... (Code to find and set max_epochs) ...
    print(f"Set train_cfg.max_epochs to: {cfg.train_cfg.max_epochs}")

    subset_size = 200
    if 'train_dataloader' in cfg and 'dataset' in cfg.train_dataloader:
        # ... (Code to locate the dataset config, potentially under RepeatDataset) ...
        if original_dataset_cfg is not None:
            original_dataset_cfg['subset_size'] = subset_size
            print(f"Set custom 'subset_size={subset_size}' in the dataset configuration.")
            # ... (Other modifications like lazy_init, removing filter_cfg) ...

    # --- Skip Validation for Subset Training ---
    print("Skipping validation for subset training run by setting val_dataloader/evaluator/cfg to None.")
    cfg.val_dataloader = None
    cfg.val_evaluator = None
    cfg.val_cfg = None
    # ------------------------------------------

    # Build the runner
    runner = Runner.from_cfg(cfg)

    # Start training
    print("\n---> Starting training loop... <---\n")
    runner.train()
```

**Execution Script (`run_train_subset.sh`):**
A simple bash script ([`run_train_subset.sh`](./bdd100k-models/det/run_train_subset.sh)) is used to launch [`train.py`](./bdd100k-models/det/train.py) with the correct arguments:

```bash
#!/bin/bash
# From: bdd100k-models/det/run_train_subset.sh

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR" || exit 1

CONFIG_FILE="./configs/det/minimal_faster_rcnn.py"
WORK_DIR="./work_dirs/minimal_faster_rcnn_subset_1epoch"
mkdir -p $WORK_DIR

echo "Starting subset training (1 epoch) with config: $CONFIG_FILE"
echo "Results will be saved in: $WORK_DIR"

# Runs train.py, which modifies the config for 1 epoch and subset size
python ./train.py ${CONFIG_FILE} --work-dir ${WORK_DIR}

exit_code=$?
# ... (Check exit code) ...
exit $exit_code
```

### 5.3 Training Run (1 Epoch on Subset)

The [`run_train_subset.sh`](./bdd100k-models/det/run_train_subset.sh) script was executed, triggering the training process defined in [`train.py`](./bdd100k-models/det/train.py) using the [`minimal_faster_rcnn.py`](./bdd100k-models/det/configs/det/minimal_faster_rcnn.py) configuration, modified for 1 epoch and 200 training samples. The following logs were captured from the terminal output:

```text
# Terminal Output from: bash ./run_train_subset.sh

Starting subset training (1 epoch) with config: ./configs/det/minimal_faster_rcnn.py
Results will be saved in: ./work_dirs/minimal_faster_rcnn_subset_1epoch
Available model keys: ['SiLU', 'FrozenBN', 'DropBlock', 'ExpMomentumEMA', 'SinePositionalEncoding', 'LearnedPositionalEncoding', 'SinePositionalEncoding3D', 'DynamicConv', 'MSDeformAttnPixelDecoder', 'Linear']...
FasterRCNN is registered in MODELS registry ✓
Original max_epochs found in 'train_cfg': 12
Set train_cfg.max_epochs to: 1
Attempting to limit train dataset to first 200 samples.
Set custom 'subset_size=200' in the dataset configuration.
Set 'lazy_init=True' in the dataset configuration.
filter_cfg not found in dataset config.
Updated model type in config from 'FasterRCNN' to 'FasterRCNN'
Skipping validation for subset training run by setting val_dataloader/evaluator/cfg to None.
04/15 16:49:58 - mmengine - INFO - 
------------------------------------------------------------
System environment:
    sys.platform: linux
    Python: 3.8.10 (default, Mar 18 2025, 20:04:55) [GCC 9.4.0]
    CUDA available: True
    MUSA available: False
    numpy_random_seed: 1979882568
    GPU 0: NVIDIA GeForce RTX 3070 Laptop GPU
    CUDA_HOME: /usr/local/cuda
    NVCC: Cuda compilation tools, release 12.1, V12.1.66
    GCC: x86_64-linux-gnu-gcc (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0
    PyTorch: 2.1.2+cu121
    PyTorch compiling details: PyTorch built with:
  - GCC 9.3
  - C++ Version: 201703
  - Intel(R) oneAPI Math Kernel Library Version 2022.2-Product Build 20220804 for Intel(R) 64 architecture applications
                 - Intel(R) MKL-DNN v3.1.1 (Git Hash 64f6bcbcbab628e96f33a62c3e975f8535a7bde4)
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - LAPACK is enabled (usually provided by MKL)
  - NNPACK is enabled
  - CPU capability usage: AVX512
  - CUDA Runtime 12.1
  - NVCC architecture flags: -gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_60,code=sm_60;-gencode;arch=compute_70,code=sm_70;-gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_80,code=sm_80;-gencode;arch=compute_86,code=sm_86;-gencode;arch=compute_90,code=sm_90       - CuDNN 8.9.2
  - Magma 2.6.1
  - Build settings: BLAS_INFO=mkl, BUILD_TYPE=Release, CUDA_VERSION=12.1, CUDNN_VERSION=8.9.2, CXX_COMPILER=/opt/rh/devtoolset-9/root/usr/bin/c++, CXX_FLAGS= -D_GLIBCXX_USE_CXX11_ABI=0 -fabi-version=11 -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOROCTRACER -DUSE_FBGEMM -DUSE_QNNPACK -DUSE_PYTORCH_QNNPACK -DUSE_XNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=old-style-cast -Wno-invalid-partial-specialization -Wno-unused-private-field -Wno-aligned-allocation-unavailable -Wno-missing-braces -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow, LAPACK_INFO=mkl, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_DISABLE_GPU_ASSERTS=ON, TORCH_VERSION=2.1.2, USE_CUDA=ON, USE_CUDNN=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=ON, USE_MKLDNN=ON, USE_MPI=OFF, USE_NCCL=1, USE_NNPACK=ON, USE_OPENMP=ON, USE_ROCM=OFF,                                                
    TorchVision: 0.16.2+cu121
    OpenCV: 4.11.0
    MMEngine: 0.10.7

Runtime environment:
    cudnn_benchmark: False
    mp_cfg: {'mp_start_method': 'fork', 'opencv_num_threads': 0}
    dist_cfg: {'backend': 'nccl'}
    seed: 1979882568
    Distributed launcher: none
    Distributed training: False
    GPU number: 1
------------------------------------------------------------

# ... (Full configuration dump omitted for brevity, see original logs) ...

04/15 16:49:58 - mmengine - INFO - Distributed training is not used, all SyncBatchNorm (SyncBN) layers in the model will be automatically reverted to BatchNormXd layers if they are used.
04/15 16:49:58 - mmengine - INFO - Hooks will be executed in the following order:
# ... (Hook execution order omitted) ...

---> Starting training loop... <---

[BDD100KDetDataset __init__] Found subset_size: 200
[BDD100KDetDataset] Loading annotations from: /media/skumar/External/bdd100k/data/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json
[BDD100KDetDataset] JSON loading took 11.16 seconds.
[BDD100KDetDataset] Taking the first 200 frames as specified by config.
[BDD100KDetDataset] Processing 200 frames...
[BDD100KDetDataset] Frame processing took 0.00 seconds.
[BDD100KDetDataset] Set self.cat_ids: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
[BDD100KDetDataset] Built cat_img_map (sample): { 0: []... }
04/15 16:50:11 - mmengine - INFO - load model from: torchvision://resnet50
04/15 16:50:11 - mmengine - INFO - Loads checkpoint by torchvision backend from path: torchvision://resnet50
04/15 16:50:11 - mmengine - WARNING - The model and loaded state dict do not match exactly
unexpected key in source state_dict: fc.weight, fc.bias

04/15 16:50:11 - mmengine - WARNING - "FileClient" will be deprecated in future. Please use io functions in https://mmengine.readthedocs.io/en/latest/api/fileio.html#file-io
04/15 16:50:11 - mmengine - WARNING - "HardDiskBackend" is the alias of "LocalBackend" and the former will be deprecated in future.
04/15 16:50:11 - mmengine - INFO - Checkpoints will be saved to /media/skumar/External/bdd100k/bdd100k-models/det/work_dirs/minimal_faster_rcnn_subset_1epoch.
04/15 16:50:26 - mmengine - INFO - Epoch(train) [1][ 50/100]  lr: 1.9820e-03  eta: 0:00:15  time: 0.3082  data_time: 0.0042  memory: 3312  loss: 1.4919  loss_rpn_cls: 0.6004  loss_rpn_bbox: 0.1391  loss_cls: 0.6614  acc: 88.8672  loss_bbox: 0.0910
04/15 16:50:40 - mmengine - INFO - Exp name: minimal_faster_rcnn_20250415_164957
04/15 16:50:40 - mmengine - INFO - Epoch(train) [1][100/100]  lr: 3.9840e-03  eta: 0:00:00  time: 0.2881  data_time: 0.0032  memory: 3312  loss: 0.9037  loss_rpn_cls: 0.2905  loss_rpn_bbox: 0.1427  loss_cls: 0.3276  acc: 92.0898  loss_bbox: 0.1430
04/15 16:50:40 - mmengine - INFO - Saving checkpoint at 1 epochs
Subset training finished successfully.

```

The logs show:
*   Confirmation that the configuration was modified for 1 epoch and a subset size of 200.
*   Successful loading of the dataset subset (200 frames).
*   Loading of pre-trained ResNet-50 backbone weights (with expected warnings about the unused classification head `fc`).
*   Progress updates every 50 iterations (batch size 2, so 100 iterations for 200 samples).
*   Loss values (overall `loss`, RPN classification `loss_rpn_cls`, RPN bounding box regression `loss_rpn_bbox`, final classification `loss_cls`, and final bounding box regression `loss_bbox`) decreasing during the epoch.
*   Learning rate (`lr`) changes according to the schedule.
*   Estimated time remaining (`eta`), time per iteration (`time`), data loading time (`data_time`), and memory usage.
*   Successful completion of 1 epoch and saving of the checkpoint to [`./work_dirs/minimal_faster_rcnn_subset_1epoch/epoch_1.pth`](./bdd100k-models/det/work_dirs/minimal_faster_rcnn_subset_1epoch/epoch_1.pth). 