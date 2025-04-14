"""Dataset settings."""

dataset_type = "BDD100KDetDataset"  # pylint: disable=invalid-name
# data_root = "../data/bdd100k/"  # OLD: Relative to config file
# data_root = "../../../../" # REMOVE: Relative path calculation seems problematic for pycocotools

# Define paths relative to project root (/media/skumar/External/bdd100k/)
# Assumes config loader resolves these correctly -> Use Absolute Paths Instead
ann_file_val = '/media/skumar/External/bdd100k/data/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json'
img_prefix_val = '/media/skumar/External/bdd100k/data/bdd100k_images_100k/bdd100k/images/100k/val/'
# Define train paths similarly if needed later -> Define now
ann_file_train = '/media/skumar/External/bdd100k/data/bdd100k/jsons/det_train_cocofmt.json' # Assuming path/format based on old config
img_prefix_train = '/media/skumar/External/bdd100k/data/bdd100k/images/100k/train/' # Assuming path based on old config

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", img_scale=(1280, 720), keep_ratio=True),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1280, 720), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

# New MMDetection 3.x dataloader structure
train_dataloader = dict(
    batch_size=4, # samples_per_gpu equivalent
    num_workers=4, # workers_per_gpu equivalent
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(img_path=img_prefix_train),
        pipeline=train_pipeline
        # Add filter_cfg if needed based on full config
        # filter_cfg=dict(filter_empty_gt=True, min_size=32)
    )
)

val_dataloader = dict(
    batch_size=1, # Typically 1 for val/test
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(img_path=img_prefix_val),
        pipeline=test_pipeline,
        test_mode=True # Required for val/test dataset in MMDetection 3.x
    )
)

test_dataloader = val_dataloader # Test often uses same config as val

# New MMDetection 3.x evaluator structure
val_evaluator = dict(
    type='CocoMetric',
    ann_file=ann_file_val,
    metric='bbox',
    format_only=False, # Set based on whether evaluation or just formatting is needed
    # outfile_prefix='./work_dirs/bdd100k_det/val' # Optional: Specify output prefix
)

test_evaluator = val_evaluator # Test often uses same evaluator config
