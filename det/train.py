"""Train a detector on a subset for one epoch."""

import argparse
import os
import os.path as osp
import sys

from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmengine.registry import RUNNERS
from mmengine.dist import init_dist # For distributed training setup, if needed

# ---> Add necessary mmdetection imports to register components <---
# Import specific modules to ensure proper registration
import mmdet
import mmdet.registry
from mmdet.registry import MODELS
import mmdet.models
import mmdet.datasets
import mmdet.engine    
import mmdet.structures
import mmdet.visualization

# Import model directly to ensure it's registered
from mmdet.models.detectors import FasterRCNN

# Verify model registration
print(f"Available model keys: {list(MODELS.module_dict.keys())[:10]}...")
if 'FasterRCNN' in MODELS.module_dict:
    print("FasterRCNN is registered in MODELS registry ✓")
else:
    print("WARNING: FasterRCNN is NOT in MODELS registry, attempting manual registration.")
    try:
        # Try to register it manually
        MODELS.register_module()(FasterRCNN)
        print("Manually registered FasterRCNN ✓")
    except Exception as e:
        print(f"Failed to register FasterRCNN: {e}")
        print("Training will likely fail. Please check your MMDetection installation.")
# --------------------------------------------------------------

# Import custom dataset - Ensure it's registered or accessible
# Might not be needed if config handles registration
# from datasets.bdd100k import BDD100KDetDataset
import datasets.bdd100k

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a detector (subset, 1 epoch)')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        action='store_true',
        help='resume from the latest checkpoint in the work_dir')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=1,
        help='Number of epochs to train for (defaults to 1 for subset run)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to TrainTool. Recommend NOT use
    # `--local-rank` when using PyTorch version >= 2.0.0. Ref:
    # https://github.com/open-mmlab/mmdetection/pull/9139
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0) # Accept both variants
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main() -> None:
    """Main training function."""
    args = parse_args()

    # Load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
        
    # Set work directory
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if work_dir is skipped
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0] + "_subset_1epoch")
    
    # --- Modifications for Bonus Task ---
    # Locate where max_epochs is defined (train_cfg or runner)
    max_epochs_location = None
    original_max_epochs = 'Not Set'
    if 'train_cfg' in cfg and cfg.train_cfg is not None and 'max_epochs' in cfg.train_cfg:
        max_epochs_location = 'train_cfg'
        original_max_epochs = cfg.train_cfg.max_epochs
    elif 'runner' in cfg and cfg.runner is not None and 'max_epochs' in cfg.runner:
        max_epochs_location = 'runner'
        original_max_epochs = cfg.runner.max_epochs

    print(f"Original max_epochs found in '{max_epochs_location}': {original_max_epochs}")

    # 1. Set max epochs based on command-line arg
    target_epochs = args.epochs
    if max_epochs_location == 'train_cfg':
        cfg.train_cfg.max_epochs = target_epochs
        print(f"Set train_cfg.max_epochs to: {cfg.train_cfg.max_epochs}")
    elif max_epochs_location == 'runner':
        cfg.runner.max_epochs = target_epochs
        print(f"Set runner.max_epochs to: {cfg.runner.max_epochs}")
    else:
        # If neither found, inject a default train_cfg (less likely with current base files)
        print(f"Warning: Could not find max_epochs in train_cfg or runner. Injecting default train_cfg with {target_epochs} epochs.")
        cfg.train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=target_epochs, val_interval=1) # Assuming default loop type
        print(f"Set train_cfg.max_epochs to: {cfg.train_cfg.max_epochs}")

    # Ensure train_cfg exists for the Runner initialization
    if 'train_cfg' not in cfg or cfg.train_cfg is None:
        print("Injecting train_cfg for Runner.")
        # Try to inherit from runner if it exists and has type
        if 'runner' in cfg and 'type' in cfg.runner and 'Loop' in cfg.runner.type:
            cfg.train_cfg = cfg.runner
            cfg.train_cfg.max_epochs = target_epochs # Ensure it's set correctly
            print(f"Copied runner config to train_cfg. Set max_epochs to {cfg.train_cfg.max_epochs}")
        else:
             # Fallback to a default EpochBasedTrainLoop
             cfg.train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=target_epochs, val_interval=1)
             print(f"Created default train_cfg. Set max_epochs to {cfg.train_cfg.max_epochs}")
    elif 'max_epochs' not in cfg.train_cfg: # If train_cfg exists but missing max_epochs
        cfg.train_cfg.max_epochs = target_epochs
        print(f"Added max_epochs={target_epochs} to existing train_cfg.")

    # Ensure optim_wrapper exists (should be loaded from base config, but check)
    if 'optim_wrapper' not in cfg or cfg.optim_wrapper is None:
        print("Attempting to construct optim_wrapper from optimizer settings...")
        if 'optimizer' in cfg:
            # Basic optim_wrapper structure
            cfg.optim_wrapper = dict(
                type='OptimWrapper', # Default wrapper
                optimizer=cfg.optimizer
            )
            # Handle gradient clipping (previously in optimizer_config)
            if 'optimizer_config' in cfg and cfg.optimizer_config is not None and 'grad_clip' in cfg.optimizer_config:
                 clip_grad_cfg = cfg.optimizer_config.get('grad_clip')
                 if clip_grad_cfg:
                     cfg.optim_wrapper['clip_grad'] = clip_grad_cfg
                     print(f"Added clip_grad settings to optim_wrapper: {clip_grad_cfg}")
                 else:
                     cfg.optim_wrapper['clip_grad'] = None # Explicitly set if None in config
            else:
                 cfg.optim_wrapper['clip_grad'] = None # Default if optimizer_config missing
            
            print(f"Constructed optim_wrapper: {cfg.optim_wrapper}")
            # Remove old keys if they exist to avoid confusion
            # cfg.pop('optimizer', None)
            # cfg.pop('optimizer_config', None)
        else:
            # This is unlikely given the base config, but could happen
            # In a real scenario, we might need to load schedule_1x.py explicitly
            # to get the optimizer settings.
            raise ValueError("optimizer settings not found, cannot construct optim_wrapper.")
            
    # Ensure param_scheduler exists (replaces lr_config)
    if 'param_scheduler' not in cfg or cfg.param_scheduler is None:
        print("Attempting to construct param_scheduler from lr_config...")
        if 'lr_config' in cfg:
            # MMEngine uses param_scheduler (often a list)
            # Need to translate lr_config to param_scheduler format
            # This is a basic translation for the common step scheduler
            lr_step_config = cfg.lr_config
            if lr_step_config.policy == 'step':
                cfg.param_scheduler = [
                    dict(
                        type='LinearLR',
                        start_factor=lr_step_config.get('warmup_ratio', 0.001),
                        by_epoch=False, # Warmup is usually by iter
                        begin=0,
                        end=lr_step_config.get('warmup_iters', 500)),
                    dict(
                        type='MultiStepLR',
                        by_epoch=True,
                        milestones=lr_step_config.get('step', [8, 11]),
                        gamma=0.1)
                ]
                print(f"Constructed param_scheduler from lr_config: {cfg.param_scheduler}")
                # cfg.pop('lr_config', None) # Remove old key
            else:
                 print(f"Warning: Cannot auto-translate lr_config policy '{lr_step_config.policy}' to param_scheduler. Leaving param_scheduler undefined.")
                 # Set to None or [] if needed, or raise error depending on strictness
                 cfg.param_scheduler = None
        else:
            print("Warning: lr_config not found. param_scheduler will be undefined.")
            cfg.param_scheduler = None # Explicitly set to None

    # Ensure val_cfg exists if val_dataloader and val_evaluator are defined
    if (cfg.get('val_dataloader') is not None and
            cfg.get('val_evaluator') is not None and
            cfg.get('val_cfg') is None):
        print("val_dataloader and val_evaluator exist, but val_cfg is missing. Injecting default val_cfg.")
        cfg.val_cfg = dict(type='ValLoop') # Default validation loop

    # Ensure test_cfg exists if test_dataloader and test_evaluator are defined (less common during training)
    if (cfg.get('test_dataloader') is not None and
            cfg.get('test_evaluator') is not None and
            cfg.get('test_cfg') is None):
        print("test_dataloader and test_evaluator exist, but test_cfg is missing. Injecting default test_cfg.")
        cfg.test_cfg = dict(type='TestLoop') # Default test loop

    # 2. Use only a subset of the training data (e.g., first 200 samples)
    subset_size = 200
    if 'train_dataloader' in cfg and 'dataset' in cfg.train_dataloader:
        # Check if the dataset is wrapped (e.g., RepeatDataset)
        original_dataset_cfg = None
        if cfg.train_dataloader.dataset.get('type') == 'RepeatDataset':
             print("Detected RepeatDataset wrapper. Modifying the underlying dataset.")
             original_dataset_cfg = cfg.train_dataloader.dataset.dataset # Get the actual dataset config
        else:
             original_dataset_cfg = cfg.train_dataloader.dataset # Assume it's the direct dataset config

        if original_dataset_cfg is not None:
            # Option 1: Modify data_prefix and ann_file if they support subset loading (less common directly)
            # Option 2: Add filtering in the pipeline (more flexible)
            # Option 3: Use dataset-specific parameters like 'indices' (if supported)
            # Option 4: Add a 'metainfo' override or filter directly if possible (new MMEngine feature?)

            # We'll use 'indices' as a common pattern.
            print(f"Attempting to limit train dataset to first {subset_size} samples.")
            # original_dataset_cfg['indices'] = subset_size # Pass an integer N - Causes TypeError in custom loader
            # original_dataset_cfg['indices'] = list(range(subset_size)) # Pass list - Causes IndexError due to double subsetting
            # Instead, pass a custom parameter for the loader to handle subsetting directly
            original_dataset_cfg['subset_size'] = subset_size 
            print(f"Set custom 'subset_size={subset_size}' in the dataset configuration.")
            # Try enabling lazy init to potentially bypass serialization issues with empty filtered list
            original_dataset_cfg['lazy_init'] = True
            print("Set 'lazy_init=True' in the dataset configuration.")
            # Also remove filtering config, as it might filter out all subset samples
            if 'filter_cfg' in original_dataset_cfg:
                 print(f"Removing filter_cfg: {original_dataset_cfg['filter_cfg']}")
                 original_dataset_cfg.pop('filter_cfg', None)
            else:
                 print("filter_cfg not found in dataset config.")

            # If using RepeatDataset, put the modified config back
            if cfg.train_dataloader.dataset.get('type') == 'RepeatDataset':
                cfg.train_dataloader.dataset.dataset = original_dataset_cfg
        else:
             print("Warning: Could not locate the primary dataset config within train_dataloader.")

    else:
        print("Warning: train_dataloader or train_dataloader.dataset not found in config. Cannot apply subset.")
    # ------------------------------------

    # Make sure work_dir exists
    os.makedirs(osp.abspath(cfg.work_dir), exist_ok=True)

    # Enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print('AMP training is already enabled in config.')
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # Enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # Resume training
    cfg.resume = args.resume

    # Update the model type in the config
    original_model_type = cfg.model.type
    cfg.model.type = original_model_type
    print(f"Updated model type in config from '{original_model_type}' to '{cfg.model.type}'")

    # --- Skip Validation for Subset Training --- 
    print("Skipping validation for subset training run by setting val_dataloader/evaluator/cfg to None.")
    cfg.val_dataloader = None
    cfg.val_evaluator = None
    cfg.val_cfg = None 
    # ------------------------------------------

    # Build the runner
    if 'runner_type' not in cfg:
        # Build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # Build customized runner
        runner = RUNNERS.build(cfg)

    # Start training
    print("\n---> Starting training loop... <---\n")
    runner.train()


if __name__ == '__main__':
    main() 