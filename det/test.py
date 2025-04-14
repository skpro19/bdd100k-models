"""Inference a pretrained model."""

import argparse
import os

# import datasets  # pylint: disable=unused-import # Keep commented
import torch
from mmengine.config import Config, DictAction
# import mmdet.datasets # Remove
# from mmcv.cnn import fuse_conv_bn # Remove (Fuse handled by deployment tools)
from torch.nn.parallel import DataParallel as MMDataParallel # Keep alias for potential future use
from torch.nn.parallel import DistributedDataParallel as MMDistributedDataParallel # Keep alias
from mmengine.dist import get_dist_info, init_dist
# from mmengine.runner.checkpoint import load_checkpoint # Remove (Handled by init_detector)
from mmdet.apis import init_detector, inference_detector
from mmengine.dataset import DefaultSampler
from torch.utils.data import DataLoader
from mmengine.registry import DATASETS
# from mmdet.datasets.transforms import PackDetInputs # Remove specific
# from mmdet.datasets.transforms import * # Remove
# from mmdet.datasets.transforms import Compose # Remove
# import mmdet # Remove

# Import specific transform classes needed by the pipeline
from mmdet.datasets.transforms.loading import LoadImageFromFile, LoadAnnotations
from mmdet.datasets.transforms.transforms import Resize
from mmdet.datasets.transforms.formatting import PackDetInputs

# Import the registry we need to populate
from mmengine.registry import TRANSFORMS

# Manually register only PackDetInputs into the mmengine registry
# TRANSFORMS.register_module(module=LoadImageFromFile) # Already registered
# TRANSFORMS.register_module(module=Resize) # Already registered
# TRANSFORMS.register_module(module=LoadAnnotations) # Already registered
TRANSFORMS.register_module(module=PackDetInputs) # This was the missing one

# Remove previous (ineffective) registration attempts
# import mmdet.datasets.transforms.loading

# Import custom dataset class AND the dataset registry
from datasets.bdd100k import BDD100KDetDataset
from mmengine.registry import DATASETS

# Import necessary components for manual DataLoader creation
from mmengine.dataset import pseudo_collate # Import the correct collate function

MODEL_SERVER = "https://dl.cv.ethz.ch/bdd100k/det/models/"


def parse_args() -> argparse.Namespace:
    """Arguements definitions."""
    parser = argparse.ArgumentParser(
        description="MMDet test (and eval) a model"
    )
    parser.add_argument("config", help="test config file path")
    parser.add_argument(
        "--work-dir",
        help="the directory to save the file containing evaluation metrics",
    )
    parser.add_argument(
        "--fuse-conv-bn",
        action="store_true",
        help="Whether to fuse conv and bn, this will slightly increase"
        "the inference speed",
    )
    parser.add_argument(
        "--format-only",
        action="store_true",
        help="Format the output results without perform evaluation. It is"
        "useful when you want to format the result to a specific format and "
        "submit it to the test server",
    )
    parser.add_argument(
        "--format-dir", help="directory where the outputs are saved."
    )
    parser.add_argument("--show", action="store_true", help="show results")
    parser.add_argument(
        "--show-dir", help="directory where painted images will be saved"
    )
    parser.add_argument(
        "--show-score-thr",
        type=float,
        default=0.3,
        help="score threshold (default: 0.3)",
    )
    parser.add_argument(
        "--gpu-collect",
        action="store_true",
        help="whether to use gpu to collect results.",
    )
    parser.add_argument(
        "--tmpdir",
        help="tmp directory used for collecting results from multiple "
        "workers, available when gpu-collect is not specified",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of validation samples to process for quick testing."
    )
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def main() -> None:
    """Main function for model inference."""
    args = parse_args()

    assert args.format_only or args.show or args.show_dir, (
        "Please specify at least one operation (save/eval/format/show the "
        "results / save the results) with the argument '--format-only', "
        "'--show' or '--show-dir'"
    )

    cfg = Config.fromfile(args.config)
    if cfg.load_from is None:
        cfg_name = os.path.split(args.config)[-1].replace(".py", ".pth")
        cfg.load_from = MODEL_SERVER + cfg_name
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    # Determine device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader manually using cfg.test_dataloader settings

    # 1. Build the dataset using the config nested in test_dataloader
    dataset_cfg = cfg.test_dataloader.dataset
    dataset = DATASETS.build(dataset_cfg)

    # 2. Build the sampler (using DefaultSampler as specified in config)
    #    We'll assume DefaultSampler for simplicity; build_sampler could be used if needed
    sampler_cfg = cfg.test_dataloader.get('sampler', dict(type='DefaultSampler'))
    # Ensure dataset is passed if needed by sampler constructor (DefaultSampler needs it)
    if sampler_cfg['type'] == 'DefaultSampler':
        sampler = DefaultSampler(dataset, shuffle=False) # Use shuffle=False for testing
    else:
        # Fallback or build using build_sampler if more complex samplers are needed
        # from mmengine.dataset import build_sampler
        # sampler = build_sampler(sampler_cfg, default_args=dict(dataset=dataset))
        print(f"Warning: Using DefaultSampler as fallback for sampler type {sampler_cfg['type']}")
        sampler = DefaultSampler(dataset, shuffle=False)

    # 3. Build the DataLoader instance
    data_loader = DataLoader(
        dataset,
        batch_size=cfg.test_dataloader.batch_size,
        sampler=sampler,
        num_workers=cfg.test_dataloader.num_workers,
        collate_fn=pseudo_collate, # Use MMEngine's collate function
        pin_memory=True,
        persistent_workers=cfg.test_dataloader.get('persistent_workers', False),
        drop_last=cfg.test_dataloader.get('drop_last', False)
    )

    # build the model and load checkpoint using init_detector
    # Pass the updated config directly
    model = init_detector(cfg, cfg.load_from, device=device)
    # Fuse Conv+BN is usually handled by deployment tools, remove for now
    # if args.fuse_conv_bn:
    #     model = fuse_conv_bn(model)

    # Get CLASSES - Check if init_detector already handled this
    if not hasattr(model, 'CLASSES'): # Check if CLASSES attribute is missing
        if hasattr(dataset, 'metainfo'): # Use metainfo if available
            model.CLASSES = dataset.metainfo.get('classes', None)
        elif hasattr(dataset, 'CLASSES'): # Fallback for older dataset style
            model.CLASSES = dataset.CLASSES
        else:
            print("Warning: Could not automatically determine CLASSES.")
            # Consider raising an error or setting a default if essential
            pass

    # New inference loop (single GPU for now)
    results = []
    from tqdm import tqdm
    # Dataloader yields dict formatted by pseudo_collate
    # Keys: 'inputs', 'data_samples'
    processed_count = 0 # Counter for processed samples
    for data in tqdm(data_loader):
        # Stop if max_samples is reached
        if args.max_samples is not None and processed_count >= args.max_samples:
            break

        # Remove manual device placement
        # # Ensure data is on the correct device
        # # Manually move inputs tensor and data_samples to device
        # inputs = data['inputs'].to(device)
        # data_samples = [ds.to(device) for ds in data['data_samples']]
        #
        # # Reconstruct data dict for model.test_step
        # batch_data = {'inputs': inputs, 'data_samples': data_samples}

        # Inference detector now expects the data dictionary from the dataloader
        with torch.no_grad():
            # Pass the original data dictionary directly to model.test_step
            # It should handle device placement internally via data_preprocessor
            result = model.test_step(data)
        results.extend(result) # test_step returns a list of DataSample results
        processed_count += len(data['data_samples']) # Increment counter by batch size (usually 1 here)

    # Remove old manual inference loop
    # # Dataloader yields dict from old pipeline (likely {'img': [Tensor], 'img_metas': [...]})
    # for data in tqdm(data_loader):
    #     # Assuming 'inputs' key holds image tensor(s) and 'data_sample' holds metadata # <- Comment outdated
    #     # Need to adapt based on actual dataloader output structure
    #     # inference_detector expects image path or numpy array (H, W, C), BGR
    # 
    #     # Example assuming data['inputs'] is a tensor B x C x H x W # <- Comment outdated
    #     # This needs careful verification based on the actual data loader output
    # 
    #     # Extract tensor based on likely old format
    #     # Data is list because of Collect transform, Batch size is 1
    #     img_tensor = data['img'][0]
    #     # Need to move tensor to the correct device if not already done by DataLoader/Collate
    #     img_tensor = img_tensor.to(device)
    # 
    #     img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    #     # Convert RGB (from ToTensor default + img_norm_cfg) to BGR for inference_detector
    #     img_np = img_np[:, :, ::-1]
    # 
    #     result = inference_detector(model, img_np) # Pass BGR numpy array
    #     results.append(result)

    # outputs = results # Use the collected results

    rank, _ = get_dist_info()
    if rank == 0:
        if args.format_only:
            # Use the dataset object obtained from the dataloader
            if hasattr(dataset, 'format_results') and callable(dataset.format_results):
                 print(f"Formatting results to: {args.format_dir}")
                 # format_results is the new name for convert_format in MMDetection 3.x
                 dataset.format_results(results, args.format_dir) # Use refactored results
            # Fallback to old name just in case
            elif hasattr(dataset, 'convert_format') and callable(dataset.convert_format):
                 print(f"Formatting results using 'convert_format' to: {args.format_dir}")
                 dataset.convert_format(results, args.format_dir)
            else:
                 print("Warning: dataset format method not found or not callable. Skipping format conversion.")
                 # Potentially save raw results here if needed
                 # import pickle
                 # with open(os.path.join(args.format_dir, 'raw_results.pkl'), 'wb') as f:
                 #    pickle.dump(results, f)


if __name__ == "__main__":
    main()
