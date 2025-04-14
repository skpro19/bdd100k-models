"""Definition of the BDD100K dataset."""

import os
import os.path as osp
from typing import List
import json
import time # Import time module

import numpy as np
from mmengine.registry import DATASETS
from mmdet.datasets import CocoDataset
from scalabel.label.io import save
from scalabel.label.transforms import bbox_to_box2d
from scalabel.label.typing import Frame, Label, Dataset


@DATASETS.register_module()
class BDD100KDetDataset(CocoDataset):  # type: ignore
    """BDD100K Dataset for detecion."""

    CLASSES = [
        "pedestrian",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
        "traffic light",
        "traffic sign",
    ]

    def __init__(self, *args, **kwargs):
        # Pop the custom argument before passing to parent
        self.subset_size = kwargs.pop('subset_size', None)
        print(f"[BDD100KDetDataset __init__] Found subset_size: {self.subset_size}")
        super().__init__(*args, **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotations from BDD100K format."""
        print(f"[BDD100KDetDataset] Loading annotations from: {self.ann_file}") # Log start
        start_time = time.time()
        assert osp.exists(self.ann_file), f'Annotation file not found: {self.ann_file}'

        with open(self.ann_file, 'r') as f:
            try:
                bdd_data = json.load(f)
                json_load_time = time.time()
                print(f"[BDD100KDetDataset] JSON loading took {json_load_time - start_time:.2f} seconds.")
            except json.JSONDecodeError as e:
                print(f"ERROR: Failed to decode JSON from {self.ann_file}: {e}")
                raise

        data_list = []
        cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}
        self.cat_ids = list(range(len(self.CLASSES)))
        self.img_ids = []
        self.cat_img_map = {cat_id: [] for cat_id in self.cat_ids}
        # img_id_counter = 0 # Use actual index if loading subset

        # Determine the list of frames to process
        # frames_to_process = bdd_data # Always process the full data
        # Check if a subset size is specified in the config
        subset_size = getattr(self, 'subset_size', None)
        if subset_size is not None and isinstance(subset_size, int) and subset_size > 0:
            print(f"[BDD100KDetDataset] Taking the first {subset_size} frames as specified by config.")
            frames_to_process = bdd_data[:subset_size]
            if len(frames_to_process) < subset_size:
                 print(f"Warning: Requested subset_size={subset_size}, but only {len(frames_to_process)} frames available in JSON.")
        else:
            frames_to_process = bdd_data # Process the full data if subset_size not specified
            
        print(f"[BDD100KDetDataset] Processing {len(frames_to_process)} frames...")
        process_start_time = time.time()

        # Use the actual index from the original file as the img_id for consistency
        # when loading a subset.
        for img_id, frame in enumerate(frames_to_process): # Use enumerate index as img_id
            self.img_ids.append(img_id) 
            # img_id_counter += 1 # Not needed if using original index
            
            img_path = osp.join(self.data_prefix.get('img_path', ''), frame['name']) # Add default ''
            if not self.data_prefix.get('img_path'):
                 print(f"Warning: data_prefix['img_path'] is missing or None.")
            
            file_name = osp.basename(img_path)
            height = -1
            width = -1
            
            data_info = {
                # 'img_id': i, # Use the unique integer ID
                'img_id': img_id,
                'img_path': img_path,
                'file_name': file_name,
                'height': height, 
                'width': width,
                'instances': []
            }

            image_has_relevant_annotation = False # Track if image has annotations for cat_img_map
            if 'labels' in frame:
                for ann in frame['labels']:
                    if ann['category'] in cat2label:
                        label = cat2label[ann['category']]
                        # Add image to cat_img_map for this category
                        if img_id not in self.cat_img_map[label]:
                            self.cat_img_map[label].append(img_id)
                        image_has_relevant_annotation = True
                            
                        if 'box2d' in ann:
                            bbox = [
                                ann['box2d']['x1'],
                                ann['box2d']['y1'],
                                ann['box2d']['x2'],
                                ann['box2d']['y2'],
                            ]
                            # Prepare instance dict
                            instance = {
                                'bbox': bbox,
                                'bbox_label': label,
                                'ignore_flag': 0, # Default to not ignored
                                # Add other fields if necessary (e.g., segmentation)
                            }
                            data_info['instances'].append(instance)

            data_list.append(data_info)
            
            if (img_id + 1) % 10000 == 0: # Log progress every 10k frames
                print(f"  Processed {img_id + 1} / {len(frames_to_process)} frames...")

        process_end_time = time.time()
        print(f"[BDD100KDetDataset] Frame processing took {process_end_time - process_start_time:.2f} seconds.")

        # Set self.cat_ids based on the CLASSES definition
        self.cat_ids = list(range(len(self.CLASSES)))
        print(f"[BDD100KDetDataset] Set self.cat_ids: {self.cat_ids}") # Debug print
        print(f"[BDD100KDetDataset] Built cat_img_map (sample): {{ {next(iter(self.cat_img_map.items()))[0]}: {next(iter(self.cat_img_map.items()))[1][:5]}... }}") # Debug print

        return data_list

    # Add the missing xyxy2xywh method
    def xyxy2xywh(self, bbox: np.ndarray) -> np.ndarray:
        """Convert [x1, y1, x2, y2] box format to [x, y, w, h] format."""
        _bbox = bbox.tolist()
        return np.array([
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ])

    def convert_format(
        self, results: List[List[np.ndarray]], out_dir: str  # type: ignore
    ) -> None:
        """Format the results to the BDD100K prediction format."""
        assert isinstance(results, list), "results must be a list"
        # Remove or comment out the problematic assertion for subset processing
        # assert len(results) == len(
        #     self
        # ), f"Length of res and dset not equal: {len(results)} != {len(self)}"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # --- Start: Reload full data list for formatting ---
        # Reload the full data list to ensure correct mapping,
        # as self.data_list might be altered by the test runner with --max-samples.
        print("Reloading full data list for formatting...") # Add print for debugging
        full_data_list = self.load_data_list()
        print(f"Full data list length: {len(full_data_list)}") # Add print for debugging
        print(f"Results length: {len(results)}") # Add print for debugging
        # Ensure the number of results doesn't exceed the full list
        assert len(results) <= len(full_data_list), (
            f"More results ({len(results)}) than entries "
            f"in full data list ({len(full_data_list)})!"
        )
        # --- End: Reload full data list ---

        frames = []
        ann_id = 0

        # Iterate based on the length of the results, not the full dataset
        for img_idx in range(len(results)):
            # Access data_list instead of data_infos and use file_name key
            # img_name = self.data_infos[img_idx]["file_name"]
            img_name = full_data_list[img_idx]["file_name"] # Use reloaded list
            frame = Frame(name=img_name, url=None, labels=[]) # Add url=None
            frames.append(frame)

            result = results[img_idx] # result is likely a DetDataSample

            # --- Start: Modify loop for DetDataSample --- 
            # Access predictions from the DetDataSample object
            pred_instances = result.pred_instances

            # Iterate through detected instances
            for i in range(len(pred_instances)):
                # Extract data for each instance
                bbox = pred_instances.bboxes[i].cpu().numpy() # Get bbox as numpy array
                label_idx = pred_instances.labels[i].cpu().item() # Get predicted label index
                score = pred_instances.scores[i].cpu().item() # Get score

                # Check if score is above a threshold (optional, but common)
                # if score < 0.3: # Example threshold
                #    continue

                # Get category name from index
                if label_idx < len(self.CLASSES):
                    category_name = self.CLASSES[label_idx]
                else:
                    # Handle cases where label_idx might be out of bounds if needed
                    category_name = f"unknown_label_{label_idx}" 
                
                ann_id += 1
                label = Label(
                    id=ann_id,
                    score=score,
                    # Convert XYXY format from mmdet to Box2D for scalabel
                    box2d=bbox_to_box2d(self.xyxy2xywh(bbox)),
                    category=category_name,
                    # Add missing optional fields required by pydantic validation
                    box3d=None,
                    poly2d=None,
                    rle=None,
                    graph=None,
                )
                frame.labels.append(label) # Add label to the current frame
            # --- End: Modify loop for DetDataSample ---

            # --- Start: Remove old loop --- 
            # for cat_idx, bboxes in enumerate(result): # Old loop iterating over result directly
            #     for bbox in bboxes:
            #         ann_id += 1
            #         label = Label(
            #             id=ann_id,
            #             score=bbox[-1],
            #             box2d=bbox_to_box2d(self.xyxy2xywh(bbox)),
            #             category=self.CLASSES[cat_idx],
            #         )
            #         frame.labels.append(label)  # type: ignore
            # --- End: Remove old loop ---

        out_path = osp.join(out_dir, "det.json")
        # Wrap frames in a Dataset object for saving
        dataset_to_save = Dataset(frames=frames, groups=None, config=None)
        save(out_path, dataset_to_save)
