import os
import argparse
import signal
import sys

# Enable fallback to CPU for unsupported MPS operations (for Apple devices)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Parse arguments before importing torch
parser = argparse.ArgumentParser(description="Video annotation pipeline with Kosmos-2 and SAM2")
parser.add_argument("-G", type=int, required=True, help="GPU index to use")
parser.add_argument("-k", type=int, required=True, help="Index of the chunk to process")
parser.add_argument("-N", type=int, required=True, help="Total number of chunks")
args = parser.parse_args()

# Set CUDA device before importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.G)
k, N = args.k, args.N

# Register signal handler for Ctrl+C
def handler(signum, frame):
    print("\n⛔ Received SIGINT (Ctrl+C), exiting...")
    sys.exit(0)

signal.signal(signal.SIGINT, handler)

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Kosmos2ForConditionalGeneration
from tqdm import tqdm
import cv2
import json
import traceback
from sam2.build_sam import build_sam2_video_predictor


def get_video_segments(predictor, frames_np_lst_selected, boxes):
    """
    Apply SAM2 segmentation to the input frames and propagate masks across frames.
    """
    inference_state = predictor.init_state(video_path=None, video_frames=frames_np_lst_selected)
    predictor.reset_state(inference_state)

    ann_frame_idx = 0  # Interact with the first frame

    for i, box in enumerate(boxes):
        ann_obj_id = i + 1
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            box=box
        )

    # Collect segmentation results for each frame
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i, 0] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    return video_segments


def remove_small_areas_and_fill_holes(mask):
    """
    Remove small connected components and fill holes in a binary mask.
    """
    total_area = np.sum(np.ones_like(mask))
    min_area_threshold = total_area / 100

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    filtered_mask = np.zeros_like(mask, dtype=np.uint8)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area_threshold:
            filtered_mask[labels == label] = 1

    kernel = np.ones((5, 5), np.uint8)
    filled_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_CLOSE, kernel)
    return filled_mask


def keep_with_edges(mask):
    """
    Determine if a mask touches more than one edge of the image.
    Return 0 if it touches more than one edge, else 1.
    """
    num_edge = 0
    if np.any(mask[0, :] == 1): num_edge += 1
    if np.any(mask[-1, :] == 1): num_edge += 1
    if np.any(mask[:, 0] == 1): num_edge += 1
    if np.any(mask[:, -1] == 1): num_edge += 1
    return 0 if num_edge > 1 else 1


def check_mask_size(mask):
    """
    Check if the proportion of the mask is within a reasonable range.
    """
    mask_portion = mask.sum() / np.ones_like(mask).sum()
    return 1 if 0.01 < mask_portion < 0.36 else 0


def mask_overlap_ratio(new_mask, prev_mask):
    """
    Calculate the overlap ratio between two binary masks.
    """
    overlap_area = np.sum((new_mask == 1) & (prev_mask > 0))
    total_area = np.sum(new_mask == 1)
    return overlap_area / total_area if total_area > 0 else 0.0


def process_segments(video_segments, frames_np_lst_selected):
    """
    Process the segmentation result to produce masks for both frames.
    Apply filtering to remove small, edge-touching, or overlapping regions.
    """
    first_frame_segment = video_segments[0]
    first_frame_mask = np.zeros_like(frames_np_lst_selected[0])[:, :, 0]
    for i in range(len(first_frame_segment)):
        mask = remove_small_areas_and_fill_holes(first_frame_segment[i + 1])
        if check_mask_size(mask) and keep_with_edges(mask) and mask_overlap_ratio(mask, first_frame_mask) < 0.8:
            first_frame_mask = first_frame_mask * (1 - mask) + (i + 1) * mask

    last_frame_segment = video_segments[1]
    last_frame_mask = np.zeros_like(frames_np_lst_selected[0])[:, :, 0]
    for i in range(len(last_frame_segment)):
        mask = remove_small_areas_and_fill_holes(last_frame_segment[i + 1])
        if check_mask_size(mask) and keep_with_edges(mask) and mask_overlap_ratio(mask, last_frame_mask) < 0.8:
            last_frame_mask = last_frame_mask * (1 - mask) + (i + 1) * mask

    image1, image2 = frames_np_lst_selected
    mask1, mask2 = first_frame_mask * 20, last_frame_mask * 20
    return image1, image2, mask1, mask2


def cap_image(image_np):
    """
    Generate caption and bounding boxes using Kosmos-2 on a given image.
    """
    height, width, _ = image_np.shape
    image = Image.fromarray(image_np)
    prompt = "<grounding> An image of"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    generated_ids = model.generate(
        pixel_values=inputs["pixel_values"],
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_embeds=None,
        image_embeds_position_mask=inputs["image_embeds_position_mask"],
        use_cache=True,
        max_new_tokens=64,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    caption, entities = processor.post_process_generation(generated_text)

    boxes_real = []
    for ent in entities:
        box = ent[2][0]
        x1, y1 = int(box[0] * width), int(box[1] * height)
        x2, y2 = int(box[2] * width), int(box[3] * height)
        boxes_real.append(np.array([x1, y1, x2, y2], dtype=np.float32))

    return caption, entities, boxes_real


def split_list(lst, N, K):
    """
    Split a list into N chunks and return the K-th chunk (1-based index).
    """
    if N <= 0 or K <= 0 or K > N:
        raise ValueError("N must be > 0 and K must be between 1 and N inclusive")

    length = len(lst)
    chunk_size = length // N
    remainder = length % N

    chunks = []
    start = 0
    for i in range(N):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(lst[start:end])
        start = end

    return chunks[K - 1]


if __name__ == '__main__':
    # Initialize models
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = Kosmos2ForConditionalGeneration.from_pretrained("/mnt/myworkspace/xic_space/models/kosmos-2-patch14-224").to(device) # your path
    processor = AutoProcessor.from_pretrained("/mnt/myworkspace/xic_space/models/kosmos-2-patch14-224") # your path

    sam2_checkpoint = '/mnt/myworkspace/xic_space/models/SAM2/sam2.1_hiera_large.pt' # your path
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    # Dataset paths and chunking
    total_indexes = list(range(2048)) # your processing part
    part_indexes = split_list(total_indexes, N, k)
    data_root = "/mnt/myworkspace/xic_space/data/frame_vidgen/"  # your data root
    save_root = "/mnt/myworkspace/xic_space/data/frame_vidgen_annotation_v2_test/"  # your save root

    try:
        for part_i in part_indexes:
            data_dir = os.path.join(data_root, f'VidGen_video_{part_i}/')
            save_dir = os.path.join(save_root, f'VidGen_video_{part_i}/')
            os.makedirs(save_dir, exist_ok=True)

            image_names = [i for i in os.listdir(data_dir) if i.endswith('.png')]#[:10]

            for i in tqdm(range(len(image_names)), leave=False):
                json_dict = {}
                image_path = os.path.join(data_dir, image_names[i])
                save_path = os.path.join(save_dir, image_names[i])

                # Load and split image into two halves
                image = Image.open(image_path)
                width, height = image.size
                half_width = width // 2
                image1 = image.crop((0, 0, half_width, height))
                image2 = image.crop((half_width, 0, width, height))
                frames_np_lst_selected = [np.array(image1), np.array(image2)]

                # Generate caption and object boxes
                caption, entities, boxes = cap_image(frames_np_lst_selected[-1])
                json_dict['caption'] = caption
                json_dict['entities'] = entities
                save_json_path = save_path.replace('.png', '.json')

                # Get segmentation masks
                video_segments = get_video_segments(predictor, frames_np_lst_selected, boxes)
                image1, image2, mask1, mask2 = process_segments(video_segments, frames_np_lst_selected)

                # Stack and concatenate masks
                mask1 = np.stack([mask1] * 3, axis=-1)
                mask2 = np.stack([mask2] * 3, axis=-1)
                mask_save = cv2.hconcat([mask1, mask2])

                if mask1.sum() > 0 and mask2.sum() > 0:
                    cv2.imwrite(save_path, mask_save)
                    with open(save_json_path, 'w') as file:
                        json.dump(json_dict, file, indent=4)

    except Exception as e:
        print("===== Error during processing =====")
        traceback.print_exc()
