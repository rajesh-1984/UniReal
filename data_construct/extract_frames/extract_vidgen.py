from tqdm import tqdm
import cv2
from local_read_video import *  # assumes read_random_frames is defined here
import argparse
import os
import numpy as np


def resize_to_l(image, l=512):
    """
    Resize the shorter edge of an image to length 'l', maintaining aspect ratio.
    """
    height, width = image.shape[:2]

    if height < width:
        new_height = l
        new_width = int((new_height / height) * width)
    else:
        new_width = l
        new_height = int((new_width / width) * height)

    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image


def split_list(lst, N, K):
    """
    Split a list into N nearly equal parts and return the K-th part (1-based index).
    """
    if N <= 0 or K <= 0 or K > N:
        raise ValueError("N must be > 0 and K must be between 1 and N")

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


def calculate_low_pixel_ratio(image: np.ndarray, threshold: int = 5) -> float:
    """
    Calculate the ratio of pixels whose values in all 3 channels are below the given threshold.
    Useful for removing videos with black borders.

    Parameters:
        image (np.ndarray): Input image of shape (H, W, C).
        threshold (int): Pixel value threshold (default: 5).

    Returns:
        float: Proportion of pixels below the threshold across all 3 channels.
    """
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Input image must have shape (H, W, 3)")

    total_pixels = image.shape[0] * image.shape[1]
    low_pixel_mask = np.all(image < threshold, axis=2)
    low_pixel_count = np.sum(low_pixel_mask)

    return low_pixel_count / total_pixels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Frame sampling and filtering pipeline.")
    # The entire dataset is split into N parts; process the k-th part to enable multiprocessing or distributed execution
    parser.add_argument("-k", type=int, required=True, help="Index of the chunk to process")
    parser.add_argument("-N", type=int, required=True, help="Total number of chunks")
    args = parser.parse_args()
    k, N = args.k, args.N

    total_index = [i for i in range(1, 2048)]
    part_indexes = split_list(total_index, N, k)


    # you should adjust the following code according to the directories of your data
    save_root = '/mnt/myworkspace/xic_space/data/frames_6s_test/' #your save root
    data_dir = '/mnt/myworkspace/xic_space/data/VIDGEN-1M/' #your data root
    interval_seconds = 6 #your sampling interval

    try:
        for i in tqdm(part_indexes):
            folder_name = f'VidGen_video_{i}'
            save_dir = os.path.join(save_root, folder_name)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                print('Created new folder:', save_dir)

            source_dir = os.path.join(data_dir, folder_name)
            video_names = os.listdir(source_dir)
            print('Number of videos:', len(video_names))

            for video_name in tqdm(video_names, leave=False):
                try:
                    video_path = os.path.join(source_dir, video_name)
                    frame_lst = read_random_frames(video_path, interval_seconds)

                    if frame_lst is not None:
                        # remove images with black boundaries
                        ratio1 = calculate_low_pixel_ratio(frame_lst[0])
                        ratio2 = calculate_low_pixel_ratio(frame_lst[1])

                        if ratio1 < 0.1 and ratio2 < 0.1:
                            concat_frame = cv2.hconcat(frame_lst)
                            save_name = os.path.join(save_dir, f"{video_name.split('.')[0]}.png")
                            cv2.imwrite(save_name, concat_frame[:, :, ::-1])  # Convert RGB to BGR

                except KeyboardInterrupt:
                    print("\n⛔ Interrupted by user. Exiting cleanly.")
                    raise  # allow the outer loop to stop
                except Exception as e:
                    print(f"⚠️ Error processing video {video_name}: {e}")
    except KeyboardInterrupt:
        print("\n⛔ Interrupted by user. Program terminated.")
    except Exception as e:
        print("===== Unexpected error occurred =====")
        import traceback
        traceback.print_exc()
