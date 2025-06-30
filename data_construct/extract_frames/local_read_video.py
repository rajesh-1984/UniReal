from PIL import Image
import numpy as np
from decord import VideoReader
import random
import cv2
from skimage.metrics import structural_similarity as ssim

def compute_ssim(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    return ssim(gray1, gray2, full=True)[0]


def resize_to_l(image, l=512):
    """
    Resize the shortest eadge to "l"
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


def read_random_frames(filename, interval_seconds=5, num_frames=2, retry=5):
    """
    Randomly sample frames from a video at a given interval and return if they are dissimilar enough.

    Parameters:
    - filename: Path to the input video file.
    - interval_seconds: Time interval (in seconds) between sampled frames.
    - num_frames: Number of frames to sample.
    - retry: Number of retries if sampling fails or frames are too similar.

    Returns:
    - A list of sampled frames (as NumPy arrays), or None if retries are exhausted.
    """
    exception = None
    for _ in range(retry):
        try:
            # Load video using decord.VideoReader
            vr = VideoReader(filename)

            # Get frame rate and total number of frames
            fps = vr.get_avg_fps()
            total_frames = len(vr)
            video_duration = total_frames / fps  # Calculate video duration

            # Calculate frame interval in frame indices
            frame_interval = int(fps * interval_seconds)

            # Randomly select a valid starting frame index
            start_frame = random.randint(0, total_frames - frame_interval * (num_frames - 1) - 1)

            # Extract frames at defined interval
            frames = []
            for i in range(num_frames):
                frame_idx = start_frame + i * frame_interval
                frames.append(vr[frame_idx].asnumpy().astype(np.uint8))  # Convert to NumPy array

            # Compute similarity between the first two frames
            frame1, frame2 = frames[0], frames[1]
            frame1, frame2 = resize_to_l(frame1, 256), resize_to_l(frame2, 256)
            similarity = compute_ssim(frame1, frame2)

            # Return frames if similarity is below threshold
            if similarity < 0.95:
                return frames

        except Exception as e:
            exception = e
            continue

    # Return None if all retries failed or similarity was too high
    return None





if __name__ == "__main__":
    video_name = "/mnt/myworkspace/xic_space/data/VIDGEN-1M/VidGen_video_439/-q5AALF57ug-Scene-0068.mp4"
    frame_lst = read_random_frames(video_name)
    vis = cv2.hconcat(frame_lst)[:,:,::-1]
    cv2.imwrite('./example_frame.png', vis)
    print(len(frame_lst))