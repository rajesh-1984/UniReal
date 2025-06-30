import os 
import argparse
from tqdm import tqdm
import json 

parser = argparse.ArgumentParser(description="An example program with argparse")
parser.add_argument("-G", type=int, required=True, help="Value of parameter a")
parser.add_argument("-k", type=int, required=True, help="Value of parameter b")
parser.add_argument("-N", type=int, required=True, help="Value of parameter b")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.G)
k, N = args.k, args.N
print('configs: ', args.k, args.N, args.G)

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import base64
import numpy as np
from io import BytesIO
from PIL import Image



model_id = '/mnt/myworkspace/xic_space/data/Qwen2.5-VL-7B-Instruct' # your model path
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
     model_id,
     torch_dtype=torch.bfloat16,
     attn_implementation="flash_attention_2",
     device_map="auto",
 )

# default processer
processor = AutoProcessor.from_pretrained(model_id)


def split_list(lst, N, K):
    """
    Split the list `lst` into N equal parts in order and return the K-th part.
    
    :param lst: The list to be split
    :param N: Number of parts to split into
    :param K: The index of the part to return (1-based)
    :return: The K-th part of the list
    """
    if N <= 0 or K <= 0 or K > N:
        raise ValueError("N must be greater than 0 and K must be between 1 and N")
    
    length = len(lst)
    chunk_size = length // N
    remainder = length % N
    
    # Calculate the start and end index of each chunk
    chunks = []
    start = 0
    for i in range(N):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(lst[start:end])
        start = end
    return chunks[K - 1]



def np_image_to_base64(np_image):
    # Convert NumPy array to PIL Image
    image = Image.fromarray(np_image)
    
    # Save image to a BytesIO object
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    
    # Convert the image to base64
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def infer_image(image_path, prompt):
    image = Image.open(image_path)

    # Get the image dimensions
    width, height = image.size

    # Assuming the two images are side by side, split the image at the midpoint of the width
    half_width = width // 2

    # Crop into two separate images
    image1 = image.crop((0, 0, half_width, height))  # Left image
    image2 = image.crop((half_width, 0, width, height))  # Right image

    # Convert both images to NumPy arrays
    np_image1 = np.array(image1)
    np_image2 = np.array(image2)

    # Convert both images to base64
    image_base64_1 = np_image_to_base64(np_image1)
    image_base64_2 = np_image_to_base64(np_image2)

    # Set up the message with both images in base64
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"data:image;base64,{image_base64_1}"},
                {"type": "image", "image": f"data:image;base64,{image_base64_2}"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Prepare inputs for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: generate the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]



total_index = [i for i in range(50)] # your labling part
selected_index = split_list(total_index, N, k)


prompt = """
Give a sentence of instruction to make the first image like the second image
Only return the instruction like "Make the boy in yellow raise up its hand."
"""

for part_index in selected_index:
    print(f'Processing fold index: {part_index}')
    sample_dir = f'/mnt/myworkspace/xic_space/data/frame_openvid/{part_index}/' # your sample dir
    sample_names = os.listdir(sample_dir)
    caption_dict = {}
    for name in tqdm(sample_names):
        try:
            image_path = os.path.join(sample_dir, name)
            caption = infer_image(image_path, prompt)
            caption_dict[name] = caption
        except:
            pass

    save_json_path = os.path.join(sample_dir, f"caption_part_{part_index}_qw.json") # your save json path
    with open(save_json_path, 'w') as file:
        json.dump(caption_dict, file, indent=4) 

