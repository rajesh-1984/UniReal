import os
import json
import random
import cv2
from PIL import Image
import base64
import requests
import io
from tqdm import tqdm
from openai import AzureOpenAI
import numpy as np
import time



def resize_image_long_side(image, ls=224):
    # Get the height and width of the image
    h, w = image.shape[:2]
    
    # Calculate the scaling factor
    if h > w:
        scale = ls / h  # Height is the longer side
    else:
        scale = ls / w  # Width is the longer side
    
    # Calculate the new size
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize the image
    resized_image = cv2.resize(image.astype(np.uint8), (new_w, new_h))
    
    return resized_image


def compute_ssim(img1, img2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    
    # Compute SSIM (Structural Similarity Index)
    return ssim(gray1, gray2, full=True)


def encode_image_from_pil(image):
    # Create a byte stream buffer
    buffered = io.BytesIO()

    # Save the image to the buffer
    image.save(buffered, format="JPEG")  # You can change the format if needed, e.g., "PNG"
    
    # Get the byte data
    img_byte = buffered.getvalue()
    
    # Encode as a Base64 string
    return base64.b64encode(img_byte).decode('utf-8') 





def gpt_4o_mini(image1, image2, prompt = None):    
    api_key = "sk-xxx" 
        
    image1 = Image.fromarray(image1)
    base64_image1 = encode_image_from_pil(image1)
    
    image2 = Image.fromarray(image2)
    base64_image2 = encode_image_from_pil(image2)
    
    prompt = """You should figure out the content differences between the second image and the first image.
    Give a sentence of instruction to make the first image like the second image, like "make the boy in yellow raise up its hand." 
    Imaging that you can only see the first image, do not say the word "the first image" or "the second image"
    """
    
    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
            "role": "user",
            "content": [                
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image1}"
                }
                },
                
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image2}"
                }
                },
                {
                "type": "text",
                "text": prompt
                },
            ]
            }
        ],
        "max_tokens": 100
        }
    
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    ans = response.json()['choices'][0]['message']['content']
    return ans



image1 = cv2.imread('/mnt/myworkspace/xic_space/data/dog-example/alvan-nee-brFsZ7qszSY-unsplash.jpeg')
image2 = cv2.imread('/mnt/myworkspace/xic_space/data/dog-example/alvan-nee-eoqnr8ikwFE-unsplash.jpeg')

            
image1 = resize_image_long_side(image1,224)
image2 = resize_image_long_side(image2,224)


caption = gpt_4o_mini(image1, image2)
print(caption)
