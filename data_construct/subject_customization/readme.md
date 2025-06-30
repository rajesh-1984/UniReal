# UniReal Customized Generation 

## 1. Setup environment 
Please refer to https://github.com/facebookresearch/sam2 to install requirements and download the ckpt of SAM 2.1 Large 

## 2. Labling grounding caption the segmentation masks 
```
cd notebooks 
bash process_data_openvid.sh 
bash process_data_vidgen.sh 
```
Details refer to process_data_openvid.py / process_data_vidgen.py