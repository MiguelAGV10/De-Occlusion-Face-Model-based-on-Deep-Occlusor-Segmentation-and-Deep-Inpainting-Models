De-Occlusion Face Model based on Deep Occlusor Segmentation and Deep Inpainting Models

Authors: Miguel A. Gutierrez-Velazquez, Mario I. Chacon-Murguia, and Juan A. Ramirez-Quintana
Article ID: 9612

Overview
We present an ensemble learning approach for face inpainting. The model can use binary masks, real occluders, and synthetic occluders to train robustly across different occlusion scenarios.

Datasets
* Face images dataset: downloaded from here https://drive.google.com/drive/folders/1VGlK2ym0gI9bS-4rTLx-uL-jMHWMO5HQ?usp=sharing
* Binary masks dataset: downloaded from here https://drive.google.com/drive/folders/19CXiaFdaKhLNtvTJTRSNXf5UvXUp8x4t?usp=sharing.
* Real occluder dataset for self-supervised training: downloaded from here https://drive.google.com/drive/folders/1SoosnSGwbHYWGx_4Luj9YHPrXHPkcEqo?usp=sharing
* Smaller dataset for general inpainting: https://drive.google.com/drive/folders/11Biz2qc2iC8s2XdiaOKCCNm3fFAGwbtX?usp=sharing

Installation:
First, create the conda environment:

![image](https://github.com/user-attachments/assets/6fec32d1-3428-47ee-b1b5-3e81bcc8cf1c)

Training
We recommend consulting the README.txt. It includes detailed instructions.
Main training scripts:

![image](https://github.com/user-attachments/assets/687c4486-9470-4a2e-88a7-2ca0076c7532)

To run train_inpainting.py, execute from the console

![image](https://github.com/user-attachments/assets/99d60604-2335-4823-8fbd-017a49b2e9d9)

where [model_name] can be:
* Reconstruction
* ablationv1
* ablationv2
* ablationv3

Inference and Evaluation
Additional scripts:

![image](https://github.com/user-attachments/assets/b72d9948-ea36-4e60-931f-23475fb9d5e5)

Notes
* Training recommendation: We suggest using the Spyder GUI when possibly to easily track image outputs during training
* Console issues: If errors occur when training from the console, they may be related to the show_images function in the engine* files. Simply comment out those lines.
* We trained our models with: NVIDIA GeForce RTX 3060, 13th Gen Intel(R) Core(TM) i9-13900KF 3.00 GHz, 64 GB Ram
