# Accelerating Inference Time in NLP Transformer BERT model using PyTorch Cuda

In this project, we explore ways to accelerate the forward propagation architecture of the BERT language model using PyTorch and CUDA. We accelerate the Multi-headed Attention module with custom cuda for Multistream processing and Matrix Multiplication, integrating it into the Python-based model via Pybind. Our goal is to boost inference speed over CPU execution, demonstrating the efficiency gains possible in NLP models with GPU acceleration.

## Steps involved
1. Environmental and Data Setup
2. Develop BERT's forward propagation architecture using Python PyTorch
   
   NOTE: We have used the official implementation of BERT.
   Copyright 2018 Dong-Hyun Lee, Kakao Brain.
   (Strongly inspired by original Google BERT code and Hugging Face's code)
   
4. Integrate Pretrained BERT weights & initial testing on language Task MRPC 
5. Custom CUDA function for Matrix Multiplication
6. Use #4 to Transition Multi Head Self Attention (Attn) module to GPU using PyBind
7. Experiment with Optimization techniques - Optimized for single sample and multiple samples 
8. Run inference and analyze on CPU vs GPU

### Getting Started

#### Prerequisites
- PyTorch C++ 
- CUDA-enabled GPU
- Python version 3.6+
- PyTorch
- PyBind

To run the project, Pybind11 and C++ modules have to be downloaded and added to the working repository. Paths have to be manipulated accordingly.

#### Testing on Inference
To start with the project, set the environment variables and run the main classification script:

```bash
python classify.py 
