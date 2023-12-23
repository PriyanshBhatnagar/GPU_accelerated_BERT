# Accelerating Inference Time in NLP Transformer BERT model using PyTorch Cuda

## Course: ECE 277- GPU Programming

In this project, we explore ways to accelerate the forward propagation architecture of the BERT language model using PyTorch and CUDA. We accelerate the Multi-headed Attention module with custom cuda for Multistream processing and Matrix Multiplication, integrating it into the Python-based model via Pybind. Our goal is to boost inference speed over CPU execution, demonstrating the efficiency gains possible in NLP models with GPU acceleration.

## Steps involved
1. Environmental and Data Setup
2. Develop BERT's forward propagation architecture using Python PyTorch
3. Integrate Pretrained BERT weights & initial testing on language Task MRPC 
4. Custom CUDA function for Matrix Multiplication
5. Use #4 to Transition Multi Head Self Attention (Attn) module to GPU using PyBind
6. Experiment with Optimization techniques - Optimized for single sample and multiple samples 
7. Run inference and analyze on CPU vs GPU

### Getting Started

#### Prerequisites
- PyTorch C++ 
- CUDA-enabled GPU
- Python version 3.6+
- PyTorch
- PyBind

#### Testing on Inference
To start with the project, set the environment variables and run the main classification script:

```bash
python classify.py 
