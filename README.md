# ComfyUI Sa2VA

## Overview
A ComfyUI node implementation for [ByteDance's Sa2VA](https://github.com/bytedance/Sa2VA) (Segment Anything 2 Video Assistant) models, enabling advanced multimodal image and video understanding with precise segmentation capabilities.
This repo only implements the image portion of the model.

### What is Sa2VA?
Sa2VA is a state-of-the-art multimodal large language model (MLLM) that combines SAM2 (Segment Anything Model 2) with VLLMs for grounded understanding of images and videos. It achieves comparable performance to SOTA MLLMs like Qwen2.5-VL and InternVL3 on question-answering benchmarks while adding advanced visual prompt understanding and dense object segmentation capabilities.

### Comparisons and Uses
This Sa2VA node can be thought of as a more advanced version of [neverbiasu's ComfyUI-SAM2 node](https://github.com/neverbiasu/ComfyUI-SAM2) that allows for segmentation of objects in an image using natural langauge. Unlike that node which is based on [Grounded SAM/Grounding DINO](https://github.com/IDEA-Research/Grounded-SAM-2), Sa2VA uses a full VLLM trained to output SAM2 segmentation masks, which means it can handle significantly longer and more descriptive text. This allows Sa2VA to be better for uses cases where simple phrases like "woman on right" isn't sufficient to completely disambiguate between objects.

It outperforms Grounding DINO on short prompts:
![](https://raw.githubusercontent.com/adambarbato/ComfyUI-Sa2VA/refs/heads/main/docs/sa2va-grounding-dino.jpg)

And can follow longer instructions quite well, such as describing a character in general, rather than their position or traits in the image itself. This lends itself well to auto-generated or agentic segmentation prompts:
![](https://raw.githubusercontent.com/adambarbato/ComfyUI-Sa2VA/refs/heads/main/docs/long-prompt.jpg)

It can also segment more than one mask at a time, but the prompt needs to be precise:
![](https://raw.githubusercontent.com/adambarbato/ComfyUI-Sa2VA/refs/heads/main/docs/multi-mask.jpg)

## Installation

### Manual Installation
```bash
cd ComfyUI/custom_nodesgit clone https://github.com/adambarbato/ComfyUI-Sa2VA.git
cd ComfyUI-Sa2VA
python install.py
```

### Quick Install (Advanced Users)
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/adambarbato/ComfyUI-Sa2VA.git
cd ComfyUI-Sa2VA
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- PyTorch 2.0+
- **transformers >= 4.57.0** (Critical!)
- CUDA 11.8+ (for GPU acceleration)
- 8GB+ VRAM recommended for 4B models
- 20GB+ VRAM for full precision

**Important:** Sa2VA models require:
- transformers >= 4.57.0 (for Qwen3-VL support)
- qwen_vl_utils (for model utilities)

Older transformers versions will fail with "No module named 'transformers.models.qwen3_vl'" error.

## Features

### 🎯 **Core Capabilities**
- **Multimodal Understanding**: Combines text generation with visual understanding
- **Dense Segmentation**: Pixel-perfect object segmentation masks
- **Video Processing**: Temporal understanding across video sequences
- **Visual Prompts**: Understanding of spatial relationships and object references
- **Integrated Mask Conversion**: Built-in conversion to ComfyUI mask and image formats
- **Cancellable, Real-time Downloads**: Large model downloads show progress and respect ComfyUI's cancel button

### 🚀 **Supported Models**
- `ByteDance/Sa2VA-Qwen3-VL-4B` (recommended - 4B parameters)
- `ByteDance/Sa2VA-Qwen2_5-VL-7B` (7B parameters)
- `ByteDance/Sa2VA-InternVL3-8B` (8B parameters)
- `ByteDance/Sa2VA-InternVL3-14B` (14B parameters)
- `ByteDance/Sa2VA-Qwen2_5-VL-3B` (3B parameters)
- `ByteDance/Sa2VA-InternVL3-2B` (2B parameters)

## Node

### **Sa2VA Segmentation**
A single, comprehensive node that provides:
- **Text Generation**: Detailed image descriptions and analysis
- **Object Segmentation**: Precise pixel-level object masks
- **Integrated Mask Conversion**: Automatic conversion to ComfyUI MASK and IMAGE formats
- **Memory Management**: Built-in VRAM optimization and model lifecycle management
- **Multiple Output Formats**: Text, ComfyUI masks, and visualizable mask images

## Quick Start

### Basic Image Description
1. Add **Load Image** node and load your image
2. Add **Sa2VA Segmentation** node
3. Connect Load Image → Sa2VA node
4. Adjust `model_name` and `mask_threshold` as needed
5. Set `segmentation_prompt`: "Please describe the image in detail."
6. Execute to get text descriptions

### Image Segmentation
1. Load image using **Load Image** node
2. Add **Sa2VA Segmentation** node
3. Connect Load Image → Sa2VA node
4. Adjust `model_name` and `mask_threshold` as needed
5. Set `segmentation_prompt`: "Please provide segmentation masks for all objects."
6. Connect the `masks` output to mask-compatible nodes or `mask_images` to **Preview Image**
7. The node automatically provides both MASK tensors and visualizable IMAGE tensors

## Model Precision

Sa2VA models use **bfloat16 precision** by default with the option to quantize to 8 bits using bits-and-bytes.


## Troubleshooting

### Common Issues

**"No module named 'transformers.models.qwen3_vl'"**
```bash
pip install transformers>=4.57.0 --upgrade
```
This is the most common issue - your transformers version is too old.

**"No module named 'qwen_vl_utils'"**
```bash
pip install qwen_vl_utils
```
This dependency is required for Sa2VA model utilities.

**"'NoneType' object is not subscriptable"**
- Model loading failed (check console for errors)
- Usually caused by outdated transformers version
- Verify internet connection for model download

**CUDA Out of Memory**
- Use 8 bit quantization
- Use smaller model variant (2B or 3B parameters)
- Reduce batch size

**Model Loading Errors**
- Check internet connection for initial download
- Ensure sufficient disk space (20GB+ per model)
- Verify CUDA compatibility
- Try: `torch.cuda.empty_cache()` to clear VRAM

**Poor Segmentation Quality**
- Use more specific prompts: "Please segment the person"
- Try different model variants
- Adjust threshold in Mask Converter
- Use higher resolution images

For detailed troubleshooting, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## Advanced Configuration

### Custom Prompts
```python
# Object-specific segmentation
"Please segment the person in the image"
"Identify and segment all vehicles"

# Multi-object segmentation
"Create separate masks for all distinct objects"
"Segment foreground and background separately"
```

## Contributing

Contributions welcome! Areas for improvement:
- Performance optimizations
- Video processing
- Better mask post-processing

## License

MIT

## Testing Installation

Run the test script to verify everything works:
```bash
cd ComfyUI-Sa2VA
python test_sa2va.py
```

## Links

- [Sa2VA Paper](https://arxiv.org/abs/2501.04001)
- [Sa2VA Models on HuggingFace](https://huggingface.co/ByteDance)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- Based on code from [ComfyUI-Transformers-Pipeline](https://github.com/mediocreatmybest/ComfyUI-Transformers-Pipeline)
- [Issues & Support](https://github.com/adambarbato/ComfyUI-Sa2VA/issues)
- [Troubleshooting Guide](TROUBLESHOOTING.md)
