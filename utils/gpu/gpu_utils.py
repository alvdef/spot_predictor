#!/usr/bin/env python3
"""
GPU utilities for optimizing model training on G4DN instances.

This module provides functions to detect, configure, and optimize GPU usage
for deep learning models running on AWS G4DN instances with NVIDIA T4 GPUs.
"""

import os
import logging
import sys
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.cuda
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed, GPU optimizations will not be available")


def check_gpu_availability() -> Dict[str, Any]:
    """
    Check if GPU is available and collect information about it.
    
    Returns:
        Dict containing GPU information or error message if unavailable
    """
    result = {
        "gpu_available": False,
        "cuda_available": False,
        "device_count": 0,
        "gpu_info": [],
        "error": None
    }
    
    if not TORCH_AVAILABLE:
        result["error"] = "PyTorch not installed"
        return result
    
    try:
        result["cuda_available"] = torch.cuda.is_available()
        if result["cuda_available"]:
            result["gpu_available"] = True
            result["device_count"] = torch.cuda.device_count()
            
            # Collect information about each GPU
            for i in range(result["device_count"]):
                gpu_info = {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_total": torch.cuda.get_device_properties(i).total_memory / (1024**3),  # GB
                }
                result["gpu_info"].append(gpu_info)
    except Exception as e:
        result["error"] = str(e)
    
    return result


def setup_gpu_environment() -> None:
    """
    Configure environment variables and PyTorch settings for optimal GPU performance.
    """
    if not TORCH_AVAILABLE:
        return
    
    # Set environment variables for better GPU performance
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Match IDs with nvidia-smi
    
    # Attempt to use TF32 precision if available (NVIDIA Ampere+ GPUs)
    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda, "matmul"):
        torch.cuda.matmul.allow_tf32 = True
    
    if hasattr(torch.backends, "cudnn"):
        # Enable cuDNN auto-tuner
        torch.backends.cudnn.benchmark = True
        # Use deterministic algorithms for reproducibility if needed
        # torch.backends.cudnn.deterministic = True
    
    logger.info("GPU environment configured for optimal performance")


def get_cuda_device(device_id: Optional[int] = None) -> torch.device:
    """
    Get the appropriate torch device to use.
    
    Args:
        device_id: Specific GPU device ID to use, or None for auto-selection
    
    Returns:
        torch.device: The appropriate device (GPU or CPU)
    """
    if not TORCH_AVAILABLE:
        return "cpu"
    
    if torch.cuda.is_available():
        if device_id is not None and device_id < torch.cuda.device_count():
            return torch.device(f"cuda:{device_id}")
        else:
            return torch.device("cuda")
    else:
        return torch.device("cpu")


def log_gpu_info() -> None:
    """
    Log detailed information about available GPUs.
    """
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        logger.info("No GPU available for logging")
        return
    
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        logger.info(f"GPU {i}: {props.name}")
        logger.info(f"  - Memory: {props.total_memory / (1024**3):.2f} GB")
        logger.info(f"  - Compute capability: {props.major}.{props.minor}")
        if hasattr(props, "multi_processor_count"):
            logger.info(f"  - SM count: {props.multi_processor_count}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run checks and setup when script is executed directly
    gpu_info = check_gpu_availability()
    if gpu_info["gpu_available"]:
        logger.info(f"GPU is available: {gpu_info['device_count']} device(s) found")
        for gpu in gpu_info["gpu_info"]:
            logger.info(f"GPU {gpu['index']}: {gpu['name']} ({gpu['memory_total']:.2f} GB)")
        
        setup_gpu_environment()
        log_gpu_info()
    else:
        logger.warning(f"No GPU available: {gpu_info['error']}")
        logger.warning("Performance will be limited. Consider using a G4DN instance with GPU.")