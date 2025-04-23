#!/usr/bin/env python3
"""
GPU utilities for optimizing model training on G4DN instances.

This module provides functions to detect, configure, and optimize GPU usage
for deep learning models running on AWS G4DN instances with NVIDIA T4 GPUs.
Also supports Apple Silicon GPUs via Metal Performance Shaders (MPS).
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
    # Check if MPS (Metal Performance Shaders) is available for Apple Silicon
    MPS_AVAILABLE = hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    MPS_AVAILABLE = False
    logger.warning("PyTorch not installed, GPU optimizations will not be available")


def check_gpu_availability() -> Dict[str, Any]:
    """
    Check if GPU is available and collect information about it.
    Supports both NVIDIA CUDA GPUs and Apple Silicon via MPS.
    
    Returns:
        Dict containing GPU information or error message if unavailable
    """
    result = {
        "gpu_available": False,
        "cuda_available": False,
        "mps_available": False,
        "device_count": 0,
        "gpu_info": [],
        "error": None
    }
    
    if not TORCH_AVAILABLE:
        result["error"] = "PyTorch not installed"
        return result
    
    try:
        # Check for CUDA (NVIDIA) GPUs
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
                    "type": "cuda"
                }
                result["gpu_info"].append(gpu_info)
        # Check for MPS (Apple Silicon)
        elif MPS_AVAILABLE:
            result["mps_available"] = True
            result["gpu_available"] = True
            result["device_count"] = 1  # MPS typically has just one GPU
            
            # Add Apple Silicon GPU info
            gpu_info = {
                "index": 0,
                "name": "Apple Silicon",  # Generic name as PyTorch doesn't provide specific model info
                "memory_total": 0,  # Memory information not directly available through MPS
                "type": "mps"
            }
            result["gpu_info"].append(gpu_info)
    except Exception as e:
        result["error"] = str(e)
    
    return result


def setup_gpu_environment() -> None:
    """
    Configure environment variables and PyTorch settings for optimal GPU performance.
    Supports both CUDA and MPS backends.
    """
    if not TORCH_AVAILABLE:
        return
    
    if torch.cuda.is_available():
        # Set environment variables for better CUDA performance
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Match IDs with nvidia-smi
        
        # Attempt to use TF32 precision if available (NVIDIA Ampere+ GPUs)
        if hasattr(torch.cuda, "amp") and hasattr(torch.cuda, "matmul"):
            torch.cuda.matmul.allow_tf32 = True
        
        if hasattr(torch.backends, "cudnn"):
            # Enable cuDNN auto-tuner
            torch.backends.cudnn.benchmark = True
            # Use deterministic algorithms for reproducibility if needed
            # torch.backends.cudnn.deterministic = True
        
        logger.info("CUDA GPU environment configured for optimal performance")
    elif MPS_AVAILABLE:
        # MPS-specific optimizations (limited compared to CUDA)
        # Apple Silicon-specific settings could be added here as they become available
        logger.info("Apple Silicon MPS environment configured for optimal performance")
    
    # No else needed here, as the function will just return if no GPU is available


def get_cuda_device(device_id: Optional[int] = None) -> torch.device:
    """
    Get the appropriate torch device to use.
    Prioritizes: 1) CUDA (if available with specified ID), 2) MPS, 3) CPU
    
    Args:
        device_id: Specific GPU device ID to use, or None for auto-selection
    
    Returns:
        torch.device: The appropriate device (CUDA GPU, MPS, or CPU)
    """
    if not TORCH_AVAILABLE:
        return "cpu"
    
    if torch.cuda.is_available():
        if device_id is not None and device_id < torch.cuda.device_count():
            return torch.device(f"cuda:{device_id}")
        else:
            return torch.device("cuda")
    elif MPS_AVAILABLE:
        return torch.device("mps")
    else:
        return torch.device("cpu")


def log_gpu_info() -> None:
    """
    Log detailed information about available GPUs.
    Supports both CUDA and MPS backends.
    """
    if not TORCH_AVAILABLE:
        logger.info("No GPU available for logging: PyTorch not installed")
        return
    
    if torch.cuda.is_available():
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
    elif MPS_AVAILABLE:
        logger.info("Apple Silicon GPU (MPS) is available")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info("MPS device: Apple Silicon")
        # MPS doesn't provide detailed GPU properties like CUDA
    else:
        logger.info("No GPU available for logging: Neither CUDA nor MPS is available")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run checks and setup when script is executed directly
    gpu_info = check_gpu_availability()
    if gpu_info["gpu_available"]:
        if gpu_info["cuda_available"]:
            logger.info(f"CUDA GPU is available: {gpu_info['device_count']} device(s) found")
            for gpu in gpu_info["gpu_info"]:
                logger.info(f"GPU {gpu['index']}: {gpu['name']} ({gpu['memory_total']:.2f} GB)")
        elif gpu_info["mps_available"]:
            logger.info("Apple Silicon GPU (MPS) is available")
        
        setup_gpu_environment()
        log_gpu_info()
    else:
        logger.warning(f"No GPU available: {gpu_info['error']}")
        logger.warning("Performance will be limited. Consider using a G4DN instance with NVIDIA GPU or a Mac with Apple Silicon.")