import time
import psutil
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple


# Determine the available device
def get_device():
    """Return the best available device (MPS for Mac, CUDA for NVIDIA, CPU otherwise)"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def get_memory_info(device: torch.device) -> Tuple[float, float]:
    """Get memory usage for the specified device

    Returns:
        Tuple of (memory_used_mb, total_memory_mb)
    """
    if device.type == "cuda":
        memory_allocated = torch.cuda.memory_allocated(device) / 1024 / 1024  # MB
        max_memory = (
            torch.cuda.get_device_properties(device).total_memory / 1024 / 1024
        )  # MB
        return memory_allocated, max_memory
    elif device.type == "mps":
        # MPS doesn't expose memory stats, so we use psutil as approximation
        memory_stats = psutil.virtual_memory()
        return memory_stats.used / 1024 / 1024, memory_stats.total / 1024 / 1024  # MB
    else:
        # For CPU
        memory_stats = psutil.virtual_memory()
        return memory_stats.used / 1024 / 1024, memory_stats.total / 1024 / 1024  # MB


def profile_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_sizes: List[int] = [32, 64, 128, 256],
    num_workers_list: List[int] = [0, 1, 2, 4],
) -> Dict[str, Dict[str, float]]:
    """Profile DataLoader performance with different batch sizes and worker counts.

    Using multiple workers with MPS or CUDA devices requires special handling on macOS.
    This function tests different configurations and measures their performance.

    Args:
        dataset: The PyTorch dataset to profile
        batch_sizes: List of batch sizes to test
        num_workers_list: List of worker counts to test

    Returns:
        Dictionary of performance metrics for each configuration
    """
    results = {}

    # Use only 0 or 1 worker on macOS due to multiprocessing limitations
    if torch.backends.mps.is_available():
        num_workers_list = [w for w in num_workers_list if w <= 1]
        if not num_workers_list:
            num_workers_list = [0]

    for batch_size in batch_sizes:
        for num_workers in num_workers_list:
            key = f"batch_size={batch_size}, workers={num_workers}"

            # Configure DataLoader - no multiprocessing context needed for num_workers=0
            loader_kwargs = {
                "batch_size": batch_size,
                "num_workers": num_workers,
                "shuffle": True,
                "pin_memory": num_workers > 0,
            }

            loader = DataLoader(dataset, **loader_kwargs)

            # Initialize metrics
            batch_times = []
            memory_usage = []
            device_memory = []
            num_batches = min(5, len(loader))

            try:
                # Warmup to cache data
                loader_iter = iter(loader)
                _ = next(loader_iter)

                # Profile loading time
                start_time = time.time()
                samples_processed = 0

                for _ in range(num_batches):
                    batch_start = time.time()
                    batch = next(loader_iter)
                    batch_end = time.time()

                    # Record metrics
                    batch_times.append(batch_end - batch_start)
                    memory_usage.append(
                        psutil.Process().memory_info().rss / 1024 / 1024
                    )  # MB

                    # Get device memory if available
                    device = get_device()
                    if device.type != "cpu":
                        device_mem, _ = get_memory_info(device)
                        device_memory.append(device_mem)

                    # Count samples
                    if isinstance(batch, (tuple, list)):
                        samples_processed += len(batch[0])
                    else:
                        samples_processed += len(batch)

                total_time = time.time() - start_time

                # Calculate metrics
                results[key] = {
                    "avg_batch_time": sum(batch_times) / len(batch_times),
                    "throughput": samples_processed / total_time,
                    "avg_memory_mb": sum(memory_usage) / len(memory_usage),
                    "avg_device_mb": (
                        sum(device_memory) / len(device_memory) if device_memory else 0
                    ),
                }

            except Exception as e:
                print(f"Error profiling {key}: {str(e)}")
                results[key] = {
                    "avg_batch_time": float("inf"),
                    "throughput": 0,
                    "avg_memory_mb": 0,
                    "avg_device_mb": 0,
                }
            finally:
                # Clean up resources
                del loader
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    return results


def find_optimal_batch_size(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    start_size: int = 32,
    max_size: int = 512,
    step_factor: int = 2,
    target_memory_usage: float = 0.8,  # 80% of available memory
) -> int:
    """
    Find the optimal batch size that maximizes GPU/MPS utilization without OOM.

    Works with CUDA, MPS (Apple Silicon), and falls back to CPU if needed.

    Args:
        model: PyTorch model
        dataset: Dataset to test with
        start_size: Initial batch size
        max_size: Maximum batch size to try
        step_factor: Multiply batch size by this factor each iteration
        target_memory_usage: Target memory usage (0.0 to 1.0)
    """
    device = get_device()
    if device.type == "cpu":
        return start_size

    batch_size = start_size
    optimal_batch = start_size

    model = model.to(device).train()

    while batch_size <= max_size:
        try:
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            batch = next(iter(loader))

            # Move to appropriate device
            if isinstance(batch, (tuple, list)):
                data, target = [b.to(device) for b in batch]
            else:
                data = batch.to(device)
                target = data  # Placeholder if no target

            # Forward pass
            with torch.autograd.detect_anomaly():
                output = model(data)

                # Construct a loss for backprop
                if hasattr(model, "criterion"):
                    loss = model.criterion(output, target)
                else:
                    loss = torch.nn.functional.mse_loss(output, target)

                # Backprop
                loss.backward()

            # Check memory usage
            mem_used, mem_total = get_memory_info(device)
            memory_ratio = mem_used / mem_total

            if memory_ratio < target_memory_usage:
                optimal_batch = batch_size
                batch_size *= step_factor
            else:
                break

            # Clean up
            if device.type == "cuda":
                torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"OOM at batch size {batch_size}, reverting to {optimal_batch}")
                break
            else:
                print(f"Error during batch size search: {str(e)}")
                break

    return optimal_batch
