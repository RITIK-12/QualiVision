"""
Memory management utilities for QualiVision.

This module provides GPU memory management and optimization utilities.
"""

import gc
import torch
from typing import Optional


def ultra_memory_cleanup():
    """Aggressive GPU memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_gpu_memory_info() -> dict:
    """Get current GPU memory usage information."""
    if not torch.cuda.is_available():
        return {'error': 'CUDA not available'}
    
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    max_allocated = torch.cuda.max_memory_allocated()
    max_reserved = torch.cuda.max_memory_reserved()
    
    return {
        'allocated_gb': allocated / 1e9,
        'reserved_gb': reserved / 1e9,
        'max_allocated_gb': max_allocated / 1e9,
        'max_reserved_gb': max_reserved / 1e9,
        'free_gb': (torch.cuda.get_device_properties(0).total_memory - allocated) / 1e9
    }


def print_gpu_memory():
    """Print current GPU memory usage."""
    info = get_gpu_memory_info()
    if 'error' in info:
        print(info['error'])
        return
    
    print(f"GPU Memory - Allocated: {info['allocated_gb']:.1f}GB, "
          f"Free: {info['free_gb']:.1f}GB, "
          f"Max Used: {info['max_allocated_gb']:.1f}GB")


class MemoryMonitor:
    """Context manager for monitoring memory usage."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_allocated = 0
        self.start_reserved = 0
    
    def __enter__(self):
        if torch.cuda.is_available():
            self.start_allocated = torch.cuda.memory_allocated()
            self.start_reserved = torch.cuda.memory_reserved()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            end_allocated = torch.cuda.memory_allocated()
            end_reserved = torch.cuda.memory_reserved()
            
            allocated_diff = (end_allocated - self.start_allocated) / 1e9
            reserved_diff = (end_reserved - self.start_reserved) / 1e9
            
            print(f"{self.name} - Memory change: "
                  f"Allocated: {allocated_diff:+.2f}GB, "
                  f"Reserved: {reserved_diff:+.2f}GB")


def cleanup_on_oom(func):
    """Decorator to cleanup memory on OOM and retry once."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("OOM detected, cleaning up memory and retrying...")
                ultra_memory_cleanup()
                return func(*args, **kwargs)
            else:
                raise e
    return wrapper 