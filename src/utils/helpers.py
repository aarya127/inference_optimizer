"""Utility functions for the inference optimizer."""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional


def load_json(filepath: str) -> Dict[str, Any]:
    """Load JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Parsed JSON data
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], filepath: str, indent: int = 2):
    """Save data as JSON.
    
    Args:
        data: Data to save
        filepath: Output path
        indent: JSON indentation
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent)


def ensure_dir(dirpath: str):
    """Ensure directory exists.
    
    Args:
        dirpath: Directory path
    """
    Path(dirpath).mkdir(parents=True, exist_ok=True)


def get_model_size(model_name: str) -> Optional[str]:
    """Extract model size from model name.
    
    Args:
        model_name: Model name
        
    Returns:
        Model size (e.g., '7b', '13b') or None
    """
    import re
    
    # Common patterns for model sizes
    patterns = [
        r'(\d+\.?\d*[bB])',  # 7b, 13b, 70b
        r'(\d+[mM])',        # 125m, 350m
    ]
    
    for pattern in patterns:
        match = re.search(pattern, model_name)
        if match:
            return match.group(1).lower()
    
    return None


def format_size(size_bytes: float) -> str:
    """Format bytes to human-readable size.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string (e.g., '1.5 GB')
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def format_number(num: float, precision: int = 2) -> str:
    """Format number with appropriate precision.
    
    Args:
        num: Number to format
        precision: Decimal precision
        
    Returns:
        Formatted string
    """
    if num >= 1000000:
        return f"{num / 1000000:.{precision}f}M"
    elif num >= 1000:
        return f"{num / 1000:.{precision}f}K"
    else:
        return f"{num:.{precision}f}"


def merge_results(result_files: List[str]) -> List[Dict[str, Any]]:
    """Merge multiple result files.
    
    Args:
        result_files: List of result file paths
        
    Returns:
        Combined results
    """
    all_results = []
    
    for filepath in result_files:
        data = load_json(filepath)
        if isinstance(data, list):
            all_results.extend(data)
        else:
            all_results.append(data)
    
    return all_results


def calculate_speedup(baseline: float, optimized: float) -> float:
    """Calculate speedup factor.
    
    Args:
        baseline: Baseline metric value
        optimized: Optimized metric value
        
    Returns:
        Speedup factor
    """
    if baseline == 0:
        return 0.0
    return baseline / optimized


def calculate_memory_savings(baseline_mb: float, optimized_mb: float) -> float:
    """Calculate memory savings percentage.
    
    Args:
        baseline_mb: Baseline memory in MB
        optimized_mb: Optimized memory in MB
        
    Returns:
        Percentage savings
    """
    if baseline_mb == 0:
        return 0.0
    return ((baseline_mb - optimized_mb) / baseline_mb) * 100
