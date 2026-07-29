"""
4-Bit Quantization Framework for MLX

NOTE: this module is not imported by the live service — it is standalone
scaffolding targeting 7B LLaVA-class models, not the SmolVLM checkpoint the
study actually uses (which ships pre-quantized).

Implements group quantization math (quantize/dequantize round trip) and a
size ESTIMATOR.  Honesty notes:
- `estimate_quantization_savings` (formerly `quantize_model`) does NOT
  modify any weights — it only computes projected sizes.
- Quantized values are stored as uint8 (1 byte/weight), not packed int4
  (0.5 byte/weight); in-memory footprint is therefore 2x the packed-int4
  figure used in the size estimates.
- `quantize_llava_model` is unimplemented and raises NotImplementedError
  (a previous version returned fabricated quality metrics).

Supports:
- Group quantization (group_size=64)
- Selective layer quantization
- Quality validation (compares two supplied models on test inputs)
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class QuantizationConfig:
    """Configuration for quantization"""
    bits: int = 4
    group_size: int = 64
    calibration_samples: int = 128
    calibration_method: str = "minmax"  # minmax, mse, or percentile
    exclude_patterns: List[str] = None
    
    def __post_init__(self):
        if self.exclude_patterns is None:
            # Default: don't quantize vision encoder, embeddings, or output heads
            self.exclude_patterns = [
                "vision_tower.*",
                "multi_modal_projector.*",
                "embed_tokens",
                "lm_head",
                "layernorm",
                "norm"
            ]


@dataclass
class QuantizationResult:
    """Result of quantization operation"""
    original_size_mb: float
    quantized_size_mb: float
    compression_ratio: float
    num_quantized_layers: int
    num_excluded_layers: int
    quality_metrics: Dict[str, float]


class GroupQuantizer:
    """Group-based quantization for weights"""
    
    def __init__(self, bits: int = 4, group_size: int = 64):
        self.bits = bits
        self.group_size = group_size
        self.n_levels = 2 ** bits
        self.scale_dtype = mx.float16
        
    def quantize_weights(
        self,
        weights: mx.array
    ) -> Tuple[mx.array, dict]:
        """
        Quantize weights using group quantization

        Args:
            weights: Weight tensor of shape [out_features, in_features]

        Returns:
            Tuple of (quantized_weights, metadata) where quantized_weights is
            a uint8 array (NOT packed int4 — 1 byte per weight in memory) and
            metadata is a dict with keys 'scale', 'zero', 'original_shape',
            'padded', 'pad_size' for use with dequantize_weights().
        """
        shape = weights.shape
        padded = False
        pad_size = 0
        
        # Reshape to groups
        # [out_features, in_features] -> [out_features, num_groups, group_size]
        if shape[-1] % self.group_size != 0:
            # Pad last dimension to multiple of group_size
            pad_size = self.group_size - (shape[-1] % self.group_size)
            weights = mx.pad(weights, [(0, 0), (0, pad_size)])
            padded = True
        
        padded_shape = weights.shape
        reshaped = weights.reshape(*shape[:-1], -1, self.group_size)
        
        # Compute min/max per group
        group_min = mx.min(reshaped, axis=-1, keepdims=True)
        group_max = mx.max(reshaped, axis=-1, keepdims=True)
        
        # Compute scale and zero point.  Epsilon floor prevents divide-by-
        # zero (NaN/inf) for constant groups where group_max == group_min.
        eps = 1e-8
        scale = mx.maximum((group_max - group_min) / (self.n_levels - 1), eps)
        zero = group_min

        # Quantize
        quantized = mx.round((reshaped - zero) / scale)
        quantized = mx.clip(quantized, 0, self.n_levels - 1)
        
        # Pack to int4 (store as int8 for now, will pack later)
        quantized = quantized.astype(mx.uint8)
        
        # Reshape back to padded shape
        quantized = quantized.reshape(padded_shape)
        scale = scale.reshape(*shape[:-1], -1)
        zero = zero.reshape(*shape[:-1], -1)
        
        # Store original shape for later unpacking
        metadata = {
            'scale': scale.astype(self.scale_dtype),
            'zero': zero.astype(self.scale_dtype),
            'original_shape': shape,
            'padded': padded,
            'pad_size': pad_size
        }
        
        return quantized, metadata
    
    def dequantize_weights(
        self,
        quantized: mx.array,
        metadata: dict
    ) -> mx.array:
        """
        Dequantize weights back to FP16
        
        Args:
            quantized: Quantized weights
            metadata: Dictionary with 'scale', 'zero', 'original_shape', 'padded', 'pad_size'
            
        Returns:
            Dequantized FP16 weights
        """
        scales = metadata['scale']
        zeros = metadata['zero']
        original_shape = metadata['original_shape']
        padded = metadata.get('padded', False)
        pad_size = metadata.get('pad_size', 0)
        
        shape = quantized.shape
        
        # Reshape to groups
        reshaped = quantized.reshape(*shape[:-1], -1, self.group_size)
        
        # Expand scales and zeros
        scales_expanded = mx.expand_dims(scales, axis=-1)
        zeros_expanded = mx.expand_dims(zeros, axis=-1)
        
        # Dequantize
        dequantized = reshaped.astype(mx.float16) * scales_expanded + zeros_expanded
        
        # Reshape back
        dequantized = dequantized.reshape(shape)
        
        # Remove padding if it was added
        if padded and pad_size > 0:
            dequantized = dequantized[..., :-pad_size]
        
        return dequantized


class QuantizedLinear(nn.Module):
    """Quantized linear layer"""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        quantizer: GroupQuantizer,
        bias: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quantizer = quantizer
        
        # Initialize with placeholder weights
        # In practice, these will be loaded from pre-quantized checkpoint
        self.weight_quantized = mx.zeros((out_features, in_features), dtype=mx.uint8)
        self.weight_scales = mx.ones((out_features, in_features // quantizer.group_size), 
                                     dtype=mx.float16)
        self.weight_zeros = mx.zeros((out_features, in_features // quantizer.group_size),
                                     dtype=mx.float16)
        
        if bias:
            self.bias = mx.zeros((out_features,), dtype=mx.float16)
        else:
            self.bias = None
    
    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with on-the-fly dequantization"""
        # Dequantize weights
        metadata = {
            'scale': self.weight_scales,
            'zero': self.weight_zeros,
            'original_shape': self.weight_quantized.shape,
            'padded': False,
            'pad_size': 0
        }
        weight_fp16 = self.quantizer.dequantize_weights(
            self.weight_quantized,
            metadata
        )
        
        # Standard linear operation
        output = mx.matmul(x, weight_fp16.T)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


class ModelQuantizer:
    """Quantize entire models with selective layer quantization"""
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.quantizer = GroupQuantizer(
            bits=config.bits,
            group_size=config.group_size
        )
        
    def should_quantize_layer(self, layer_name: str) -> bool:
        """Check if layer should be quantized based on exclusion patterns"""
        import re
        
        for pattern in self.config.exclude_patterns:
            if re.match(pattern, layer_name):
                return False
        return True
    
    def estimate_quantization_savings(
        self,
        model: nn.Module,
        calibration_data: Optional[List[mx.array]] = None
    ) -> Tuple[nn.Module, QuantizationResult]:
        """
        ESTIMATE the size savings of quantizing this model.

        HONESTY NOTE — NO WEIGHTS ARE MODIFIED.  This method runs the
        quantization math per layer to exercise it, but the model is
        returned UNCHANGED; the QuantizationResult contains projected
        sizes only.  (This method was previously named `quantize_model`,
        which computed quantized weights into locals, discarded them, and
        returned the unmodified model while reporting "compression" stats
        as if applied.)

        Size accounting assumes PACKED int4 storage (0.5 byte/weight).
        This framework actually stores quantized values as uint8
        (1 byte/weight), so the real in-memory ratio is 2x worse than the
        estimate until packing is implemented.

        Args:
            model: MLX model to analyze
            calibration_data: Unused (kept for signature compatibility)

        Returns:
            Tuple of (model UNCHANGED, QuantizationResult with projected sizes)
        """
        print(f"Estimating {self.config.bits}-bit quantization savings "
              f"(group_size={self.config.group_size}) — no weights are modified")

        original_size = 0
        quantized_size = 0
        num_quantized = 0
        num_excluded = 0

        # Traverse model layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Check if should quantize
                if self.should_quantize_layer(name):
                    # Get original weights
                    weights = module.weight
                    original_size += weights.size * 2  # FP16 = 2 bytes

                    # Run the quantization math (result discarded — estimate only)
                    q_weights, q_meta = self.quantizer.quantize_weights(weights)
                    scales = q_meta['scale']
                    zeros = q_meta['zero']

                    # Projected size ASSUMING packed int4 (0.5 B/weight).
                    # Actual uint8 storage in this framework is 1 B/weight.
                    quantized_size += q_weights.size * 0.5
                    quantized_size += (scales.size + zeros.size) * 2  # FP16 scales/zeros

                    num_quantized += 1

                    print(f"  Would quantize: {name}")
                else:
                    # Keep in FP16
                    weights = module.weight
                    original_size += weights.size * 2
                    quantized_size += weights.size * 2
                    num_excluded += 1

                    print(f"  Excluded: {name}")

        # Convert to MB
        original_size_mb = original_size / (1024 ** 2)
        quantized_size_mb = quantized_size / (1024 ** 2)
        compression_ratio = (
            original_size_mb / quantized_size_mb if quantized_size_mb > 0 else 1.0
        )

        result = QuantizationResult(
            original_size_mb=original_size_mb,
            quantized_size_mb=quantized_size_mb,
            compression_ratio=compression_ratio,
            num_quantized_layers=num_quantized,
            num_excluded_layers=num_excluded,
            quality_metrics={}
        )

        print(f"\nQuantization Savings Estimate (no weights modified):")
        print(f"  Original size: {original_size_mb:.1f} MB")
        print(f"  Projected size: {quantized_size_mb:.1f} MB "
              f"(assumes packed int4; uint8 storage would be ~2x)")
        print(f"  Projected compression: {compression_ratio:.2f}x")
        print(f"  Layers that would be quantized: {num_quantized}")
        print(f"  Excluded layers: {num_excluded}")

        return model, result
    
    def validate_quantization(
        self,
        original_model: nn.Module,
        quantized_model: nn.Module,
        test_inputs: List[mx.array]
    ) -> Dict[str, float]:
        """
        Validate quantization quality
        
        Args:
            original_model: Original FP16 model
            quantized_model: Quantized model
            test_inputs: List of test inputs
            
        Returns:
            Dictionary of quality metrics
        """
        print("Validating quantization quality...")
        
        metrics = {
            'mean_squared_error': 0.0,
            'max_absolute_error': 0.0,
            'cosine_similarity': 0.0
        }
        
        # Compare outputs on test inputs.
        # Note: MLX has no `mx.no_grad()` context manager (gradients are only
        # computed via explicit transforms like mx.grad), so plain forward
        # calls are already inference-only.
        for i, test_input in enumerate(test_inputs[:self.config.calibration_samples]):
            # Forward pass
            orig_output = original_model(test_input)
            quant_output = quantized_model(test_input)

            # Compute metrics
            mse = mx.mean((orig_output - quant_output) ** 2).item()
            mae = mx.max(mx.abs(orig_output - quant_output)).item()
            
            # Cosine similarity
            orig_norm = mx.sqrt(mx.sum(orig_output ** 2))
            quant_norm = mx.sqrt(mx.sum(quant_output ** 2))
            cos_sim = mx.sum(orig_output * quant_output) / (orig_norm * quant_norm)
            
            metrics['mean_squared_error'] += mse
            metrics['max_absolute_error'] = max(metrics['max_absolute_error'], mae)
            metrics['cosine_similarity'] += cos_sim.item()
        
        # Average metrics
        n = min(len(test_inputs), self.config.calibration_samples)
        metrics['mean_squared_error'] /= n
        metrics['cosine_similarity'] /= n
        
        print(f"  MSE: {metrics['mean_squared_error']:.6f}")
        print(f"  Max AE: {metrics['max_absolute_error']:.6f}")
        print(f"  Cosine Similarity: {metrics['cosine_similarity']:.4f}")
        
        return metrics


def quantize_llava_model(
    model_path: str,
    output_path: str,
    bits: int = 4,
    group_size: int = 64
) -> QuantizationResult:
    """
    Convenience function to quantize a LLaVA model — NOT IMPLEMENTED.

    Model loading/saving was never implemented, and a previous version of
    this function returned fully fabricated quality metrics (MSE 0.0012,
    cosine 0.998) as if measured.  It now raises instead of inventing
    numbers.

    Raises:
        NotImplementedError: always — no model loading exists in this repo,
            and no quality metrics have ever been measured for this path.
    """
    raise NotImplementedError(
        "quantize_llava_model is unimplemented scaffolding: model "
        "loading/saving does not exist in this repo and no quality metrics "
        "have been measured. Use GroupQuantizer directly on real weight "
        "arrays, or ModelQuantizer.estimate_quantization_savings() for a "
        "size projection."
    )


if __name__ == "__main__":
    """Test quantization framework"""
    print("=" * 80)
    print("AMIO Phase 0 - Quantization Framework Test")
    print("=" * 80)
    print()
    
    # Test group quantizer
    print("Testing Group Quantizer:")
    print("-" * 80)
    
    quantizer = GroupQuantizer(bits=4, group_size=64)
    
    # Create test weights
    test_weights = mx.random.normal((4096, 4096))
    print(f"Original weights shape: {test_weights.shape}")
    print(f"Original size: {test_weights.size * 2 / (1024**2):.1f} MB (FP16)")
    
    # Quantize
    q_weights, q_meta = quantizer.quantize_weights(test_weights)
    scales = q_meta['scale']
    zeros = q_meta['zero']
    print(f"Quantized weights shape: {q_weights.shape} (dtype={q_weights.dtype})")
    # Size accounting: values are STORED as uint8 (1 B/weight); the packed
    # int4 figure (0.5 B/weight) is a projection, not the actual footprint.
    stored_size = (q_weights.size * 1.0 + (scales.size + zeros.size) * 2) / (1024**2)
    packed_size = (q_weights.size * 0.5 + (scales.size + zeros.size) * 2) / (1024**2)
    print(f"Stored size (uint8 + FP16 scales): {stored_size:.1f} MB")
    print(f"Projected size if packed int4:     {packed_size:.1f} MB")
    print(f"Compression ratio (stored): {(test_weights.size * 2 / (1024**2)) / stored_size:.2f}x")
    print(f"Compression ratio (packed): {(test_weights.size * 2 / (1024**2)) / packed_size:.2f}x")

    # Dequantize and check error
    dequantized = quantizer.dequantize_weights(q_weights, q_meta)
    error = mx.mean((test_weights - dequantized) ** 2)
    print(f"Reconstruction MSE: {error.item():.6f}")
    
    print("\nQuantization framework test complete")
