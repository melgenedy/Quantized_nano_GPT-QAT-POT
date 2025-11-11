

import re
from abc import ABC, abstractmethod
from typing import Any

import torch
from torch.ao.quantization.observer import (
    _with_args,
    default_fixed_qparams_range_0to1_observer,
    default_fixed_qparams_range_neg1to1_observer,
    FixedQParamsObserver,
    HistogramObserver,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
)
from torch.nn import Module


from torch.ao.quantization.fake_quantize import FakeQuantize, _is_float_qparams, _is_per_channel, _is_per_tensor


class DifferentiablePowerOfTwo(torch.autograd.Function):
    """Custom autograd function compliant for power-of-two quantization"""
    
    @staticmethod
    def forward(ctx, input, scale, zero_point, quant_min, quant_max):#num_bits=8):
        # forward pass: actual power-of-two quantization
        max_val = torch.max(torch.abs(input))
        
        if max_val > 0:
            scale = scale# 2 ** torch.ceil(torch.log2(max_val / (2**(num_bits-1) - 1)))
        else:
            scale = 1.0
        
        # Normalize
        normalized = input / scale
        
        # Convert to power of two
        sign = torch.sign(normalized)
        abs_val = torch.round(torch.abs(normalized))
        #abs_val = torch.ceil(torch.clamp(abs_val, min=1e-8))
        mask_zero = (abs_val == 0)
        abs_val = torch.where(mask_zero, torch.tensor(1), abs_val)    
        
        log2_val = torch.log2(abs_val)
        rounded_log2 = torch.round(log2_val)  # Non-differentiable
        power_of_two = sign * torch.pow(2.0, rounded_log2)
        power_of_two = torch.where(mask_zero, torch.tensor(0), power_of_two)
        
        # Clamp to range
        #output_clamp = torch.clamp(output, -(2**(num_bits-1)), 2**(num_bits-1) - 1)
        output_clamp = torch.clamp(power_of_two, quant_min, quant_max)

        # Scale back
        output = output_clamp * scale


        
        # Save for backward pass
        ctx.save_for_backward(input, output)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        # backward pass: straight-through estimator STE
        input, output = ctx.saved_tensors
        
        # STE: pretend the quantization was identity function
        # Gradient goes through as if (no quantization) happened
        # other alternatives can be used to get closer to log function
        grad_input = grad_output.clone()
        
        return grad_input, None, None, None, None  # None for other inputs of forward path (no gradient needed)


class FakeQuantize_PoT(FakeQuantize):
    ''' This is an inherited fake quantize that will call the Power of Two quantization (DifferentiablePowerOfTwo) instead of uniform quantization'''

    scale: torch.Tensor
    zero_point: torch.Tensor

    def __init__(
        self,
        observer=MovingAverageMinMaxObserver,
        quant_min=None,
        quant_max=None,
        is_dynamic=False,
        **observer_kwargs,
    ):
        super().__init__()
        # Populate quant_min/quant_max to observer_kwargs if valid
        if quant_min is not None and quant_max is not None:
            assert quant_min <= quant_max, (
                "quant_min must be less than or equal to quant_max"
            )
            dtype = observer_kwargs.get("dtype", torch.quint8)
            if hasattr(observer, "p"):
                # In case observer is _PartialWrapper, dtype can be stored in
                # observer.p.keywords["dtype"]
                dtype = getattr(getattr(observer, "p", {}), "keywords", {}).get(
                    "dtype", dtype
                )
            assert torch.iinfo(dtype).min <= quant_min, "quant_min out of bound"
            assert quant_max <= torch.iinfo(dtype).max, "quant_max out of bound"
            observer_kwargs.update({"quant_min": quant_min, "quant_max": quant_max})
        observer_kwargs["is_dynamic"] = is_dynamic
        self.activation_post_process = observer(**observer_kwargs)
        # TODO: keeping self.quant_min/max for BC; remove after a couple releases
        # Users should use self.activation_post_process.quant_min
        self.quant_min = self.activation_post_process.quant_min
        self.quant_max = self.activation_post_process.quant_max
        self.is_dynamic = self.activation_post_process.is_dynamic
        if _is_float_qparams(self.activation_post_process.qscheme):
            zero_point_dtype = torch.float
        else:
            zero_point_dtype = torch.int
        self.register_buffer("scale", torch.tensor([1.0], dtype=torch.float))
        self.register_buffer("zero_point", torch.tensor([0], dtype=zero_point_dtype))
        self.dtype = self.activation_post_process.dtype
        self.qscheme = self.activation_post_process.qscheme
        self.ch_axis = (
            self.activation_post_process.ch_axis
            if hasattr(self.activation_post_process, "ch_axis")
            else -1
        )
        assert _is_per_channel(self.qscheme) or _is_per_tensor(self.qscheme), (
            "Only per channel and per tensor quantization are supported in fake quantize"
            + " got qscheme: "
            + str(self.qscheme)
        )
        self.is_per_channel = _is_per_channel(self.qscheme)

    @torch.jit.export
    def calculate_qparams(self):  # type: ignore[override]
        return self.activation_post_process.calculate_qparams()

    def forward(self, X):
        if self.observer_enabled[0] == 1:
            self.activation_post_process(X.detach())
            _scale, _zero_point = self.calculate_qparams()
            _scale, _zero_point = (
                _scale.to(self.scale.device),
                _zero_point.to(self.zero_point.device),
            )
            if self.scale.shape != _scale.shape:
                self.scale.resize_(_scale.shape)
                self.zero_point.resize_(_zero_point.shape)
            self.scale.copy_(_scale)
            self.zero_point.copy_(_zero_point)

        if self.fake_quant_enabled[0] == 1:
            ''' calling PoT quantization instead of uniform affine/symmetric quatization'''
            X = DifferentiablePowerOfTwo.apply(
                    X,self.scale,
                    self.zero_point,
                    self.activation_post_process.quant_min,
                    self.activation_post_process.quant_max,                         
            )


            '''if self.is_per_channel:
                X = torch.fake_quantize_per_channel_affine(
                    X,
                    self.scale,
                    self.zero_point,
                    self.ch_axis,
                    self.activation_post_process.quant_min,
                    self.activation_post_process.quant_max,
                )
            else:
                X = torch.fake_quantize_per_tensor_affine(
                    X,
                    self.scale,
                    self.zero_point,
                    self.activation_post_process.quant_min,
                    self.activation_post_process.quant_max,
                )'''

        return X

    ''' following are not needed and should be cleaned up in future'''
    @torch.jit.export
    def extra_repr(self):
        return (
            f"fake_quant_enabled={self.fake_quant_enabled}, observer_enabled={self.observer_enabled}, "
            f"quant_min={self.activation_post_process.quant_min}, quant_max={self.activation_post_process.quant_max}, "
            f"dtype={self.dtype}, qscheme={self.qscheme}, ch_axis={self.ch_axis}, "
            f"scale={self.scale}, zero_point={self.zero_point}"
        )

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # We cannot currently register scalar values as buffers, so need to manually
        # specify serialization here.
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + "scale"] = self.scale
        destination[prefix + "zero_point"] = self.zero_point

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Removing this function throws an error that the size of the loaded tensor does not match the original size
        # i.e., These buffers start out with numel 0 and become numel 1 once they have their first forward pass.
        local_state = ["scale", "zero_point"]
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                # Custom handling to allow loading scale and zero_point
                # of size N into uninitialized buffers of size 0. The
                # buffers are resized here, and the values are copied in
                # the default state_dict loading code of the parent.
                if name == "scale":
                    self.scale.resize_(val.shape)
                else:
                    assert name == "zero_point"
                    self.zero_point.resize_(val.shape)
                # For torchscript module we need to update the attributes here since we do not
                # call the `_load_from_state_dict` function defined module.py
                if torch.jit.is_scripting():
                    if name == "scale":
                        self.scale.copy_(val)
                    else:
                        assert name == "zero_point"
                        self.zero_point.copy_(val)
            elif strict:
                missing_keys.append(key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
