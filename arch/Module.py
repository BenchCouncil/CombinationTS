import torch
from torch import nn
from typing import Callable, Optional, Any, Union
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


class BaseModule(nn.Module):
    in_tensors: int
    # in_tensors represents the number of input tensors the module can handle
    # `1` means single tensor input
    # `n (>1)` means list of `n` tensors as input
    # `-1` means flexible number of input tensors, accepting both single tensor and list of tensors
    out_tensors: int
    # out_tensors represents the number of output tensors the module produces
    # `1` means single tensor output
    # `n (>1)` means list of `n` tensors as output
    # `-1` means flexible number of output tensors, producing the same number of output tensors as input tensors
    init_shape: list[tuple]
    # _init_shape represents the shape(s) of the input tensor(s) used during module initialization
    _output_shape_template: Union[tuple, list[tuple], int]
    # _output_shape_template represents the template shape(s) of the output tensor(s)
    # set `-1` return input shape
    _flexible_template: bool
    # a flag indicating whether the output shape template contains flexible dimensions
    output_shape: Union[list[tuple], None]
    # output_shape represents the shape(s) of the output tensor(s)
    hidden_dim: int
    channel_independence: bool
    task: dict[str, int]
    layers: nn.ModuleList  # for modules with multiple layers to hook filters
    
    def __init__(
        self,
        init_shape: Union[tuple, list[tuple]],
    ):
        r"""Base class for all modules in the framework.

        The module can handle single tensor or multiple tensors as input and output and be applied
        functions to itself after initialization. It also provides methods to help automatically
        calculate the output shape.

        Args:
            init_shape (tuple or list of tuple): shape of the input tensor(s) used during
                module initialization
            task (dict): task information, e.g., {'seq_len': 96, 'pred_len': 16, 'n_vars': 7}
            channel_independence (bool): whether the module operates in channel independence mode
            in_tensors (int): number of input tensors the module can handle, `-1` for flexible number
            out_tensors (int): number of output tensors the module produces, `-1` for the same number as input tensors
        """
        super(BaseModule, self).__init__()
        if isinstance(init_shape[0], list):
            self.init_shape = [tuple(shape) for shape in init_shape]
        else:
            self.init_shape = [tuple(init_shape)]
        self.output_shape = None

    def apply_self(self, fn: Callable[[nn.Module], Optional[Any]]):
        r"""Applies ``fn`` to the module itself.

        Use it to modify or query attributes of the module. You can modify the module after init.

        Args:
            fn (:class:`Module` -> `Optional[Any]`): function that applies to this module
        
        Returns:
            Optional[Any]: the return value of ``fn``
        """
        return fn(self)

    def set_output_shape_template(self, shape_template: Union[tuple, list[tuple], int]):
        r"""Set the output shape template of the module.
        Use `-1` for flexible dimensions.

        Args:
            output_shape (tuple or list of tuple): shape of the output tensor(s)
        """
        if isinstance(shape_template, list):
            shape_template = [tuple(shape) for shape in shape_template]
            self._flexible_template = any(
                x < 0 for shape in shape_template for x in shape)
        elif isinstance(shape_template, tuple):
            shape_template = tuple(shape_template)
            self._flexible_template = any(dim == -1 for dim in shape_template)
        elif shape_template == -1:
            self._flexible_template = True
        else:
            raise ValueError("Output shape template must be a tuple or list of tuple.")

        self._output_shape_template = shape_template
    
    def get_output_shape(self, input_shape: Union[tuple, list[tuple], None]) -> Union[tuple, list[tuple]]:
        r"""Calculate the output shape of the module given the input shape.
        If module accepts flexible input shapes, callculate the output shape accordingly.

        Args:
            input_shape (tuple or list of tuple): shape of the input tensor(s)

        Returns:
            tuple or list of tuple: shape of the output tensor(s)
        """
        assert self.output_shape is not None or self._output_shape_template is not None, \
            "Output shape or output shape template is not set. Call `set_output_shape_template` method to set the output shape template."
        input_shape = (input_shape if isinstance(input_shape, list) else [input_shape]) if input_shape else None
        if self.output_shape is not None:
            assert input_shape == self.init_shape or input_shape is None, f"Input shape {input_shape} does not match the initialized shape {self.init_shape}."
            return self.output_shape

        if self._output_shape_template == -1:
            assert input_shape is not None, "Input shape must be provided when output shape template is -1."
            return input_shape

        assert hasattr(self, '_output_shape_template'), f"Output shape template is not set. Call \
            `set_output_shape_template` method to set the output shape template or override `cal_output_shape` method."

        if not self._flexible_template:

            if input_shape == self.init_shape:
                return self._output_shape_template  # type: ignore

            elif isinstance(input_shape, list):
                if isinstance(self.init_shape, tuple):
                    for in_shape in input_shape:
                        assert in_shape == self.init_shape, f"Input shape {in_shape} does not match the initialized shape {self.init_shape}."
                    return [self._output_shape_template for _ in input_shape]   # type: ignore
                elif isinstance(self.init_shape, list):
                    assert len(input_shape) == len(self.init_shape), "Input shape list length does not match the initialized shape list length."
                    for in_shape, init_shape in zip(input_shape, self.init_shape):
                        assert in_shape == init_shape, f"Input shape {in_shape} does not match the initialized shape {init_shape}."
                    return self._output_shape_template  # type: ignore

            elif isinstance(input_shape, tuple):
                assert input_shape == self.init_shape, f"Input shape {input_shape} does not match the initialized shape {self.init_shape}."
        else:

            def resolve_shape(template: tuple, input_shape: tuple) -> tuple:
                assert len(template) == len(input_shape), "Template shape and input shape must have the same number of dimensions."
                out_shape = []
                for t_dim, i_dim in zip(template, input_shape):
                    if t_dim != -1:
                        out_shape.append(t_dim)
                    else:
                        out_shape.append(i_dim)
                out_shape = tuple(out_shape)
                return out_shape

            if isinstance(input_shape, tuple):
                assert not isinstance(self._output_shape_template, list), "Output shape template must be a tuple when input shape is a tuple."
                return resolve_shape(self._output_shape_template, input_shape) # type: ignore

            elif isinstance(input_shape, list):
                if isinstance(self._output_shape_template, tuple):
                    return [resolve_shape(self._output_shape_template, in_shape) for in_shape in input_shape]
                elif isinstance(self._output_shape_template, list):
                    assert len(input_shape) == len(self._output_shape_template), "Input shape list length does not match the output shape template list length."
                    out_shapes = []
                    for in_shape, out_template in zip(input_shape, self._output_shape_template):
                        out_shapes.append(resolve_shape(out_template, in_shape))
                    return out_shapes

        raise ValueError("Unable to calculate output shape with given input shape.")


def build_module(
    cfg: DictConfig, init_shape: Union[tuple, list[tuple]],
    **kwargs
) -> BaseModule:
    OmegaConf.set_struct(cfg, False)  # allow adding new fields
    cfg['_convert_'] = 'all'  # hydra instantiate config conversion. Convert parameters as Dict, List, etc.
    OmegaConf.set_struct(cfg, True)
    if cfg.in_tensors == 1:
        init_shape = init_shape[0]
    module = instantiate(cfg, input_shape=init_shape, **kwargs)
    assert isinstance(module, BaseModule), f"Module {module} is not an instance of BaseModule."
    return module