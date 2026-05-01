import torch
from torch import nn
from omegaconf import DictConfig, OmegaConf
from typing import Union, Callable, Optional, Any
from arch import BaseModule, build_module


class Block(nn.Module):
    r"""A block that can contain one or more modules.

    The block can handle single tensor or multiple tensors as input and output.
    """
    _x_to_module: list[tuple[int, BaseModule]]
    # _x_to_module maps input tensor indices to their corresponding modules
    in_tensors: int
    out_tensors: int
    init_shapes: list[tuple]
    output_shapes: list[tuple]
    block_modules: nn.ModuleList
    skip_build_module: bool = False # For subclasses that do not build modules in __init__

    def __init__(
        self, cfgs: Union[DictConfig, list[DictConfig]], init_shapes: Union[tuple, list[tuple]],
        seq_len: int, label_len: int, pred_len: int, n_vars: int,
        d_model: int, dropout: float, channel_independence: bool,
        shared_module: bool=False,
    ):
        super(Block, self).__init__()
        self.cfgs = cfgs if isinstance(cfgs, list) else [cfgs]
        self.init_shapes = init_shapes if isinstance(init_shapes, list) else [init_shapes]
        self.in_tensors = len(self.init_shapes)
        self.out_tensors = 0
        self.output_shapes = []
        self._x_to_module = []
        self.block_modules = nn.ModuleList()
        self.shared_module = shared_module
        self.params = {
            "seq_len": seq_len,
            "label_len": label_len,
            "n_vars": n_vars,
            "pred_len": pred_len,
            "d_model": d_model,
            "dropout": dropout,
            "channel_independence": channel_independence,
        }

        for cfg in self.cfgs:
            OmegaConf.set_struct(cfg, False)  # allow adding new fields
            in_tensors = cfg.get('in_tensors', 1)  # set default in_tensors to 1 if not specified
            cfg['in_tensors'] = in_tensors

        assert all(cfg.in_tensors > 0 or cfg.in_tensors == -1 for cfg in self.cfgs), \
            f"in_tensors in configuration must be positive integer or -1 for flexible input tensors.\
              got {[{cfg.name: cfg.in_tensors} for cfg in self.cfgs if cfg.in_tensors <= 0 and cfg.in_tensors != -1]}"

        if not type(self).skip_build_module:
            if len(self.cfgs) == 1 and not self.shared_module and self.cfgs[0].in_tensors > 0:
                self._build_modules_with_single_cfg()
            elif len(self.cfgs) == 1 and self.shared_module:
                self._build_shared_module()
            elif len(self.cfgs) == 1 and self.cfgs[0].in_tensors == -1:
                self._build_shared_module()
            else:
                self._build_multi_module()


    def forward(self, x: Union[torch.Tensor, list[torch.Tensor]], ex_x: Union[torch.Tensor, list[torch.Tensor], None] = None, **args) -> list[torch.Tensor]:
        r"""Forward pass of the block.

        Args:
            x (torch.Tensor or list of torch.Tensor): input tensor(s)

        Returns:
            torch.Tensor or list of torch.Tensor: output tensor(s)
        """
        x_list = x if isinstance(x, list) else [x]
        assert len(x_list) == self.in_tensors, \
            f"Number of input tensors ({len(x_list)}) does not match block's in_tensors ({self.in_tensors})"
        
        outputs = []
        idx = 0
        for in_tensors, module in self._x_to_module:
            x_input = x_list[idx:idx+in_tensors]
            x_mark_input = ex_x[idx:idx+in_tensors] if ex_x is not None and isinstance(ex_x, list) else ex_x
            if in_tensors == 1:
                x_input = x_input[0]
                x_mark_input = x_mark_input[0] if x_mark_input is not None and isinstance(x_mark_input, list) else x_mark_input
            out = module(x_input, x_mark_input) if x_mark_input is not None else module(x_input)
            if isinstance(out, torch.Tensor):
                outputs.append(out)
            else:
                outputs.extend(out)
            idx += in_tensors
        
        return outputs


    def apply_self(self, fn: Callable[[nn.Module], Optional[Any]]):
        r"""Applies ``fn`` to the module itself.

        Use it to modify or query attributes of the module. You can modify the module after init.

        Args:
            fn (:class:`Module` -> `Optional[Any]`): function that applies to this module
        
        Returns:
            Optional[Any]: the return value of ``fn``
        """
        return fn(self)

    
    def get_output_shapes(self) -> list[tuple]:
        r"""Get the output shapes of the block.

        Returns:
            list of tuple: output shapes of the block
        """
        return self.output_shapes


    def _build_modules_with_single_cfg(self):
        r"""Build modules when there is a single configuration.
        Each module processes a group of input tensors based on its `in_tensors` setting.
        """
        assert len(self.cfgs) == 1, "Block with single configuration must have exactly one configuration."
        cfg = self.cfgs[0]
        in_tensors = cfg.in_tensors
        assert len(self.init_shapes) % in_tensors == 0, \
            f"Number of init_shapes ({len(self.init_shapes)}) must be divisible by in_tensors ({in_tensors})"

        output_shapes = []
        
        for i in range(0, len(self.init_shapes), in_tensors):
            init_shape_group = self.init_shapes[i:i+in_tensors]
            module = self._build_module(
                cfg, init_shape_group
            )
            output_shapes.append(module.output_shape)
            self._register_module(module, in_tensors, init_shape_group)
        

    def _build_shared_module(self):
        r"""Build a shared module for all input tensors.
        All input tensors are processed by the same module instance.

        There are two cases:
        1. Fixed input tensor module: the shared module has a fixed number of input tensors
            (i.e., `in_tensors > 0`). Each group of input tensors is processed by the shared module.
        2. Flexible input tensor module
            the shared module can handle a flexible number of input tensors
            (i.e., `in_tensors == -1`). All input tensors are processed together by the shared module.
        """
        assert len(self.cfgs) == 1, "Shared module block must have exactly one configuration."
        cfg = self.cfgs[0]
        in_tensors = cfg.in_tensors
        assert len(self.init_shapes) % in_tensors == 0 or in_tensors == -1, \
            f"Number of init_shapes ({len(self.init_shapes)}) must be divisible by in_tensors ({in_tensors})"

        # flexible input tensor module
        if in_tensors == -1:
            in_tensors = len(self.init_shapes)
            module = self._build_module(
                cfg, self.init_shapes,
            )
            self._register_module(module, in_tensors, self.init_shapes)
            return

        assert in_tensors > 0, "If you want to build a flexible input tensor module, please set `in_tensors` to `-1` in the configuration."

        module = self._build_module(
            cfg, self.init_shapes[:in_tensors]
        )
        for i in range(0, len(self.init_shapes), in_tensors):
            input_shape_group = self.init_shapes[i:i+in_tensors]
            self._register_module(module, in_tensors, input_shape_group)


    def _build_multi_module(self):
        r"""Build modules when there are multiple configurations.
        Each configuration corresponds to a module that processes a group of input tensors.

        Only one module can have flexible input tensors (i.e., `in_tensors == -1`).
        """
        _flex_in_tensors = [cfg.in_tensors for cfg in self.cfgs if cfg.in_tensors == -1]
        assert len(_flex_in_tensors) <= 1, f"Only one module can have flexible input tensors, but got {len(_flex_in_tensors)}."

        exist_flexible = len(_flex_in_tensors) > 0
        if exist_flexible:
            non_flexible_total = sum(
                cfg.in_tensors for cfg in self.cfgs if cfg.in_tensors != -1
            )
            flex_in_tensor = len(self.init_shapes) - non_flexible_total
            assert flex_in_tensor > 0, \
                f"Flexible module must have at least one input tensor. All input tensors: {len(self.init_shapes)}, \
                non-flexible total: {non_flexible_total}. flex_in_tensor: {flex_in_tensor}"
            in_tensors = [
                flex_in_tensor if cfg.in_tensors == -1 else cfg.in_tensors for cfg in self.cfgs
            ]
        else:
            in_tensors = [cfg.in_tensors for cfg in self.cfgs]

        assert sum(in_tensors) == len(self.init_shapes), \
            f"Sum of in_tensors ({sum(in_tensors)}) must equal to number of init_shapes ({len(self.init_shapes)})"
        
        idx = 0
        for cfg, in_tensor in zip(self.cfgs, in_tensors):
            init_shape_group = self.init_shapes[idx:idx+in_tensor]
            module = self._build_module(
                cfg, init_shape_group,
            )
            self._register_module(module, in_tensor, init_shape_group)
            idx += in_tensor


    def _build_module(
        self, cfg: DictConfig, input_shapes: Union[tuple, list[tuple]]
    ) -> BaseModule:
        r"""Build a module from configuration and append it to the block.

        Args:
            cfg (DictConfig): configuration of the module
            in_tensors (int): number of input tensors the module can handle
            input_shapes (tuple or list of tuple): input shape(s) of the module

        Returns:
            BaseModule: built module
        """
        module = build_module(
            cfg, input_shapes, **self.params
        )
        self.block_modules.append(module)
        return module

    def _register_module(
        self, module: BaseModule, in_tensors: int, input_shapes: Union[tuple, list[tuple]]
    ):
        r"""Register a module to the block.

        Modules accept `Block` input tensors according to register order.

        Args:
            module (:class:`BaseModule`): module to be registered
            in_tensors (int): number of input tensors the module can handle
        """
        self._x_to_module.append((in_tensors, module))
        output_shapes = module.get_output_shape(input_shapes)
        if isinstance(output_shapes, tuple):
            self.output_shapes.append(output_shapes)
        else:
            self.output_shapes.extend(output_shapes)
        self.out_tensors += (1 if isinstance(output_shapes, tuple) else len(output_shapes))