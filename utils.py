import torch
import torch.nn as nn
import numpy as np
from typing import Iterable, Optional
import weakref
import copy
import contextlib


@torch.jit.script
def spatial_hash(x, t: int):
    pi = (1, 2_654_435_761, 805_459_861, 2_971_215_073, 433_494_437)
    b, d = x.shape
    if d > len(pi):
        raise ValueError(f"d must be <= {len(pi)}")
    h = x[:, 0] * pi[0]
    for i in range(1, d):
        h ^= ((x[:, i] % t) * (pi[i] % t) % t)
    return h % t


@torch.jit.script
def pos_encoding(x: torch.Tensor, num_freq: int, dim: int = 1):
    out = [x]
    for i in range(num_freq):
        out.extend((torch.cos(np.pi * x * (2 ** i)), torch.sin(np.pi * x * (2 ** i))))
    return torch.cat(out, dim=dim)


@torch.jit.script
def sh_encoding(t: torch.Tensor, degree: int, dim: int = 1):
    if degree > 5:
        raise ValueError("degree > 5 not supported")
    b, d = t.shape
    if d != 3:
        raise ValueError("spherical harmonics encoding only defined for 3 dimensions")
    x, y, z = t[:, 0], t[:, 1], t[:, 2]
    # see https://github.com/NVlabs/tiny-cuda-nn/blob/master/include/tiny-cuda-nn/encodings/spherical_harmonics.h
    # only degree up to 5
    xy = x * y
    xz = x * z
    yz = y * z
    x2 = x * x
    y2 = y * y
    z2 = z * z
    x4 = x2 * x2
    y4 = y2 * y2
    z4 = z2 * z2
    enc = torch.zeros((b, int(degree ** 2)))
    # degree 1
    enc[:, 0] = 0.28209479177387814  # 1/(2*sqrt(pi))
    if degree >= 2:
        enc[:, 1] = -0.48860251190291987 * y  # -sqrt(3)*y/(2*sqrt(pi))
        enc[:, 2] = 0.48860251190291987 * z  # sqrt(3)*z/(2*sqrt(pi))
        enc[:, 3] = -0.48860251190291987 * x  # -sqrt(3)*x/(2*sqrt(pi))
    if degree >= 3:
        enc[:, 4] = 1.0925484305920792 * xy  # sqrt(15)*xy/(2*sqrt(pi))
        enc[:, 5] = -1.0925484305920792 * yz  # -sqrt(15)*yz/(2*sqrt(pi))
        enc[:, 6] = 0.94617469575755997 * z2 - 0.31539156525251999  # sqrt(5)*(3*z2 - 1)/(4*sqrt(pi))
        enc[:, 7] = -1.0925484305920792 * xz  # -sqrt(15)*xz/(2*sqrt(pi))
        enc[:, 8] = 0.54627421529603959 * x2 - 0.54627421529603959 * y2  # sqrt(15)*(x2 - y2)/(4*sqrt(pi))
    if degree >= 4:
        enc[:, 9] = 0.59004358992664352 * y * (-3.0 * x2 + y2)  # sqrt(70)*y*(-3*x2 + y2)/(8*sqrt(pi))
        enc[:, 10] = 2.8906114426405538 * xy * z  # sqrt(105)*xy*z/(2*sqrt(pi))
        enc[:, 11] = 0.45704579946446572 * y * (1.0 - 5.0 * z2)  # sqrt(42)*y*(1 - 5*z2)/(8*sqrt(pi))
        enc[:, 12] = 0.3731763325901154 * z * (5.0 * z2 - 3.0)  # sqrt(7)*z*(5*z2 - 3)/(4*sqrt(pi))
        enc[:, 13] = 0.45704579946446572 * x * (1.0 - 5.0 * z2)  # sqrt(42)*x*(1 - 5*z2)/(8*sqrt(pi))
        enc[:, 14] = 1.4453057213202769 * z * (x2 - y2)  # sqrt(105)*z*(x2 - y2)/(4*sqrt(pi))
        enc[:, 15] = 0.59004358992664352 * x * (-x2 + 3.0 * y2)  # sqrt(70)*x*(-x2 + 3*y2)/(8*sqrt(pi))
    if degree >= 5:
        enc[:, 16] = 2.5033429417967046 * xy * (x2 - y2)  # 3*sqrt(35)*xy*(x2 - y2)/(4*sqrt(pi))
        enc[:, 17] = 1.7701307697799304 * yz * (-3.0 * x2 + y2)  # 3*sqrt(70)*yz*(-3*x2 + y2)/(8*sqrt(pi))
        enc[:, 18] = 0.94617469575756008 * xy * (7.0 * z2 - 1.0)  # 3*sqrt(5)*xy*(7*z2 - 1)/(4*sqrt(pi))
        enc[:, 19] = 0.66904654355728921 * yz * (3.0 - 7.0 * z2)  # 3*sqrt(10)*yz*(3 - 7*z2)/(8*sqrt(pi))
        enc[:, 20] = -3.1735664074561294 * z2 + 3.7024941420321507 * z4 + 0.31735664074561293  # 3*(-30*z2 + 35*z4 + 3)/(16*sqrt(pi))
        enc[:, 21] = 0.66904654355728921 * xz * (3.0 - 7.0 * z2)  # 3*sqrt(10)*xz*(3 - 7*z2)/(8*sqrt(pi))
        enc[:, 22] = 0.47308734787878004 * (x2 - y2) * (7.0 * z2 - 1.0)  # 3*sqrt(5)*(x2 - y2)*(7*z2 - 1)/(8*sqrt(pi))
        enc[:, 23] = 1.7701307697799304 * xz * (-x2 + 3.0 * y2)  # 3*sqrt(70)*xz*(-x2 + 3*y2)/(8*sqrt(pi))
        enc[:, 24] = -3.7550144126950569 * x2 * y2 + 0.62583573544917614 * x4 + 0.62583573544917614 * y4  # 3*sqrt(35)*(-6*x2*y2 + x4 + y4)/(16*sqrt(pi))
    return torch.cat((t, enc), dim=dim)


def make_mlp(eta_dim, output_dim, feature_dim=2, levels=16, hidden_layers=2, hidden_dim=64, output_nonlinearity=None):
    layers = []
    # input layer
    layers.extend((
        nn.Linear(levels * feature_dim + eta_dim, hidden_dim),
        nn.ReLU()
    ))
    # hidden layers
    for _ in range(hidden_layers):
        layers.extend((
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        ))
    # output layers
    layers.append(nn.Linear(hidden_dim, output_dim))
    if output_nonlinearity is not None:
        layers.append(output_nonlinearity)

    return nn.Sequential(*layers)


# From https://www.github.com/fadel/pytorch_ema/master/torch_ema/ema.py

# Partially based on:
# https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/training/moving_averages.py
class ExponentialMovingAverage:
    """
    Maintains (exponential) moving average of a set of parameters.

    Args:
        parameters: Iterable of `torch.nn.Parameter` (typically from
            `model.parameters()`).
            Note that EMA is computed on *all* provided parameters,
            regardless of whether or not they have `requires_grad = True`;
            this allows a single EMA object to be consistantly used even
            if which parameters are trainable changes step to step.

            If you want to some parameters in the EMA, do not pass them
            to the object in the first place. For example:

                ExponentialMovingAverage(
                    parameters=[p for p in model.parameters() if p.requires_grad],
                    decay=0.9
                )

            will ignore parameters that do not require grad.

        decay: The exponential decay.

        use_num_updates: Whether to use number of updates when computing
            averages.
    """

    def __init__(
            self,
            parameters: Iterable[torch.nn.Parameter],
            decay: float,
            use_num_updates: bool = True
    ):
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        parameters = list(parameters)
        self.shadow_params = [
            p.clone().detach()
            for p in parameters
        ]
        self.collected_params = None
        # By maintaining only a weakref to each parameter,
        # we maintain the old GC behaviour of ExponentialMovingAverage:
        # if the model goes out of scope but the ExponentialMovingAverage
        # is kept, no references to the model or its parameters will be
        # maintained, and the model will be cleaned up.
        self._params_refs = [weakref.ref(p) for p in parameters]

    def _get_parameters(
            self,
            parameters: Optional[Iterable[torch.nn.Parameter]]
    ) -> Iterable[torch.nn.Parameter]:
        if parameters is None:
            parameters = [p() for p in self._params_refs]
            if any(p is None for p in parameters):
                raise ValueError(
                    "(One of) the parameters with which this "
                    "ExponentialMovingAverage "
                    "was initialized no longer exists (was garbage collected);"
                    " please either provide `parameters` explicitly or keep "
                    "the model to which they belong from being garbage "
                    "collected."
                )
            return parameters
        else:
            parameters = list(parameters)
            if len(parameters) != len(self.shadow_params):
                raise ValueError(
                    "Number of parameters passed as argument is different "
                    "from number of shadow parameters maintained by this "
                    "ExponentialMovingAverage"
                )
            return parameters

    def update(
            self,
            parameters: Optional[Iterable[torch.nn.Parameter]] = None
    ) -> None:
        """
        Update currently maintained parameters.

        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; usually the same set of
                parameters used to initialize this object. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        parameters = self._get_parameters(parameters)
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(
                decay,
                (1 + self.num_updates) / (10 + self.num_updates)
            )
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            for s_param, param in zip(self.shadow_params, parameters):
                tmp = (s_param - param)
                # tmp will be a new tensor so we can do in-place
                tmp.mul_(one_minus_decay)
                s_param.sub_(tmp)

    def copy_to(
            self,
            parameters: Optional[Iterable[torch.nn.Parameter]] = None
    ) -> None:
        """
        Copy current averaged parameters into given collection of parameters.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        parameters = self._get_parameters(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            param.data.copy_(s_param.data)

    def store(
            self,
            parameters: Optional[Iterable[torch.nn.Parameter]] = None
    ) -> None:
        """
        Save the current parameters for restoring later.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                temporarily stored. If `None`, the parameters of with which this
                `ExponentialMovingAverage` was initialized will be used.
        """
        parameters = self._get_parameters(parameters)
        self.collected_params = [
            param.clone()
            for param in parameters
        ]

    def restore(
            self,
            parameters: Optional[Iterable[torch.nn.Parameter]] = None
    ) -> None:
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored parameters. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        if self.collected_params is None:
            raise RuntimeError(
                "This ExponentialMovingAverage has no `store()`ed weights "
                "to `restore()`"
            )
        parameters = self._get_parameters(parameters)
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    @contextlib.contextmanager
    def average_parameters(
            self,
            parameters: Optional[Iterable[torch.nn.Parameter]] = None
    ):
        r"""
        Context manager for validation/inference with averaged parameters.

        Equivalent to:

            ema.store()
            ema.copy_to()
            try:
                ...
            finally:
                ema.restore()

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored parameters. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        parameters = self._get_parameters(parameters)
        self.store(parameters)
        self.copy_to(parameters)
        try:
            yield
        finally:
            self.restore(parameters)

    def to(self, device=None, dtype=None) -> None:
        r"""Move internal buffers of the ExponentialMovingAverage to `device`.

        Args:
            device: like `device` argument to `torch.Tensor.to`
        """
        # .to() on the tensors handles None correctly
        self.shadow_params = [
            p.to(device=device, dtype=dtype)
            if p.is_floating_point()
            else p.to(device=device)
            for p in self.shadow_params
        ]
        if self.collected_params is not None:
            self.collected_params = [
                p.to(device=device, dtype=dtype)
                if p.is_floating_point()
                else p.to(device=device)
                for p in self.collected_params
            ]
        return

    def state_dict(self) -> dict:
        r"""Returns the state of the ExponentialMovingAverage as a dict."""
        # Following PyTorch conventions, references to tensors are returned:
        # "returns a reference to the state and not its copy!" -
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow_params": self.shadow_params,
            "collected_params": self.collected_params
        }

    def load_state_dict(self, state_dict: dict) -> None:
        r"""Loads the ExponentialMovingAverage state.

        Args:
            state_dict (dict): EMA state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        state_dict = copy.deepcopy(state_dict)
        self.decay = state_dict["decay"]
        if self.decay < 0.0 or self.decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.num_updates = state_dict["num_updates"]
        assert self.num_updates is None or isinstance(self.num_updates, int), \
            "Invalid num_updates"

        self.shadow_params = state_dict["shadow_params"]
        assert isinstance(self.shadow_params, list), \
            "shadow_params must be a list"
        assert all(
            isinstance(p, torch.Tensor) for p in self.shadow_params
        ), "shadow_params must all be Tensors"

        self.collected_params = state_dict["collected_params"]
        if self.collected_params is not None:
            assert isinstance(self.collected_params, list), \
                "collected_params must be a list"
            assert all(
                isinstance(p, torch.Tensor) for p in self.collected_params
            ), "collected_params must all be Tensors"
            assert len(self.collected_params) == len(self.shadow_params), \
                "collected_params and shadow_params had different lengths"

        if len(self.shadow_params) == len(self._params_refs):
            # Consistant with torch.optim.Optimizer, cast things to consistant
            # device and dtype with the parameters
            params = [p() for p in self._params_refs]
            # If parameters have been garbage collected, just load the state
            # we were given without change.
            if not any(p is None for p in params):
                # ^ parameter references are still good
                for i, p in enumerate(params):
                    self.shadow_params[i] = self.shadow_params[i].to(
                        device=p.device, dtype=p.dtype
                    )
                    if self.collected_params is not None:
                        self.collected_params[i] = self.collected_params[i].to(
                            device=p.device, dtype=p.dtype
                        )
        else:
            raise ValueError(
                "Tried to `load_state_dict()` with the wrong number of "
                "parameters in the saved state."
            )
