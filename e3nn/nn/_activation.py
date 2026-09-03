import torch

from e3nn.o3._irreps import Irreps
from e3nn.math import normalize2mom
from e3nn.util.jit import compile_mode


@compile_mode("trace")
class Activation(torch.nn.Module):
    r"""Scalar activation function.

    Scalar inputs that are odd under parity or time reversal require activation functions with a defined parity (odd or even).

    Parameters
    ----------
    irreps_in : `e3nn.o3.Irreps`
        representation of the input

    acts : list of function or None
        list of activation functions, `None` if non-scalar or identity

    Examples
    --------

    >>> a = Activation("256x0o", [torch.abs])
    >>> a.irreps_out
    256x0ee

    >>> a = Activation("256x0o+16x1e", [None, None])
    >>> a.irreps_out
    256x0oe+16x1ee
    """

    def __init__(self, irreps_in, acts) -> None:
        super().__init__()
        irreps_in = Irreps(irreps_in)
        if len(irreps_in) != len(acts):
            raise ValueError(f"Irreps in and number of activation functions does not match: {len(acts), (irreps_in, acts)}")

        # normalize the second moment
        acts = [normalize2mom(act) if act is not None else None for act in acts]

        from e3nn.util._argtools import _get_device

        irreps_out = []
        for (mul, (l_in, p_in, t_in)), act in zip(irreps_in, acts):
            if act is not None:
                if l_in != 0:
                    raise ValueError("Activation: cannot apply an activation function to a non-scalar input.")

                x = torch.linspace(0, 10, 256, device=_get_device(act))

                a1, a2 = act(x), act(-x)
                if (a1 - a2).abs().max() < 1e-5:
                    p_act = 1
                elif (a1 + a2).abs().max() < 1e-5:
                    p_act = -1
                else:
                    p_act = 0

                p_out = p_act if p_in == -1 else p_in
                t_out = p_act if t_in == -1 else t_in
                if p_out == 0 or t_out == 0:
                    raise ValueError(
                        "Activation: parity or time-reversal symmetry is violated! The input scalar is odd under a "
                        "symmetry but the activation is neither even nor odd."
                    )
                irreps_out.append((mul, (0, p_out, t_out)))
            else:
                irreps_out.append((mul, (l_in, p_in, t_in)))

        self.irreps_in = irreps_in
        self.irreps_out = Irreps(irreps_out)
        self.acts = torch.nn.ModuleList(acts)
        self.paths = [(mul, (l, p, t), act) for (mul, (l, p, t)), act in zip(self.irreps_in, self.acts)]
        assert len(self.irreps_in) == len(self.acts)

    def __repr__(self) -> str:
        acts = "".join(["x" if a is not None else " " for a in self.acts])
        return f"{self.__class__.__name__} [{acts}] ({self.irreps_in} -> {self.irreps_out})"

    def forward(self, features, dim: int = -1):
        """evaluate

        Parameters
        ----------
        features : `torch.Tensor`
            tensor of shape ``(...)``

        Returns
        -------
        `torch.Tensor`
            tensor of shape the same shape as the input
        """
        # - PROFILER - with torch.autograd.profiler.record_function(repr(self)):
        output = []
        index = 0
        for mul, (l, _, _), act in self.paths:
            ir_dim = 2 * l + 1
            if act is not None:
                output.append(act(features.narrow(dim, index, mul)))
            else:
                output.append(features.narrow(dim, index, mul * ir_dim))
            index += mul * ir_dim

        if len(output) > 1:
            return torch.cat(output, dim=dim)
        elif len(output) == 1:
            return output[0]
        else:
            return torch.zeros_like(features)
