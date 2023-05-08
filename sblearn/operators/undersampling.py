import torch
from torch import Tensor

from sblearn.operators.linop import LinearOperator


class Undersampling(LinearOperator):
    """An undersampling operator to exclude certain inputs."""

    def __init__(self, mask: Tensor, zero_out: bool = False) -> Tensor:
        """Initializes the undersampling mask.

        Parameters
        ----------
        mask : Tensor
            A 1D boolean tensor, where true elements are retained
            and false elements are excluded.
        zero_out : bool
            If true, the output shape retains the input shape and sets
            masked elements to zero.  If false, the output is a vector
            of selected elements according to the mask.

        """
        self.mask = mask
        self.zero_out = zero_out

    def apply(self, inp: Tensor) -> Tensor:
        out = inp
        if self.zero_out:
            out[..., ~self.mask] = 0
        else:
            out = out[..., self.mask]
        return out

    def transpose(self, out: Tensor) -> Tensor:
        if self.zero_out:
            if not torch.all(out[..., ~self.mask] == 0):
                raise ValueError("Invalid input to Undersampling disobeys mask.")
            inp = out
        else:
            tensor_inp_shape = out.shape[:-1] + (self.inp_dim,)
            inp = torch.zeros(tensor_inp_shape, device=out.device, dtype=out.dtype)
            inp[..., self.mask] = out
        return inp

    @property
    def inp_dim(self) -> int:
        return len(self.mask)

    @property
    def out_dim(self) -> int:
        if self.zero_out:
            return len(self.mask)
        else:
            return self.mask.sum().cpu().item()
