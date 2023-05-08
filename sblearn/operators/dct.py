import torch
from torch import Tensor
from torch.fft import fft, ifft
import numpy as np
from scipy.fftpack import dct, idct

from sblearn.operators.linop import LinearOperator


class DiscreteCosine1D(LinearOperator):
    """The orthogonal discrete cosine transform (DCT) in one dimension.

    This function can optionally use the fast SciPy implementation for
    DCT on CPU and a more indirect method based on the fast Fourier
    transform from PyTorch.

    """

    def __init__(self, length: int, use_fft: bool = False) -> None:
        """Initializes the 1D DCT.

        Parameters
        ----------
        length : int
            The dimensionality of the data for the DCT.
        use_fft : bool
            If true, uses PyTorch's fast Fourier transform to compute
            the DCT (recommended on GPU).  If false, uses SciPy's
            direct implementation for DCT (recommended on CPU).

        """
        self.length = length
        self.use_fft = use_fft

    def apply(self, inp: Tensor) -> Tensor:
        if self.use_fft:
            out = self._torch_dct(inp)
        else:
            inp_np = inp.numpy()
            out_np = dct(inp_np, norm="ortho", axis=-1)
            out = torch.tensor(out_np, dtype=inp.dtype, device=inp.device)
        return out

    def transpose(self, out: Tensor) -> Tensor:
        if self.use_fft:
            inp = self._torch_idct(out)
        else:
            out_np = out.numpy()
            inp_np = idct(out_np, norm="ortho", axis=-1)
            inp = torch.tensor(inp_np, dtype=out.dtype, device=out.device)
        return inp

    @staticmethod
    def _torch_dct(x: Tensor) -> Tensor:
        """PyTorch implementation of DCT using FFT.

        Adapted from https://github.com/zh217/torch-dct/blob/master/torch_dct/_dct.py/.

        Parameters
        ----------
        x : Tensor
            The input tensor.

        Returns
        -------
        Tensor
            The DCT of the input tensor.

        """

        N = x.size(dim=-1)
        v = torch.cat([x[..., ::2], x[..., 1::2].flip(dims=[-1])], dim=-1)
        Vc = fft(v, dim=-1)

        k = torch.arange(N, dtype=x.dtype, device=x.device)
        Vc *= 2 * torch.exp(-1j * np.pi * k / (2 * N))

        Vc[..., 0] /= np.sqrt(N) * 2
        Vc[..., 1:] /= np.sqrt(N / 2) * 2

        return Vc.real

    @staticmethod
    def _torch_idct(X: Tensor) -> Tensor:
        """PyTorch implementation of inverse DCT using inverse FFT.

        Adapted from https://github.com/zh217/torch-dct/blob/master/torch_dct/_dct.py/.

        Parameters
        ----------
        X : Tensor
            The DCT of the input tensor.

        Returns
        -------
        Tensor
            The input tensor.

        """
        N = X.size(dim=-1)
        X_v = X / 2

        X_v[..., 0] *= np.sqrt(N) * 2
        X_v[..., 1:] *= np.sqrt(N / 2) * 2

        k = torch.arange(N, dtype=X.dtype, device=X.device) * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V_t_r = X_v
        V_t_i = torch.cat([X_v[..., :1] * 0, -X_v.flip(dims=[-1])[..., :-1]], dim=-1)

        V_r = V_t_r * W_r - V_t_i * W_i
        V_i = V_t_r * W_i + V_t_i * W_r

        V = torch.complex(real=V_r, imag=V_i)

        v = ifft(V, dim=-1).real
        x = v.new_zeros(v.size())
        x[..., 0::2] += v[..., : N - (N // 2)]
        x[..., 1::2] += v.flip(dims=[-1])[..., : N // 2]

        return x

    @property
    def inp_dim(self) -> int:
        return self.length

    @property
    def out_dim(self) -> int:
        return self.length
