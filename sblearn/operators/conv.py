from torch import Tensor
import torch.nn.functional as F
import torch
from torch.fft import fft, ifft

from sblearn.operators.linop import LinearOperator


class Convolution1D(LinearOperator):
    """One-dimensional convolution with a specified kernel.

    To apply the convolution to an input, this object can optionally
    invoke either the conv1d function from the PyTorch library or an
    element-wise product in the Fourier domain through the Convolution
    Theorem.

    """

    def __init__(self, length: int, kernel: Tensor, use_fft: bool = False) -> None:
        """Initializes the convolution.

        Parameters
        ----------
        length : Tensor
            The dimensionality of the data for the convolution.
        kernel : Tensor
            A one-dimensional kernel.  Length of the kernel must not
            exceed length of the data.
        use_fft: bool
            Whether or not to apply the convolution in the Fourier
            domain.  This option is recommended for long kernels with
            length close to the data length.

        """
        if len(kernel.size()) > 1:
            raise ValueError("Kernel must be a 1D tensor.")
        elif len(kernel) > length:
            raise ValueError("Kernel length cannot be longer than data length.")

        self.length = length
        self.kernel = kernel.unsqueeze(dim=0).unsqueeze(dim=0)
        self.use_fft = use_fft

    def apply(self, inp: Tensor) -> Tensor:
        if self.use_fft:
            out = self._fft_apply(inp, self.kernel)
        else:
            inp_padded = F.pad(inp, (self.kernel.size(dim=-1) - 1, 0))
            out = F.conv1d(
                inp_padded, weight=self.kernel.flip(dims=[-1]), groups=inp.size(dim=1)
            )
        return out

    def transpose(self, out: Tensor) -> Tensor:
        if self.use_fft:
            inp = self._fft_transpose(out, self.kernel)
        else:
            out_padded = F.pad(out, (0, self.kernel.size(dim=-1) - 1))
            inp = F.conv1d(out_padded, weight=self.kernel, groups=out.size(dim=1))
        return inp

    @staticmethod
    def _fft_apply(x: Tensor, kernel: Tensor):
        """Use the FFT to apply convolution.

        Parameters
        ----------
        x : Tensor
            The data tensor.
        kernel : Tensor
            The kernel tensor.

        """
        x_padded = F.pad(x, (kernel.size(dim=-1) - 1, 0))
        kernel_padded = F.pad(kernel, (0, x_padded.size(dim=-1) - kernel.size(dim=-1)))
        y = fft(x_padded)
        ky = fft(kernel_padded)
        conv = ifft(y * ky)
        out = conv[..., -x.size(dim=-1) :].real
        return out

    @staticmethod
    def _fft_transpose(x: Tensor, kernel: Tensor):
        """Use the FFT to apply convolution transposed.

        Parameters
        ----------
        x : Tensor
            The data tensor.
        kernel : Tensor
            The kernel tensor.

        """
        x_padded = F.pad(x, (0, kernel.size(dim=-1) - 1))
        kernel_padded = F.pad(
            kernel.flip(dims=[-1]), (0, x_padded.size(dim=-1) - kernel.size(dim=-1))
        )
        y = fft(x_padded)
        ky = fft(kernel_padded)
        conv = ifft(y * ky)
        out = conv[..., -x.size(dim=-1) :].real
        return out

    @property
    def inp_dim(self) -> int:
        return self.length

    @property
    def out_dim(self) -> int:
        return self.length
