from typing import Tuple
from torch import Tensor

from sblearn.operators.linop import LinearOperator


class DenseMatrix(LinearOperator):
    """A linear operator defined by a dense matrix of values.

    This is the most general form of a linear operator, but it is not
    necessary if there exist efficient ways to apply the operator
    and its transpose to vectors.

    """

    def __init__(self, matrix: Tensor) -> None:
        """Initializes the operator.

        Parameters
        ----------
        matrix : Tensor
            A tensor with shape (output_dim x input_dim).

        """
        assert len(matrix.size()) == 2
        self._mat = matrix

    def apply(self, inp: Tensor) -> Tensor:
        out = (self._mat @ inp.unsqueeze(dim=-1)).squeeze(dim=-1)
        return out

    def transpose(self, out: Tensor) -> Tensor:
        out = (self._mat.T @ out.unsqueeze(dim=-1)).squeeze(dim=-1)
        return out

    @property
    def inp_dim(self) -> Tuple[int]:
        return self._mat.shape[1]

    @property
    def out_dim(self) -> Tuple[int]:
        return self._mat.shape[0]
