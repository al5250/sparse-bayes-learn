from typing import List, Optional

import torch
from torch import Tensor

from sblearn.operators.linop import LinearOperator


class Sequential(LinearOperator):
    """Combines a list of operators into a single operator.

    Any product of linear operators is itself a linear operator.  This
    object encapsulates a list of operators by chaining their apply
    methods in sequence.

    """

    def __init__(
        self, ops: List[LinearOperator], transposed: Optional[List[bool]] = None
    ) -> None:
        """Initializes the sequential object.

        Parameters
        ----------
        ops : List[LinearOperator]
            The list of operators to be applied in order.
        transposed : List[bool], optional
            A list of whether or not to transpose the corresponding
            operators.

        """
        self.ops = ops
        if transposed is None:
            transposed = [False] * len(self.ops)
        else:
            assert len(ops) == len(transposed)
        self.transposed = transposed

    def apply(self, inp: Tensor) -> Tensor:
        """Applies the sequence of operators to an input."""
        out = inp
        for dic, tposed in zip(self.ops, self.transposed):
            if tposed:
                out = dic.T(out)
            else:
                out = dic(out)
        return out

    def transpose(self, out: Tensor) -> Tensor:
        """Applies the reversed sequence of transposed operators."""
        inp = out
        for dic, tposed in zip(self.ops[::-1], self.transposed[::-1]):
            if tposed:
                inp = dic(inp)
            else:
                inp = dic.T(inp)
        return inp

    @property
    def inp_dim(self) -> int:
        return self.ops[0].out_dim if self.transposed[0] else self.ops[0].inp_dim

    @property
    def out_dim(self) -> int:
        return self.ops[-1].inp_dim if self.transposed[-1] else self.ops[-1].out_dim
