from abc import ABC, abstractmethod
from typing import Tuple, Callable, Optional

from torch import Tensor

from sblearn.operators.linop import LinearOperator


class InferenceAlgorithm(ABC):

    """Base class for SBL inference algorithm.

    Given data, objects in this class learn the parameters of a SBL
    model using a particular algorithm (e.g. EM or CoFEM).

    """

    @abstractmethod
    def run(
        self,
        y: Tensor,
        phi: LinearOperator,
        logger: Optional[Callable[[int, Tensor, Tensor, Tensor], None]],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Execute the inference algorithm.

        This is the main method for running sparse Bayesian learning.
        Different inference algorithms vary in how they optimize the
        SBL objective through this method.

        Parameters
        ----------
        y : Tensor
            The observed data vector with dimension N.  The computation
            will be performed using the device of y (i.e. "cpu" or
            "cuda").
        phi : LinearOperator
            The dictionary that is assumed to have generated y.  Output
            dimension must be N to match y.  Input dimension is D.
        logger : Callable[[int, Tensor, Tensor, Tensor]], optional
            A function that can log the progress of the algorithm over
            time.  Takes (iteration, alpha, mu, sigma) as inputs.

        Returns
        -------
        Tensor
            The point estimate for alpha.
        Tensor
            The posterior mean.
        Tensor
            The posterior (co)variance.

        """
        pass
