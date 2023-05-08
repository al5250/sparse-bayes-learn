from typing import Optional, Callable
import warnings

from torch import Tensor

from sblearn.inference import InferenceAlgorithm, ExpMax, CovFreeExpMax
from sblearn.operators import LinearOperator


class SBLModel(object):
    """Sparse Bayesian learning model wrapper.

    A stateful object that runs an SBL inference algorithm and saves
    the outputs (alpha, mu, sigma) as attributes.

    """

    def __init__(self, alg: InferenceAlgorithm) -> None:
        self.alg = alg
        self.alpha = None
        self.mu = None
        self.sigma = None

    @classmethod
    def with_em(
        cls,
        num_iters: int,
        noise_precision: float = 1e6,
        init_prior_precision: float = 1,
        non_negative: bool = False,
    ):
        alg = ExpMax(num_iters, noise_precision, init_prior_precision, non_negative)
        return cls(alg)

    @classmethod
    def with_cofem(
        cls,
        num_iters: int,
        noise_precision: float = 1e6,
        init_prior_precision: float = 1,
        num_probes: int = 10,
        cg_tol: float = 1e-5,
        max_cg_iters: int = 1000,
        non_negative: bool = False,
        precondition: bool = False,
    ):
        alg = CovFreeExpMax(
            num_iters,
            noise_precision,
            init_prior_precision,
            num_probes,
            cg_tol,
            max_cg_iters,
            non_negative,
            precondition,
        )
        return cls(alg)

    def fit(
        self,
        y: Tensor,
        phi: LinearOperator,
        logger: Optional[
            Callable[[int, Tensor, Tensor, Tensor, Optional[Tensor]], None]
        ] = None,
    ) -> None:
        """Fit the SBL model to data by running the inference algorithm."""
        if self.alpha is not None:
            warnings.warn(
                "This model has already been fit to data. By calling fit "
                "again, you are overriding previously saved attributes."
            )
        self.alpha, self.mu, self.sigma = self.alg.run(y, phi, logger)
