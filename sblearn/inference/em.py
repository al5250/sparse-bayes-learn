from typing import Optional, Tuple, Callable

from torch import Tensor
import torch
import numpy as np

from sblearn.operators import LinearOperator
from sblearn.inference.algorithm import InferenceAlgorithm


class ExpMax(InferenceAlgorithm):
    """Expectation-Maximization (EM) algorithm for SBL inference.

    The EM algorithm alternates between (1) an E-Step that computes
    posterior statistics (mu, sigma) given current alpha estimates and
    (2) an M-Step that optimizes alpha given (mu, sigma).

    Notation follows (Lin et al., 2022, "Covariance-free sparse
    Bayesian learning").

    """

    def __init__(
        self,
        num_iters: int,
        beta: float,
        alpha_init: float,
        non_negative: bool,
    ) -> None:
        """Initializes the EM algorithm.

        Parameters
        ----------
        num_iters : int
            Number of iterations to run EM.
        beta : float
            The precision (inverse variance) of the observation noise.
        alpha_init : float
            Initial value for all elements of alpha.
        non_negative : bool
            Whether or not to use a rectified Gaussian prior to limit
            the posterior to only have mass on non-negative elements.
            Set this to true only if you are confident the recovered
            sparse signal is non-negative.

        """
        self.num_iters = num_iters
        self.beta = beta
        self.alpha_init = alpha_init
        self.non_negative = non_negative

    def run(
        self,
        y: Tensor,
        phi: LinearOperator,
        logger: Optional[Callable[[int, Tensor, Tensor, Tensor], None]] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Execute the EM algorithm.

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
            The learned solution for alpha with shape (D,).
        Tensor
            The posterior mean with shape (D,).
        Tensor
            The posterior covariance with shape (D, D).

        """
        # Initialize variables
        alpha = self.alpha_init * torch.ones(
            phi.inp_dim, dtype=y.dtype, device=y.device
        )
        phi_T_y = phi.T(y)
        phi_mat = self._create_matrix(phi, y.device, y.dtype)
        phi_matTmat = phi_mat.T @ phi_mat
        del phi_mat

        # Iterate E-Steps and M-Steps
        mu, sigma = self._estep(alpha, phi_matTmat, phi_T_y)
        if logger is not None:
            logger(0, alpha, mu, sigma)
        for t in range(self.num_iters):
            alpha_new = self._mstep(mu, sigma)
            del alpha, mu, sigma
            mu_new, sigma_new = self._estep(alpha_new, phi_matTmat, phi_T_y)
            alpha, mu, sigma = alpha_new, mu_new, sigma_new
            if logger is not None:
                logger(t + 1, alpha, mu, sigma)

        return alpha, mu, sigma

    def _estep(
        self, alpha: Tensor, phi_matTmat: Tensor, phi_T_y: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Execute one iteration of the expectation step.

        Parameters
        ----------
        alpha : Tensor
            The current alpha vector with shape (D,).
        phi_matTmat : Tensor
            The dictionary matrix times its transpose with shape (D, D).
        phi_T_y : Tensor
            The transpose dictionary times the data with shape (D,).

        Returns
        -------
        Tensor
            The inferred mean vector with shape (D,).
        Tensor
            The inferred covariance matrix with shape (D, D).

        """
        precision = self.beta * phi_matTmat
        diag_idx = torch.arange(precision.size(dim=-1))
        precision[diag_idx, diag_idx] += alpha
        sigma = torch.inverse(precision, out=precision)
        mu = self.beta * (precision @ phi_T_y.unsqueeze(dim=-1)).squeeze(dim=-1)
        return mu, sigma

    def _mstep(self, mu: Tensor, sigma: Tensor) -> Tensor:
        """Execute one iteration of the maximization step.

        Parameters
        ----------
        mu : Tensor
            The current mean vector with shape (D,).
        sigma : Tensor
            The current covariance matrix with shape (D, D).

        Returns
        -------
        Tensor
            The udpated alpha vector with shape (D,).

        """
        sigma_diag = torch.diagonal(sigma, dim1=-2, dim2=-1)
        expected_x2 = (mu**2) + sigma_diag
        if self.non_negative:
            correction = (
                mu
                * torch.sqrt(sigma_diag / np.pi)
                * torch.exp(-(mu**2) / 2 / sigma_diag)
                / torch.erfc(-mu / torch.sqrt(2 * sigma_diag))
            )
            expected_x2 += correction
        alpha = 1 / (expected_x2)
        return alpha

    @staticmethod
    def _create_matrix(phi: LinearOperator, device: str, dtype: str) -> Tensor:
        """Construct the physical matrix representation of a dictionary.

        Parameters
        ----------
        phi : LinearOperator
            The dictionary object.
        device : str
            The device that the matrix will be constructed on.
        dtype : str
            The data type of the matrix.

        Returns
        -------
        Tensor
            A tensor with shape (N, D) representing phi in matrix form,
            where N is the output dimension and D is the input
            dimension.

        """
        identity = torch.eye(phi.inp_dim, device=device, dtype=dtype)
        mat = phi(identity)
        return mat.T
