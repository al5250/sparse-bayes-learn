from typing import Tuple, Optional, Callable
import warnings

import torch
from torch.distributions import Bernoulli
from torch import Tensor
import numpy as np

from sblearn.operators import LinearOperator
from sblearn.inference.algorithm import InferenceAlgorithm


class CovFreeExpMax(InferenceAlgorithm):
    """Covariance-Free Expectation-Maximization (CoFEM) algorithm.

    This algorithm is more time/space efficient than EM in high
    dimensions.  The key novelty is a simplified E-Step that quickly
    estimates the mean and variances of the posterior without computing
    the entire covariance matrix.

    Notation follows (Lin et al., 2022, "Covariance-free sparse
    Bayesian learning").

    """

    def __init__(
        self,
        num_iters: int,
        beta: float,
        alpha_init: float,
        num_probes: int,
        cg_tol: float,
        max_cg_iters: int,
        non_negative: bool,
        precondition: bool,
    ) -> None:
        """Initializes the CoFEM algorithm.

        Parameters
        ----------
        num_iters : int
            Number of iterations to run CoFEM.
        beta : float
            The precision (inverse variance) of the observation noise.
        alpha_init : float
            Initial value for all elements of alpha.
        num_probes : int
            Number of probes for variance estimation.
        cg_tol : float
            Tolerance level for exiting the conjugate gradient
            algorithm.
        max_cg_iters : int
            Maximum number of conjugate gradient iterations per E-Step.
        non_negative : bool
            Whether or not to use a rectified Gaussian prior to limit
            the posterior to only have mass on non-negative elements.
            Set this to true only if you are confident the recovered
            sparse signal is non-negative.
        precondition : bool
            Whether or not to use the diagonal preconditioner of
            (Lin et al., 2022) designed for matrices satisfying the
            restricted isometry property (RIP).

        """
        self.num_iters = num_iters
        self.beta = beta
        self.alpha_init = alpha_init
        self.num_probes = num_probes
        self.cg_tol = cg_tol
        self.max_cg_iters = max_cg_iters
        self.non_negative = non_negative
        self.precondition = precondition

    def run(
        self,
        y: Tensor,
        phi: LinearOperator,
        logger: Optional[Callable[[int, Tensor, Tensor, Tensor], None]] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Execute the CoFEM algorithm.

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
            An estimate of the posterior variances with shape (D,).

        """
        # Initialize variables
        alpha = self.alpha_init * torch.ones(
            phi.inp_dim, dtype=y.dtype, device=y.device
        )
        phi_T_y = phi.T(y)

        # Start algorithm
        mu, sigma_diag, cg_converge_iter = self._estep(alpha, phi, phi_T_y)
        if logger is not None:
            logger(0, alpha, mu, sigma_diag)

        for t in range(self.num_iters):
            alpha_new = self._mstep(mu, sigma_diag)
            mu_new, sigma_diag_new, cg_converge_iter = self._estep(
                alpha_new, phi, phi_T_y
            )

            # Check if conjugate gradient failed to converge
            if cg_converge_iter is None:
                warnings.warn(
                    f"Exiting CoFEM inference algorithm at step {t} instead of "
                    f"step {self.num_iters} because of failed CG convergence. "
                    f"Try increasing max_cg_iters to run for more steps or "
                    f"increasing num_probes for better variance estimation.",
                    stacklevel=1,
                )
                break

            alpha, mu, sigma_diag = alpha_new, mu_new, sigma_diag_new
            if logger is not None:
                logger(t + 1, alpha, mu, sigma_diag)

        return alpha, mu, sigma_diag

    def _estep(
        self,
        alpha: Tensor,
        phi: LinearOperator,
        phi_T_y: Tensor,
    ) -> Tuple[Tensor, Tensor, Optional[int]]:
        """Execute one iteration of the simplified E-Step.

        Parameters
        ----------
        alpha : Tensor
            The vector of precisions with shape (D,).
        phi : LinearOperator
            The dictionary operator.
        phi_T_y : Tensor
            The dictionary's transpose applied to the data with shape
            (D,).

        Returns
        -------
        Tensor
            The posterior mean of the current iteration with shape (D,).
        Tensor
            The posterior variances of the current iteration with
            shape (D,).
        int, optional
            The number of iterations for conjugate gradient to converge.
            Returns None if CG fails to converge.

        """
        # Sample probe vectors and assemble inputs to linear solver
        phi_T_y = phi_T_y.unsqueeze(dim=0)
        probes = self._samp_probes(
            size=(self.num_probes, phi.inp_dim),
            device=phi_T_y.device,
            dtype=phi_T_y.dtype,
        )
        b = torch.cat([phi_T_y, probes], dim=0)
        A = lambda x: phi.T(phi(x)) + alpha / self.beta * x

        # Set preconditioner if required
        if self.precondition:
            M_inv = lambda x: 1 / (1 + alpha / self.beta) * x
        else:
            M_inv = lambda x: x

        # Run parallelized conjugate gradient
        x, converge_iter = self._conj_grad(
            A, b, M_inv=M_inv, max_iters=self.max_cg_iters, tol=self.cg_tol
        )

        # Unpack conjugate gradient outputs
        mu = x[0]
        sigma_diag = 1 / self.beta * (probes * x[1:]).mean(dim=0).clamp(min=0)
        return mu, sigma_diag, converge_iter

    def _mstep(self, mu: Tensor, sigma_diag: Tensor) -> Tensor:
        """Execute one iteration of the maximization step.

        Parameters
        ----------
        mu : Tensor
            The current mean vector with shape (D,).
        sigma_diag : Tensor
            The current estimate of the covariance diagonal with shape
            (D,).

        Returns
        -------
        Tensor
            The udpated alpha vector with shape (D,).

        """
        expected_x2 = mu**2 + sigma_diag
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
    def _samp_probes(
        size: Tuple[int, ...], device: str = "cpu", dtype: str = "float"
    ) -> Tensor:
        """Sample Rademacher probes.

        Parameters
        ----------
        size : Tuple[int, ...]
            The shape of the probes to sample.
        device : str
            The device to store the tensor.
        dtype : str
            The data type of the tensor.

        Returns
        -------
        Tensor
            The tensor of random probe variables.

        """
        p = torch.tensor(0.5, device=device, dtype=dtype)
        z = 2 * Bernoulli(probs=p).sample(size) - 1
        return z

    @staticmethod
    def _conj_grad(
        A: Callable[[Tensor], Tensor],
        b: Tensor,
        M_inv: Optional[Callable[[Tensor], Tensor]] = None,
        max_iters: int = 1000,
        tol: float = 1e-10,
    ) -> Tuple[Tensor, Optional[int]]:
        """Parallelized implementation of conjugate gradient algorithm.

        Parameters
        ----------
        A : Callable[[Tensor], Tensor]
            The operator corresponding to a positive semi-definite
            matrix that inputs a tensor of shape (D,) and outputs a
            tensor of shape (D,).
        b : Tensor
            A tensor of shape (Q, D) corresponding to the multiple
            right hand sides of conjugate gradient.  Computation is
            parallelized over Q (i.e. the batch dimension).
        M_inv : Callable[[Tensor], Tensor], optional
            A function corresponding to the preconditioner for
            conjugate gradient.  If None, then no preconditioner is
            used.
        max_iters : int
            The maximum number of iterations for conjugate gradient.
        tol : float
            The tolerance level to assess convergence of CG.

        Returns
        -------
        Tensor
            The tensor of random probe variables.
        int, optional
            The number of iterations for conjugate gradient to converge.
            Returns None if CG fails to converge.

        """
        x = torch.zeros_like(b)
        r = b
        z = r if M_inv is None else M_inv(r)
        p = z
        rz = torch.sum(r * z, dim=-1, keepdim=True)
        t = 0

        for t in range(max_iters):
            Ap = A(p)
            pAp = torch.sum(p * Ap, dim=-1, keepdim=True)
            alpha = rz / pAp
            x = x + alpha * p
            r = r - alpha * Ap

            eps = (torch.norm(r) / torch.norm(b)) ** 2
            if eps < tol:
                return x, t

            rz_old = rz
            z = r if M_inv is None else M_inv(r)
            rz = torch.sum(r * z, dim=-1, keepdim=True)
            beta = rz / rz_old
            p = z + beta * p

        warnings.warn(
            f"Conjugate gradient failed to converge to a tolerance level of {tol:.3e} "
            f"after {max_iters} iterations.  Exiting with an error of {eps:.3e}.",
            stacklevel=1,
        )
        return x, None
