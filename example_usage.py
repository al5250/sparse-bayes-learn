import numpy as np
import time
import torch
from torch import Tensor

from sblearn.operators import DenseMatrix, DiscreteCosine1D, Undersampling, Sequential
from sblearn.model import SBLModel


def main():
    # Set seed for reproducibility
    SEED = 1234  # Can change to None for true randomness
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Simulate data and dictionary
    noise_std = 0.01
    signal, data, operator = simulate_data(
        dim_signal=4096,
        dim_data=1024,
        num_nonzero=256,
        noise_std=noise_std,
        device="cpu",  # Can change this to "cuda" if GPU is available
        use_dense=False,  # True = Dense dictionary; False = DCT dictionary
    )

    # Create logger
    def logger(t: int, alpha: Tensor, mu: Tensor, sigma: Tensor):
        if t % 10 == 0:
            err = torch.norm(signal - mu) / torch.norm(signal)
            print(f"Iter {t:02d} | Error {err.cpu().item() * 100:>6.2f}%")

    # # Run EM inference
    print("***** EM Inference *****")
    t = time.time()
    model_em = SBLModel.with_em(num_iters=50, noise_precision=1 / noise_std**2)
    model_em.fit(data, operator, logger)
    em_time = time.time() - t
    print(f"Total Time: {em_time:.2f} seconds\n")

    # Run CoFEM inference
    print("***** CoFEM Inference *****")
    t = time.time()
    model_cofem = SBLModel.with_cofem(
        num_iters=50,
        noise_precision=1 / noise_std**2,
        num_probes=10,
        cg_tol=1e-7,
        max_cg_iters=400,
        precondition=True,
    )
    model_cofem.fit(data, operator, logger)
    cofem_time = time.time() - t
    print(f"Total Time: {cofem_time:.2f} seconds\n")


def simulate_data(dim_signal, dim_data, num_nonzero, noise_std, device, use_dense):
    op_type = "dense" if use_dense else "DCT"
    print(
        f"Simulating latent sparse signal with {dim_signal} dimensions...\n"
        f"Generating data with {dim_data} dimensions from {op_type} matrix...\n"
    )

    # Simulate signal
    dense_idx = torch.tensor(
        np.random.choice(dim_signal, size=num_nonzero, replace=False), device=device
    )
    signal = torch.zeros((dim_signal,), device=device)
    signal[dense_idx] = torch.randn(num_nonzero, device=device)

    # Simulate data
    if use_dense:
        # Construct dense dictionary
        proj = (
            1
            / np.sqrt(dim_data)
            * torch.randn(size=(dim_data, dim_signal), device=device)
        )
        operator = DenseMatrix(proj)
    else:
        # Construct discrete cosine transform dictionary
        dct_idx = torch.tensor(
            np.random.choice(dim_signal, size=dim_data, replace=False), device=device
        )
        dct_mask = torch.tensor([False] * dim_signal, device=device)
        dct_mask[dct_idx] = True
        operator = Sequential(
            ops=[
                DiscreteCosine1D(length=dim_signal, use_fft=True),
                Undersampling(dct_mask, zero_out=False),
            ],
            transposed=[True, False],
        )
    data = operator(signal)
    data = data + noise_std * torch.randn(dim_data, device=device)

    return signal, data, operator


if __name__ == "__main__":
    main()
