# Sparse-Bayes-Learn

This is a Python package for sparse Bayesian learning (SBL).  It is powered by [PyTorch](https://pytorch.org/), which enables computation to be performed on both CPU and GPU hardware.  This package accompanies the following papers:

- Lin, A., Song, A. H., Bilgic, B., & Ba, D. (2022). [Covariance-free sparse Bayesian learning](https://ieeexplore.ieee.org/document/9807393). In *IEEE Transactions on Signal Processing*.

- Lin, A., Song, A. H., Bilgic, B., & Ba, D. (2022). [High-dimensional sparse Bayesian learning without covariance matrices](https://ieeexplore.ieee.org/document/9746177/). In *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*.

If our code is useful for your research, please consider citing these publications.  Please direct any questions to `alin@seas.harvard.edu`.

## Code Organization

This package provides different inference algorithms for SBL, implemented in the `sblearn/inference` directory.  The different options are
1. **Expectation-Maximization (EM)**: the original algorithm for SBL
2. **Covariance-Free Expectation-Maximization (CoFEM)**: our newly proposed time/space-efficient algorithm that significantly accelerates EM in high dimensions.  This is the main subject of the aforementioned papers.

To maximally accelerate EM, CoFEM relies on the ability to perform fast matrix-vector multiplications with linear operators.  In the `sblearn/operators` directory, we implement several different types of operators, including the generic dense matrix as well as more structured matrices (e.g. convolution, discrete cosine transform, undersampling).  You can add your own custom structured matrices by extending the `sblearn/inference/LinearOperator` base class.


## Package Usage

To use this package, execute the following: 
```bash
git clone https://github.com/al5250/sparse-bayes-learn.git
cd sparse-bayes-learn
pip install -r requirements.txt
```
This will download the source code and ensure that all the dependencies are installed.  Our code was tested with Python version 3.8 (earlier versions may also work, but have not been tested).

The main entry point for running SBL is the `SBLModel` object in the `sblearn/model.py` file.  The script `example_usage.py` provides an example of how to use this object, comparing the relative speeds of EM versus CoFEM.  You can run it with the following command:
```python
python example_usage.py 
```   
This will generate the following example output (exact errors and computation times may vary depending on your machine):
```
Simulating latent sparse signal with 4096 dimensions...
Generating data with 1024 dimensions from DCT matrix...

***** EM Inference *****
Iter 00 | Error  86.36%
Iter 10 | Error  43.63%
Iter 20 | Error  13.37%
Iter 30 | Error   4.18%
Iter 40 | Error   3.26%
Iter 50 | Error   3.34%
Total Time: 83.62 seconds

***** CoFEM Inference *****
Iter 00 | Error  86.37%
Iter 10 | Error  43.28%
Iter 20 | Error  11.93%
Iter 30 | Error   3.78%
Iter 40 | Error   3.31%
Iter 50 | Error   3.44%
Total Time: 2.76 seconds
```  
You can change the variables in the `example_usage.py` script to run the simulation with different signal sizes, data sizes, dictionaries (i.e. dense vs. structured), algorithm parameters (e.g. number of CG steps, number of probe vectors), and hardware (i.e. CPU vs. GPU).

