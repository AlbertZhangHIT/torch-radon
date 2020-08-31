# Torch Radon 
PyTorch implementation of Radon transform. Right now only 2-dimentional case on CPU is supported. Contributions to higher dimentional cases and GPU cases are welcome.

# Motivation
The motivation of this project is the disagreement of the inverse radon transform in scikit-image implementation with MATLAB (refer to issue [#3742](https://github.com/scikit-image/scikit-image/issues/3742)).

# Requirements
The requirements for the conda environment in which I have tested this code are started in `requirements.txt`. The main dependencies are

    1. python >= 3.6.2
    2. torch >= 1.0
    3. numpy >= 1.16.2
    4. scipy >= 1.2.1
    5. scikit-image >= 0.16.2

# Install from source
    python setup.py install

# Usage
some examples are provided in [tests](https://github.com/AlbertZhangHIT/torch-radon/tree/master/tests). The results from [phantom image](https://github.com/AlbertZhangHIT/torch-radon/tree/master/tests/test_phantom.py) are presented in following figures.

![](/tests/test_circular.png)

![](/tests/test_phantom.png)

The comparison with the signogram and reconstruction from MATLAB on phantom image is provided by [test_comparison.py](https://github.com/AlbertZhangHIT/torch-radon/tree/master/tests/test_comparison.py):

    sinogram rms error: 0.42, mae error: 0.0169
    FBP rms error: 0.00805, mae error: 0.00601

# License
MIT License see [LICENSE](https://github.com/AlbertZhangHIT/torch-radon/tree/master/LICENSE).