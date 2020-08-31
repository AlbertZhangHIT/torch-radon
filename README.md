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
some examples are provided in [tests](https://github.com/AlbertZhangHIT/torch-radon/tree/master/tests). The results are presented in following figures.

![](https://github.com/AlbertZhangHIT/torch-radon/tree/master/tests/test_circular.png)

![](https://github.com/AlbertZhangHIT/torch-radon/tree/master/tests/test_phantom.png)



# License
MIT License see (https://github.com/AlbertZhangHIT/torch-radon/tree/master/LICENSE)