# DeepLearning
This repo is a C++ implementation of various neurual networks(including feed forward, restricted Boltzmann machine, CNN, RNN, LSTM) for deep learning purpose. We use Armadillo C++ linear algebra library (http://arma.sourceforge.net/) to support our linear algebra operation in the CPU. We also have our GPU linear algebra library(wrapping cuBlas) to speed-up matrix/vector operations. Our implmentation supports seamless switch from CPU to GPU using a single Flag.   

## Requirement & Dependency
#### Compiler: GNU g++ > 4.8
#### Armadillo linear algebra library [Link] (http://arma.sourceforge.net/)
#### Cuda toolkit [link] (https://developer.nvidia.com/cuda-toolkit)
#### Boost [link] (http://www.boost.org/ )
#### Gtest [link] (https://code.google.com/p/googletest/)
#### Google protocol buffer [link] (https://developers.google.com/protocol-buffers/)

