# Deep Learning Inference Engine

A lightweight, high-performance implementation of a Multilayer Perceptron (MLP) neural network written entirely from scratch in modern C++. 

Tons of machine learning projects rely on high-level frameworks like TensorFlow or PyTorch to abstract away the underlying mathematics; this project was built to demonstrate a fundamental understanding of the linear algebra, memory management, and computational bottlenecks that drive modern AI-accelerated hardware. 

By modeling the exact matrix operations that run on GPUs and advanced AI architectures, this engine simulates real-time inference without any external dependencies.

## Key Features

* Zero Dependencies: Architected completely from scratch using standard C++ libraries.
* Custom Linear Algebra: Includes a custom Matrix class engineered to handle dynamic matrix multiplication and vector addition.
* Forward Propagation: A fully functional feedforward mechanism that accurately simulates real-time inference workloads.
* Core Algorithms: Implements foundational machine learning activation functions, including ReLU and Sigmoid.
* Object-Oriented Design: Built with a highly modular and scalable architecture to allow for future layer expansion.

## Why C++?

Python is the standard for machine learning research, but C++ remains the industry standard for deploying high-performance production systems. So this project was developed in C++ to prioritize execution speed and direct memory control

## Getting Started

Because this project relies on zero external libraries, compiling and running the engine is straightforward 

### Prerequisites
* standard C++ compiler (GCC, Clang, or MSVC)

### Compilation
Navigate to the repository directory and compile the source code using your preferred compiler:

`g++ -o nn_engine NeuralNetwork.cpp`

### Execution
Run the compiled executable to initialize the network architecture and perform a forward pass:

`./nn_engine`

## Under the Hood

The core computational bottleneck of any artificial intelligence model is matrix multiplication. This engine directly models that constraint. 

When the `feedForward` method is called, the network executes the following sequence:
1. Multiplies the input matrix by the hidden layer weights.
2. Adds the hidden layer biases via element-wise matrix addition.
3. Applies the ReLU activation function to introduce non-linearity.
4. Multiplies the hidden layer output by the final output weights.
5. Adds the output layer biases.
6. Applies a Sigmoid activation function to map the final output to a probability distribution.

## Author

[cite_start]Adnan Jasim Sudheesh [cite: 38]
[cite_start]github.com/adnanjasims [cite: 39]
