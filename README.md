# Deep Learning Algorithms

This repository contains implementations of core **deep learning algorithms** and neural network architectures built from scratch in **Python**.  
It is designed for learning and experimentation — no high-level frameworks like TensorFlow or PyTorch are used, so the math and mechanics of deep learning are fully exposed.

---

## 📌 Features

- **Custom Neural Network Framework**  
  - Manual implementation of forward pass, backpropagation, and gradient updates.  
  - Modular layers (activations, convolutions, pooling, etc.).  

- **Classic Architectures**  
  - Implementation and training of **LeNet** on image datasets.  
  - Implementation of **Elman RNN** (simple recurrent neural network) for sequence modeling.  
  - Extendable to more architectures.  

- **Optimization**  
  - Gradient descent and other optimizers.  
  - Loss functions and evaluation metrics.  

- **Testing & Validation**  
  - Unit tests for network operations.  
  - Dedicated convolution operation tests.  

---

## 📂 Project Structure

Deep-Learning-Algorithms \
├── Data  # Datasets or sample data \
├── Layers # Custom layer implementations \
├── Models # Neural network models (LeNet, Elman RNN, etc.) \
├── Optimization # Optimizers and training utilities \
├── NeuralNetwork.py # Core neural network implementation \
├── TrainLeNet.py # Script to train LeNet \
├── TrainElmanRNN.py # Script to train Elman RNN \
├── NeuralNetworkTests.py # Unit tests for the network \
└── SoftConvTests.py # Tests for convolution ops 


---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/VuKe77/Deep-Learning-Algorithms.git
cd Deep-Learning-Algorithms

Create a virtual environment (recommended):

python3 -m venv venv
source venv/bin/activate     # On Windows: .\venv\Scripts\activate

Install dependencies:

pip install numpy scipy matplotlib
```

## 🚀 Usage
 ``` bash
#Train LeNet
python TrainLeNet.py

#Train Elman RNN
python TrainElmanRNN.py
```
## 🧠 Learning Goals

This project is intended for: \
    - Understanding how deep learning works under the hood. \
    - Experimenting with forward/backpropagation math. \
    - Understanding sequence modeling with Elman RNNs. \
    - Building intuition for neural networks without relying on frameworks. \
    - Serving as a foundation to extend toward more complex architectures (e.g., VGG, ResNet, LSTMs). 

## 👤 Author

    VuKe77
    Zhang-Jiaxin-Cindy
