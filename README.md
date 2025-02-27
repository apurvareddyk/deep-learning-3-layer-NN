# 🚀 Deep Learning - 3-Layer Neural Network Implementations  
### [Colab link](https://colab.research.google.com/drive/1qWyq-oQhJhaeyZ2V5CQkcG7DoRJH4V_7?usp=sharing)
## 📖 Overview  
This repository contains multiple implementations of a **3-layer deep neural network** for **nonlinear regression**, using different deep learning frameworks.  
The goal is to compare **manual backpropagation, PyTorch, and TensorFlow approaches** while ensuring the network has **three hidden layers**, unlike the traditional 2-layer models in Colab examples.  

Additionally, this project focuses on:  
- Using **`tf.einsum()`** instead of `tf.matmul()` for TensorFlow implementations.  
- Implementing **manual backpropagation** in NumPy and PyTorch for better understanding.  
- **Generating synthetic data** using a **3-variable nonlinear equation**.  
- **Visualizing data in 4D plots** to observe relationships between variables.  

---

## 🔹 Implementations  
Each notebook in this repository implements a **3-layer deep neural network** using different techniques, progressively increasing the level of abstraction.

### **📝 Implementations in Different Frameworks**  

### **1️⃣ NumPy (From Scratch) - Manual Backpropagation**  
- **What I did:**  
  - Implemented a **fully manual neural network** using NumPy arrays.  
  - **Manually computed gradients** using the **chain rule** and updated weights without any libraries.  
  - Applied **nonlinear activation functions** and optimized parameters via **gradient descent**.  
  - **Why it matters:** Helps in understanding how neural networks truly work **without relying on frameworks**.  

---

### **2️⃣ PyTorch (No Built-in Layers) - Fully Manual Implementation**  
- **What I did:**  
  - Used PyTorch **without** using `torch.nn.Module` or built-in layers.  
  - Manually defined **weight matrices and activation functions**.  
  - Implemented **forward propagation, loss calculation, and backpropagation manually**.  
  - **Why it matters:** Demonstrates how PyTorch operates at a low level before leveraging its built-in capabilities.  

---

### **3️⃣ PyTorch (Class-Based Model) - Using `torch.nn.Module`**  
- **What I did:**  
  - Created a **PyTorch class** using `torch.nn.Module`.  
  - Defined **layers, forward pass, and weight initialization** within a structured model.  
  - Used **automatic differentiation (`autograd`) for gradient computation**.  
  - **Why it matters:** Introduces **object-oriented design** in PyTorch and simplifies model structuring.  

---

### **4️⃣ PyTorch Lightning - Automated Training Loops**  
- **What I did:**  
  - Used **PyTorch Lightning** to **automate training** with less boilerplate code.  
  - Defined a **clean, modular, and reusable model** using `LightningModule`.  
  - Handled **loss calculation, optimizer selection, and backpropagation** with PyTorch Lightning.  
  - **Why it matters:** Speeds up deep learning workflows and makes the training process more **scalable and manageable**.  

---

### **5️⃣ TensorFlow (From Scratch) - No High-Level API**  
- **What I did:**  
  - Built a **neural network without `tf.keras`**, manually defining **forward propagation and loss functions**.  
  - Used **`tf.GradientTape()`** to track computations and compute gradients.  
  - Manually updated **weights and biases using an optimizer**.  
  - **Why it matters:** Helps in understanding **how TensorFlow executes low-level operations** before using high-level APIs.  

---

### **6️⃣ TensorFlow (Using Built-in Layers)**  
- **What I did:**  
  - Implemented the same **3-layer neural network** but using **TensorFlow’s built-in layers (`tf.keras.layers.Dense`)**.  
  - Used **automatic differentiation and optimizer updates** handled by TensorFlow.  
  - **Why it matters:** Demonstrates how TensorFlow simplifies deep learning model creation.  

---

### **7️⃣ TensorFlow (Functional API) - Explicit Layer Connections**  
- **What I did:**  
  - Used **`tf.keras.Model` Functional API** to define the neural network explicitly.  
  - Connected **input, hidden, and output layers** explicitly instead of sequentially.  
  - **Why it matters:** Enables **more complex architectures** (e.g., multiple inputs/outputs) while keeping flexibility.  

---

### **8️⃣ TensorFlow (Model API - Subclassing `tf.keras.Model`)**  
- **What I did:**  
  - Created a **custom model class** by subclassing `tf.keras.Model`.  
  - Defined **layers in `__init__()` and forward pass in `call()`**.  
  - **Why it matters:** Gives full control over model design while using TensorFlow’s built-in functionalities.  

---

### **9️⃣ TensorFlow (Sequential API) - Simplified Approach**  
- **What I did:**  
  - Used **`tf.keras.Sequential()`** to create the simplest version of the model.  
  - **Stacked layers in a linear order** with automatic weight initialization.  
  - **Why it matters:** Shows the easiest way to build a deep learning model in TensorFlow.  

---

## 🔹 Key Features  
✔ **3-Layer Deep Neural Network (Unlike Standard 2-Layer Colab Examples)**  
✔ **Uses `tf.einsum()` Instead of `tf.matmul()` for TensorFlow Operations**  
✔ **Manually Implements Backpropagation in NumPy and PyTorch**  
✔ **Generates Synthetic Data Using a 3-Variable Nonlinear Equation**  
✔ **Includes a 4D Plot for Data Visualization**  
✔ **Uses Modern Deep Learning Techniques for Training and Optimization**  

---

## 📄 Walkthrough Video  
A complete walkthrough video explaining the **Colab notebooks, implementation details, and training process** is available:  

📺 **[Video Implementation](www.youtube.com)**  

---

## 📌 Summary  
This repository **compares different deep learning frameworks** while implementing a **3-layer neural network from scratch to high-level APIs**. The goal is to **understand backpropagation, optimization strategies, and API differences in deep learning**.  

