# 🧠 Handwritten Character Recognition (0–9 + A–Z)

This project is a deep learning-based web app that can recognize **handwritten digits and English uppercase letters** using a Convolutional Neural Network (CNN). The app is built with **TensorFlow**, trained on **MNIST** and **EMNIST** datasets, and deployed via **Gradio** for an interactive user interface.

---

## 🌐 Live Demo

🎯 Try the app live here:  
🔗 [Click to Open App](https://rsltfrmr-rslt2.hf.space/?logs=container&__theme=system&deep_link=c8u_dy0XO7k)

---

## 📌 Features

- ✅ Recognizes handwritten digits `0–9`
- ✅ Recognizes uppercase letters `A–Z`
- ✅ Combined training on MNIST + EMNIST
- ✅ Live drawing interface using Gradio `Sketchpad`
- ✅ Easy to run locally (just one file)

---

## 🧠 Model Architecture

A simple CNN with:
- 2 Convolutional layers
- 2 MaxPooling layers
- 1 Dense hidden layer with dropout
- Final softmax output layer (36 classes)

---

## 🗃️ Datasets Used

| Dataset         | Characters | Source                       |
|-----------------|------------|------------------------------|
| MNIST           | 0–9        | Built-in in TensorFlow       |
| EMNIST Letters  | A–Z        | [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/emnist) |

---

## 🚀 Getting Started

### 🔧 Requirements

- Python 3.7+
- TensorFlow
- Gradio
- OpenCV
- TensorFlow Datasets

Install all dependencies:

```bash
pip install -r requirements.txt

