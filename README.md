# Handwritten Digit Recognition with LeNet-5

## 📌 Project Overview
This project implements a Convolutional Neural Network (CNN) based on the classic **LeNet-5** architecture to recognize handwritten digits. Trained on the MNIST dataset, the project features a custom-trained AI model integrated into a user-friendly Graphical User Interface (GUI) for real-time digit drawing and prediction.

## 👥 Team Members & Responsibilities
This is a collaborative project dividing the workflow into AI architecture design and Software development:

* **Tien Dinh:** Designed and trained the LeNet-5 model using **PyTorch**. Handled data pipelines, model evaluation, and managed the model weights (`.pth` files). Leveraged **vibe coding** to intuitively write and streamline the deep learning workflows and data processing scripts.
* **Hao Le Minh:** Developed the interactive GUI (`gui.py`) and optimized the inference logic (`improve.py`) to seamlessly bridge the PyTorch model with the user interface.

## 🛠️ Tech Stack
* **Deep Learning Framework:** PyTorch
* **Programming Language:** Python
* **Model Architecture:** LeNet-5 (CNN)

## 🚀 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/tiendinh-hcmut/Handwritten-Digit-Recognition-LeNet5.git
   ```
2. Ensure you have Python and PyTorch installed in your environment.
3. Run the application interface:
```bash
python gui.py
```
