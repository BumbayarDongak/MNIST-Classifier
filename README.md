# MNIST-Classifier

A neural network built with **Python and TensorFlow/Keras** for recognizing handwritten digits from the MNIST dataset.

The project demonstrates the basic workflow of training and evaluating an image classification model.

## 🚀 Features

- 🔢 Recognizes handwritten digits from 0 to 9
- 🧠 Neural network based on TensorFlow/Keras
- 📚 Uses the MNIST dataset
- 🖼️ Supports image preprocessing
- 📊 Model training and evaluation
- 💾 Saves the trained model
- 📷 Can be used with handwritten digit images

## 🛠️ Technologies

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- PIL / Pillow

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/mnist-classifier.git
cd mnist-classifier
```

Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install tensorflow numpy matplotlib pillow
```

Or:

```bash
pip install -r requirements.txt
```

## ▶️ Running the project

Start the training script:

```bash
python train.py
```

After training, the model is saved to a file such as:

```text
mnist_model.keras
```

To make a prediction:

```bash
python predict.py
```

## 🧠 How It Works

The project follows these steps:

```text
MNIST Dataset
      ↓
Image preprocessing
      ↓
Neural Network
      ↓
Model training
      ↓
Model evaluation
      ↓
Saved model
      ↓
Digit prediction
```

Images are normalized before being passed to the neural network.

## 📁 Project Structure

```text
mnist-classifier/
│
├── train.py
├── predict.py
├── mnist_model.keras
├── requirements.txt
└── README.md
```
