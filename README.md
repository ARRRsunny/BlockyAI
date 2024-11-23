# BlockyAI

BlockyAI is a drag-and-drop GUI application that provides an intuitive way to design, visualize, and generate Python code for building and training machine learning models using TensorFlow/Keras. It simplifies the process of creating neural networks by allowing users to construct models using pre-defined blocks, which represent various neural network layers and functions.

## Features

- **Drag-and-Drop Interface**: Easily create machine learning models by dragging blocks representing layers onto the canvas.
- **Code Generation**: Automatically generate Python code for the model you design, including dataset loading, preprocessing, and model training.
- **Model Visualization**: Visualize the architecture of the model in real-time as you construct it.
- **Real-Time Editing**: Adjust settings such as the number of units in a Dense layer or the size of a Resizing layer directly in the interface.
- **Support for Common Layers**:
  - Dense Layer
  - Conv2D Layer
  - Flatten
  - Activation
  - Resizing Layer
  - Dropout
  - BatchNormalization
  - AveragePooling2D
  - MaxPooling2D
  - Output Layer
- **Settings Configuration**: Customize the dataset, optimizer, batch size, epochs, and learning rate directly within the application.
- **Run Generated Code**: Execute the generated Python script directly from the application.

## How to Use

### Prerequisites

Make sure you have the following installed on your system:

- **Python 3.10 or above**
- **TensorFlow 2.17 or higher**
- **OpenCV 4.10 or higher**
- **Tkinter** (usually included with Python installations)
- **Numpy 1.26 or higher**

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ARRRsunny/BlockyAI.git
   cd BlockyAI
