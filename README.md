# BlockyAI

BlockyAI is a drag-and-drop GUI application designed to empower senior developers, AI learners, and educators to easily create, visualize, and execute machine learning models. Inspired by tools like Micro:bit, BlockyAI simplifies the process of building AI models, making it accessible for all — even for primary school students. This tool encourages hands-on learning and teaching of AI concepts in a fun and interactive way.

![BlockyAI](https://github.com/ARRRsunny/BlockyAI/blob/main/asset/image.png)
## Purpose

The primary goal of BlockyAI is to make AI accessible to everyone, regardless of their programming expertise. Whether you're:

- **A senior coder**: Quickly prototype machine learning models and generate Python code.
- **A beginner learning AI**: Experiment with AI concepts without worrying about the technical details of code syntax.
- **An educator**: Teach AI concepts to students using visual blocks, enabling them to build their own models easily.

BlockyAI is an ideal platform for introducing young learners to AI, inspiring a new generation of AI enthusiasts by making complex concepts simple and approachable.

---

## Features

- **Drag-and-Drop Visual Blocks**: Create machine learning models by dragging and connecting blocks that represent neural network layers.
- **Code Generation**: Automatically generates Python code based on the visual model, enabling users to see the real implementation behind their designs.
- **Model Visualization**: Real-time graphical representation of the model's architecture.
- **Educational Focus**: Designed for teaching and learning AI concepts, making it easy for educators to explain and learners to explore.
- **Support for Common AI Layers**:
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
- **Run Models Directly**: Execute the generated Python code directly from the application.
- **Settings Configuration**: Adjust dataset, optimizer, batch size, epochs, and learning rate to customize training.

---

## Why BlockyAI?

BlockyAI lowers the barrier to entry for understanding and experimenting with AI. It’s like using Micro:bit for coding or Scratch for programming — simple, visual, and interactive. With BlockyAI, even primary school students can start building and training their own neural networks, making it the perfect tool for:

- **Students**: Learn AI concepts visually and experiment with models.
- **Teachers**: Teach AI in a classroom setting with an engaging and interactive tool.
- **Professionals**: Prototype models quickly and focus on creativity rather than syntax.

---

## How to Use

### Prerequisites

Ensure the following dependencies are installed:

- **Python 3.10 or above**
- **TensorFlow 2.17 or higher**
- **OpenCV 4.10 or higher**
- **Tkinter** (included with most Python installations)
- **Numpy 1.26 or higher**

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ARRRsunny/BlockyAI.git
   cd BlockyAI
   ```

2. Install the required Python dependencies:
   ```bash
   pip install tensorflow numpy opencv-python
   ```

### Running the Application

1. Run the `server.py` file:
   ```bash
   python server.py
   ```

2. Start building your model:
   - Drag blocks from the **Blocks Holder** to the **Canvas**.
   - Customize settings for individual blocks (e.g., number of units, resizing dimensions).
   - Connect blocks to define the model's flow.

3. Use the **Settings Panel** to configure:
   - Dataset
   - Optimizer
   - Batch size
   - Epochs
   - Learning rate

4. View the model's architecture in the **Model Visualization Area**.

5. Click **Run** in the Code Display Area to generate and execute the Python code for your model.

---

## Block Types

| Block Name                   | Input Field | Resize Field | Deletable | Description                                                                 |
|------------------------------|-------------|--------------|-----------|-----------------------------------------------------------------------------|
| Starting Block               | No          | No           | No        | The entry point of the model. Cannot be deleted.                           |
| Dense Layer                  | Yes         | No           | Yes       | Fully connected layer with customizable number of units.                   |
| Conv2D Layer                 | Yes         | No           | Yes       | Convolutional layer with customizable number of filters.                   |
| Flatten                      | No          | No           | Yes       | Flattens the input to a 1D vector.                                         |
| Activation                   | No          | No           | Yes       | Applies an activation function (default: ReLU).                            |
| Resizing Layer               | No          | Yes          | Yes       | Resizes the input to specified dimensions (width, height).                 |
| AveragePooling2D Layer       | No          | No           | Yes       | Reduces spatial dimensions by taking the average over a pooling window.    |
| MaxPooling2D Layer           | No          | No           | Yes       | Reduces spatial dimensions by taking the maximum over a pooling window.    |
| BatchNormalization Layer     | No          | No           | Yes       | Normalizes the input across the batch.                                     |
| Dropout                      | Yes         | No           | Yes       | Applies dropout regularization with a customizable dropout rate.           |

---

## Example Workflow

### Building a Simple Model

1. Drag the following blocks onto the canvas:
   - **Starting Block**
   - **Conv2D Layer** (set filters to 32)
   - **MaxPooling2D Layer**
   - **Flatten**
   - **Dense Layer** (set units to 128)
   - **Output Layer**

2. Configure the settings:
   - Dataset: `mnist`
   - Optimizer: `Adam`
   - Batch Size: `32`
   - Epochs: `5`
   - Learning Rate: `0.001`

3. Click **Run** to train the model and see predictions.

---

## Code Generation

BlockyAI generates Python code that includes:

- Dataset loading and preprocessing
- Model architecture definition
- Model compilation and training
- Random prediction and result visualization using OpenCV

### Example Generated Code

Here’s a snippet of the kind of code BlockyAI generates:

```python

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
import numpy as np
import cv2

tf.keras.backend.clear_session()
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0
if len(x_train.shape) == 3:
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
num_classes = len(labels)

train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes)
test_one_hot = tf.keras.utils.to_categorical(y_test, num_classes)

def build_model(num_classes):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(28, 28, 1)),
        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        AveragePooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

model = build_model(num_classes)
model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
callbacks = [EarlyStopping(monitor='accuracy', patience=3)]
model.fit(x_train, train_one_hot, batch_size=32, epochs=5, callbacks=callbacks, validation_data=(x_test, test_one_hot))
prediction = model.predict(x_test)

N = np.random.randint(0, high=len(x_test), dtype=int)

print(f'sum: {np.sum(prediction, axis=1)}')
print(f'predict index: {np.argmax(prediction, axis=1)}')
print(f'Predict: {labels[np.argmax(prediction, axis=1)[N]]}')
print(f'Correct: {labels[y_test[N]]}')

image = x_test[N]
if image.shape[-1] == 1:
    image = image.reshape(image.shape)

cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.resizeWindow('img',300,300)
cv2.imshow('img',image)
cv2.waitKey(0)
cv2.destroyAllWindows()

```

---

## Educational Benefits

- **For Students**: Learn AI concepts visually, experiment with neural networks, and see the real-world impact of AI.
- **For Teachers**: A powerful teaching tool to explain AI concepts interactively in classrooms.
- **For Beginners**: A simplified introduction to AI without requiring prior coding experience.

---

## Contributing

We welcome contributions! Feel free to submit issues or pull requests on the [GitHub repository](https://github.com/ARRRsunny/BlockyAI).

---

## Credits

- **Created by**: [@ARRRsunny](https://github.com/ARRRsunny)
- **Inspiration**: Tools like Micro:bit, Scratch, and TensorFlow.

---

**Empowering the next generation of AI enthusiasts, one block at a time!**

