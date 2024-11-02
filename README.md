# Self-Driving Car - Behavior Cloning Training for Predicting Steering Angle Using Deep Learning and Data Augmentation

![Prediction Example](steering.gif)

A machine learning project to predict steering angles for a self-driving car based on input images and driving telemetry. This project uses a deep learning model to interpret driving behavior, specifically focusing on real-time steering angle prediction.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project explores the development of a deep learning model to assist in the autonomous navigation of vehicles by predicting the optimal steering angle based on visual and driving data. By leveraging a Convolutional Neural Network (CNN), we can teach the model to interpret the road and suggest steering corrections in real-time.

## Features

- Predicts steering angle based on input images and driving data
- Uses a CNN-based architecture for real-time decision-making
- Provides tools for data visualization and model performance evaluation

## Dataset

The dataset consists of images and telemetry data such as steering angles from a self-driving car. Each sample includes:
- Input image: Captures the forward-facing view from the car
- Telemetry: Recorded steering angle associated with each image

Dataset link : https://www.kaggle.com/datasets/andy8744/udacity-self-driving-car-behavioural-cloning/data.

## Model Architecture

This project implements a Convolutional Neural Network (CNN) inspired by Nvidia's architecture for self-driving cars, optimized for predicting steering angles from images and relevant sensor data. Key layers include:
1. Convolutional layers to capture spatial hierarchies in image data
2. Fully connected layers to map image features to steering angles
3. ReLU activations to introduce non-linearity
4. Dropout layers to prevent overfitting

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/self-driving-car-steering-prediction.git
   cd self-driving-car-steering-prediction
   ```

2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

## Results

### Training Performance
Details on training accuracy, loss, and performance metrics over time.

### Model Evaluation
Visualizations of steering angle predictions vs. ground truth, along with metrics like Mean Absolute Error (MAE) to assess model performance.

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the project
2. Create a feature branch (`git checkout -b feature/NewFeature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to replace placeholders and adjust this template as needed! Let me know if you'd like any specific details included.
