# Question 5: Neural Networks – Regularization, Transfer Learning, and Interpretability

## 🎯 Objective
The objective of this task is to design and train deep neural networks with advanced regularization, compare their performance against transfer learning models, and apply interpretability and robustness techniques.

## 📊 Dataset
* **Dataset Used:** CIFAR-10.

## 🛠️ Implementation Tasks

### 1. Model Architectures
Implement the following architectures from scratch using PyTorch or TensorFlow:
* **Deep MLP:** At least 4 hidden layers with 512 neurons each, including batch normalization, dropout, and early stopping.
* **CNN:** At least 3 convolutional layers with pooling, dropout, batch normalization, and data augmentation (random crops, flips, rotations).
* **Transfer Learning:** Fine-tuning a pre-trained model (e.g., ResNet18 or VGG16) by updating the last 2 layers.

### 2. Optimization & Evaluation
* **Hyperparameter Tuning:** Utilizing Optuna or Ray Tune to optimize learning rate, batch size, dropout rate, and weight decay.
* **Metrics:** Accuracy, Macro-F1, Top-5 error rate, Confusion Matrix, and Per-class recall.
* **Visualizations:** Training/validation loss and accuracy curves.

### 3. Interpretability Methods
Applying techniques to explain model decisions:
* **Grad-CAM:** Visualizing attention maps for the CNN on misclassified examples.
* **LIME or SHAP:** Explaining individual predictions for the MLP.
* **Error Analysis:** Providing Grad-CAM heatmaps for at least 10 misclassified examples per model to discuss failure modes.

### 4. Adversarial Robustness
* **Attack Methods:** Testing models using FGSM (Fast Gradient Sign Method) or PGD (Projected Gradient Descent).
* **Comparison:** Reporting accuracy under attack to compare the robustness of the MLP, CNN, and pre-trained models.

## 🔍 Analysis & Discussion
* **Regularization Impact:** Discussion on overfitting and the effectiveness of batch norm and dropout.
* **Trade-offs:** A concluding analysis on the trade-offs between model complexity, interpretability, and robustness.
