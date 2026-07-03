# Cybersecurity Spam Detection with Deep Netts

A binary classification project for detecting spam messages using a feed-forward neural network implemented in **Java** with the **Deep Netts** library.

The project compares several neural network architectures, selects a final model using validation-set performance, evaluates the selected model on an unseen test set, and visualizes training and validation loss across epochs.

---

## Project Overview

The goal is to predict whether a message is spam using five numerical features:

- `num_links`
- `num_words`
- `has_offer`
- `sender_score`
- `all_caps`

Target variable:

- `is_spam`
  - `1` — spam
  - `0` — not spam

The dataset contains **20,000 messages**.

---

## Data Split

The dataset is divided into:

| Dataset | Samples | Percentage |
|---|---:|---:|
| Training set | 14,000 | 70% |
| Validation set | 3,000 | 15% |
| Test set | 3,000 | 15% |

Class distribution was checked after splitting and remained approximately consistent across all three subsets.

To prevent data leakage, the scaler is fitted only on the training set and then applied to the training, validation, and test sets.

---

## Preprocessing

Feature values are normalized using `MinMaxScaler`.

```java
MinMaxScaler scaler = new MinMaxScaler(trainSet);
scaler.apply(trainSet);
scaler.apply(validationSet);
scaler.apply(testSet);
```

---

## Neural Network Configurations

Three feed-forward neural network architectures were trained and compared:

1. `5 → 32 → 16 → 1`
2. `5 → 16 → 1`
3. `5 → 64 → 32 → 1`

All hidden layers use the **ReLU** activation function, while the output layer uses **Sigmoid** activation.

The loss function is **Cross-Entropy**, suitable for binary classification with a sigmoid output.

### Training parameters

| Parameter | Value |
|---|---|
| Optimizer | SGD |
| Learning rate | 0.001 |
| Batch size | 64 |
| Epochs | 50 |
| Random seed | 42 |
| Early stopping | Disabled |
| Loss function | Cross-Entropy |

---

## Validation Results

| Architecture | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| `5 → 32 → 16 → 1` | 95.57% | 77.74% | 75.86% | 76.79% |
| `5 → 16 → 1` | 94.20% | 79.00% | 54.48% | 64.49% |
| `5 → 64 → 32 → 1` | 95.63% | 77.70% | 76.90% | 77.30% |

Although the largest network achieved a slightly higher validation score, the final selected architecture was:

```text
5 → 32 → 16 → 1
```

This model provides almost identical predictive performance while using a simpler architecture with fewer neurons and lower computational complexity.

---

## Final Test Results

The selected network was evaluated only once on the unseen test set.

| Metric | Result |
|---|---:|
| Accuracy | **95.80%** |
| Precision | **77.32%** |
| Recall | **76.19%** |
| F1 Score | **76.75%** |
| Specificity | **97.76%** |
| False Positive Rate | **2.24%** |
| False Negative Rate | **23.81%** |

### Confusion Matrix Values

|  | Predicted Spam | Predicted Not Spam |
|---|---:|---:|
| Actual Spam | 208 | 65 |
| Actual Not Spam | 61 | 2666 |

---

## Example Prediction

The trained model can be wrapped as a binary classifier and used to predict the spam probability of an individual message.

```java
BinaryClassifier<float[]> binClassifier =
        new FeedForwardNetBinaryClassifier(neuralNet);

float[] testEmail = testSet.get(0).getInput().getValues();

Float result = binClassifier.classify(testEmail);

System.out.println("Spam probability: " + result);
```

Example output:

```text
Spam probability: 1.4150928E-4
```

The predicted probability is very close to zero, so the example is classified as not spam.

---

## Loss Curve

Training and validation Binary Cross-Entropy loss were recorded after every epoch.

![Training and validation loss](loss_curve.png)

The losses decreased consistently across all 50 epochs:

| Loss | Epoch 1 | Epoch 50 |
|---|---:|---:|
| Training loss | 0.2295 | 0.1203 |
| Validation loss | 0.2354 | 0.1173 |

The curves remain close and decrease throughout training, which indicates stable learning without an obvious sign of overfitting during the observed epochs.

---

## Visual AI Builder Note

An attempt was made to visualize the experiment using **Deep Netts Visual AI Builder**.

However, the available spam-classification template was created for an older project structure and was not compatible with the current Visual Workflow plugin. The builder produced null-reference errors while linking the dataset, training configuration, and network architecture.

Because of this compatibility issue, the loss curve was generated programmatically using the **XChart** Java library. This approach records the loss directly from the actual model used in the project and saves the result as `loss_curve.png`.

---

## Project Structure

```text
CyberSecuritySpamDetectionDeepNetts/
│
├── datasets/
│   └── spam_detection_dataset.csv
│
├── src/main/java/com/nemanja/spam/
│   ├── SpamClassifierDeepNetts.java
│   └── LossChart.java
│
├── loss_curve.png
├── pom.xml
└── README.md
```

---

## Technologies

- Java 22
- Deep Netts 4.0.0-beta
- Maven
- NetBeans
- XChart
- Java Vector API incubator module

---

## Running the Project

Because Deep Netts uses the Java Vector API, the following VM option is required:

```text
--add-modules jdk.incubator.vector
```

The main class is:

```text
com.nemanja.spam.SpamClassifierDeepNetts
```

After running the project, the application:

1. loads and splits the dataset;
2. normalizes the features;
3. trains three neural network configurations;
4. compares validation metrics;
5. evaluates the selected model on the test set;
6. predicts the spam probability for one example;
7. displays and saves the loss curve.

---

## Author

**Nemanja Orelj**
