# 🧠 CIFAR-10 Classification with ResNet50 (Kaggle Notebook)

This project demonstrates how to classify CIFAR-10 images using **transfer learning** with a **pretrained ResNet50** model, fine-tuned on selectively trainable convolutional layers. The experiment was run entirely in a **Kaggle Notebook**.

---
<img width="790" alt="image" src="https://github.com/user-attachments/assets/887f8660-76c2-44a3-a893-b1e1a4a20d73" />



## 🔧 Key Features

- **Dataset**: CIFAR-10 (10 classes, 32×32 images)
- **Base model**: ResNet50 (`include_top=False`, pretrained on ImageNet)
- **Custom head**: Fully-connected layers with L2 regularization, batch normalization, and dropout
- **Fine-tuning**: Only `Conv2D` layers of ResNet50 were unfrozen
- **Data Augmentation**: Random flip, rotation, and zoom during training
- **Validation**: 10% split from training data using `train_test_split`
- **Tracking**: Training metrics were logged using MLflow (local-only in Kaggle runtime)

---

## 📈 Final Performance

| Metric            | Result     |
|-------------------|------------|
| **Train Accuracy**    | 99.24%     |
| **Validation Accuracy** | 96.30%     |
| **Test Accuracy**      | **96.06%** |
| **F1-Score (Macro Avg)** | 0.96      |

✔️ Model generalizes well  
✔️ No significant overfitting  
✔️ Minor confusion between similar classes (e.g., cats & dogs)

---

## 🧪 How It Works

1. **Preprocess images**: resize to 224×224 to match ResNet50 input
2. **Apply augmentations** only to training set
3. **Split data**: 90% for training, 10% for validation
4. **Train in two phases**:
   - Phase 1: train classifier head only
   - Phase 2: fine-tune Conv2D layers of ResNet50
5. **Evaluate** model on separate test set

---

## ✅ Technologies Used

- Python 3 (Kaggle Notebook)
- TensorFlow / Keras
- Scikit-learn
- OpenCV (for interpolation)
- MLflow (local logging only)

---

## 📦 Notes

- The model was **not saved locally** due to Kaggle environment restrictions.
- If needed, you can add `model.save('model.keras')` to export and download the final model manually.

---

## 📚 References

- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [ResNet50 - Keras Applications](https://keras.io/api/applications/resnet/)
- [Kaggle Notebooks](https://www.kaggle.com/code/svitlanakovalivska/cifar10-on-resnet50-notebook-kovalivska)

-
