# 🤖 Customer Purchase Prediction using PCA and MLP

## 📖 Description
This project focuses on predicting customer purchase behavior using a combination of **Principal Component Analysis (PCA)** for dimensionality reduction and a **Multi-Layer Perceptron (MLP)** for classification. The dataset contains demographic and purchasing behavior data, allowing insights into customer trends and segmentation.

---

## 🚀 Objectives
- Apply PCA to reduce dimensionality while preserving 90% of the dataset's variance.
- Train an MLP to classify customers based on their likelihood of making a purchase.
- Analyze the impact of class imbalance and suggest strategies for improvement.

---

## 📊 Key Sections
1. **Data Preprocessing**:
   - Handling missing values and cleaning data.
   - Normalizing and encoding variables using pipelines for seamless preprocessing.

2. **Unsupervised Analysis with PCA**:
   - Reduced features to 18 principal components.
   - Visualized cumulative explained variance to optimize component selection.
   - Identified key variables influencing customer purchase behavior (e.g., `Income`, `MntWines`, `MntMeatProducts`).

3. **Modeling with MLP**:
   - Implemented a neural network with the following architecture:
     - **Input Layer**: 18 features from PCA.
     - **Hidden Layers**:
       - Layer 1: 128 neurons, ReLU activation.
       - Layer 2: 64 neurons, ReLU activation.
     - **Output Layer**: Sigmoid activation for binary classification.
   - Regularization with dropout layers to reduce overfitting.
   - Early stopping to optimize training.

4. **Model Evaluation**:
   - Achieved an accuracy of **86.20%** on the test set.
   - Precision, recall, and F1-score highlighted challenges due to class imbalance.
   - Confusion matrix and learning curves were used for further analysis.

---

## 🚀 Technologies Used
- **Python**: Core programming language.
- **Jupyter Notebook**: For interactive data exploration and analysis.
- **Libraries**:
  - Pandas & NumPy: Data manipulation and preprocessing.
  - Scikit-learn: PCA and model evaluation metrics.
  - TensorFlow & Keras: Implementation of the MLP model.
  - Matplotlib & Seaborn: Data visualization.

---

## 📈 Results
- PCA effectively reduced the dataset dimensions while preserving critical information.
- The MLP demonstrated strong generalization, with balanced performance across training and validation datasets.
- Class imbalance led to lower recall for the "purchase" class, indicating areas for improvement.

---

## 💡 Insights
1. Variables such as `Income`, `MntWines`, and `MntMeatProducts` significantly influence customer behavior.
2. PCA not only simplified the dataset but also improved the model's efficiency.
3. Future improvements include addressing class imbalance through techniques like oversampling or undersampling.
