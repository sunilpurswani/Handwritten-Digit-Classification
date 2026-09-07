# Handwritten Digit Classification

I built this project to compare a traditional machine-learning model with a neural network for handwritten digit classification. The analysis uses 42,000 handwriting samples and evaluates how well each model recognizes digits from pixel-level features.

## Objective

The project explores whether handwriting classification can support early identification of students who may need additional fine-motor-skill development support.

I compared:

- K-Nearest Neighbors (KNN)
- Feed-forward Neural Network

## Dataset

The dataset contains:

- 42,000 handwriting samples
- 10 digit classes (0-9)
- 45 pixel-intensity features
- No missing values
- A reasonably balanced class distribution

The dataset file is not included in the repository. To run the notebook, place `letters.csv` inside the `data/` folder.

![Class distribution](assets/class_distribution.png)

## Approach

### Data Preparation

- Used an 80/20 stratified train-test split
- Applied MinMax scaling to pixel features
- Preserved class distribution across training and test data

### K-Nearest Neighbors

I tuned KNN with randomized cross-validation across:

- Number of neighbors
- Uniform vs. distance weighting
- Euclidean, Manhattan, and Minkowski distance metrics

The best configuration used 14 neighbors with distance weighting and Minkowski distance.

### Neural Network

I built a feed-forward neural network with:

- Dense layers with 256, 256, and 128 neurons
- ReLU activation
- Batch normalization
- Dropout regularization
- Adam optimizer
- Early stopping
- Learning-rate reduction

## Results

| Metric | KNN | Neural Network |
|---|---:|---:|
| Accuracy | 65.86% | 70.95% |
| Weighted Precision | 66.04% | 71.76% |
| Weighted Recall | 65.86% | 70.95% |
| Weighted F1-Score | 65.39% | 70.90% |

The neural network performed better across all four evaluation metrics.

### KNN Confusion Matrix

![KNN confusion matrix](assets/knn_confusion_matrix.png)

KNN performed well on clearer digits such as 0, 1, and 6, but had more difficulty separating visually similar digits such as 3, 8, and 9.

### Neural Network Confusion Matrix

![Neural network confusion matrix](assets/nn_confusion_matrix.png)

The neural network produced stronger class separation and more consistent performance across the digit classes.

## Training Behavior

![Neural network accuracy](assets/nn_accuracy_curve.png)

![Neural network loss](assets/nn_loss_curve.png)

Training and validation performance stabilized without a large gap, indicating that the regularization strategy helped control overfitting.

## Conclusion

The neural network was the stronger model for this dataset, improving weighted F1-score from 65.39% to 70.90%. Its ability to learn non-linear relationships between pixel features gave it an advantage over the distance-based KNN model.

For future work, I would test convolutional neural networks if the available pixel features can be reconstructed into meaningful two-dimensional image representations.

## Repository Structure

```text
Handwritten-Digit-Classification/
├── assets/
│   ├── class_distribution.png
│   ├── knn_confusion_matrix.png
│   ├── nn_accuracy_curve.png
│   ├── nn_confusion_matrix.png
│   └── nn_loss_curve.png
├── data/
│   └── README.md
├── notebooks/
│   └── handwritten_digit_classification.ipynb
├── report/
│   └── Handwritten_Text_Classification.pdf
├── .gitignore
├── README.md
└── requirements.txt
```

## Tools

Python, pandas, NumPy, scikit-learn, TensorFlow/Keras, Matplotlib, Jupyter Notebook

## Run the Project

1. Clone the repository.
2. Place `letters.csv` in the `data/` folder.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Open the notebook:

```bash
jupyter notebook notebooks/handwritten_digit_classification.ipynb
```

## Author

Author: Sunil Purswani
