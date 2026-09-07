# Handwritten Digit Classification

This project compares a traditional machine-learning approach with a neural network for handwritten digit classification using 42,000 handwriting samples and pixel-level features.

## Objective

The goal is to evaluate how effectively different supervised-learning methods can classify handwritten digits and to compare their performance across standard classification metrics.

The two models used are:

- K-Nearest Neighbors (KNN)
- Feed-forward Neural Network

## Dataset

The dataset contains:

- 42,000 handwriting samples
- 10 digit classes (0-9)
- 45 pixel-intensity features
- No missing values
- A balanced distribution across digit classes

The dataset is included in the repository at [`data/letters.csv`](./data/letters.csv).

## Approach

### Data Preparation

- Used an 80/20 stratified train-test split
- Applied MinMax scaling to pixel features
- Preserved the class distribution across training and test data

### K-Nearest Neighbors

KNN was tuned using randomized cross-validation across:

- Number of neighbors
- Uniform vs. distance weighting
- Euclidean, Manhattan, and Minkowski distance metrics

The best configuration used 14 neighbors with distance weighting and Minkowski distance.

### Neural Network

The feed-forward neural network used:

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

The neural network outperformed KNN across all four evaluation metrics. It improved weighted F1-score from **65.39% to 70.90%** and showed stronger overall class separation.

KNN performed better on clearer digits such as 0, 1, and 6, while visually similar digits such as 3, 8, and 9 were more difficult to distinguish. The neural network handled these nonlinear pixel relationships more effectively.

## Conclusion

The neural network was the stronger model for this dataset because it was better able to learn complex relationships between pixel features. KNN provided a useful baseline, but its distance-based approach was less effective for visually similar handwritten digits.

A possible next step would be to test a convolutional neural network if the pixel features can be reconstructed into meaningful two-dimensional image representations.

## Project Files

```text
Handwritten-Digit-Classification/
├── Handwritten_Text_Classification.ipynb
├── data/
│   └── letters.csv
├── .gitignore
├── README.md
└── requirements.txt
```

## Tools

Python, pandas, NumPy, scikit-learn, TensorFlow/Keras, Matplotlib, Jupyter Notebook

## Run the Project

1. Clone the repository.
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Open the notebook:

```bash
jupyter notebook Handwritten_Text_Classification.ipynb
```

The full analysis and model code are available in [`Handwritten_Text_Classification.ipynb`](./Handwritten_Text_Classification.ipynb).

## Author

Author: Sunil Purswani
