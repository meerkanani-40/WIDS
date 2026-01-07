# Stock Prediction model using LSTM

In week 1, we learned python basics and few libraries. Also we did few exercises to practice the concepts learned which are uploaded in this repository.

## Python Basics

### Key Concepts Covered
- **Variables & Data Types:** Integers, floats, strings, booleans, and `None`
- **Operators:** Arithmetic, comparison, and logical operators
- **Control Flow:** `if-else` conditions and loops (`for`, `while`)
- **Data Structures:** Lists, tuples, and dictionaries
- **Functions:** Modular code using user-defined functions

These concepts formed the backbone for later library-based implementations.

---

## NumPy

NumPy was used for numerical computation and array-based operations.

### Learning Outcomes
- Creation and manipulation of multi-dimensional arrays
- Element-wise operations and broadcasting
- Indexing, slicing, and basic statistical functions
- Random number generation for simulations

NumPy significantly improved computational efficiency compared to native Python lists.

---

## Pandas

Pandas enabled structured data analysis using tabular formats.

### Key Applications
- Creating `DataFrame` and `Series` objects
- Importing data from CSV files
- Handling missing values
- Filtering, sorting, and aggregating data
- Basic grouping and summarization

This library helped bridge raw data and meaningful insights.

---

## Matplotlib

Matplotlib was introduced for data visualization and result interpretation.

### Visualizations Explored
- Line plots for trends
- Scatter plots for relationships
- Bar charts for categorical comparison
- Histograms for distribution analysis

Plot customization using labels, titles, and legends improved clarity and presentation.

---

In week 2, we were introduced to Machine Learning, which included supervised and unsupervised learning, also we were introduced with a new library - yfinance.

## What is Machine Learning?

Machine Learning is a subset of artificial intelligence where systems learn from data and improve performance without being explicitly programmed. The learning process generally involves:
- Providing data to a model
- Learning patterns or relationships
- Making predictions or decisions based on unseen data

---

## Supervised Learning

Supervised learning involves training models on labeled data, where the input-output relationship is known.

### Key Concepts Learned
- **Regression:** Predicting continuous values (e.g., prices, scores)
- **Classification:** Predicting discrete classes (e.g., yes/no, categories)

### Common Algorithms Introduced
- Linear Regression
- Logistic Regression

Supervised learning highlighted the importance of data quality and proper evaluation metrics.

---

## Unsupervised Learning

Unsupervised learning works with unlabeled data and focuses on discovering hidden structures.

### Key Concepts Learned
- **Clustering:** Grouping similar data points

### Algorithms Discussed
- K-Means Clustering

This approach helped in understanding how patterns can emerge without explicit labels.

---

## Model Evaluation Basics

Basic evaluation ideas were introduced to assess model performance:
- Accuracy for classification
- Error and loss concepts
- Overfitting vs underfitting
- Importance of generalization

These concepts emphasized that building a model is not enough; validating its performance is equally critical.

---

## yfinance Library

The `yfinance` library was introduced as a practical tool to fetch real-world financial data directly from online sources.

### Key Learnings
- Downloading historical stock price data using ticker symbols
- Accessing open, high, low, close, and volume data
- Fetching data over custom time ranges
- Working with downloaded data as Pandas DataFrames

### Applications
- Exploratory data analysis on stock prices
- Visualization of price trends
- Preparing datasets for ML-based financial analysis

The library enabled seamless integration of real market data with machine learning workflows.

---

In next two weeks we learned concepts such as Neural Networks, Gradient Descent, Backpropogation in NN and Convolutional Neural Networks.

## Introduction to Neural Networks

Neural Networks are inspired by the structure of the human brain and consist of interconnected layers of neurons.

### Key Components
- **Input Layer:** Receives raw features
- **Hidden Layers:** Learn intermediate representations
- **Output Layer:** Produces final predictions
- **Weights and Biases:** Learnable parameters of the model
- **Activation Functions:** Introduce non-linearity (ReLU, Sigmoid)

Neural networks enable modeling of complex, non-linear relationships in data.

---

## Gradient Descent

Gradient Descent is an optimization algorithm used to minimize the loss function of a model.

### Core Ideas
- Iteratively updates weights to reduce error
- Uses learning rate to control update magnitude
- Moves parameters in the direction of steepest loss reduction

Understanding gradient descent provided insight into how models improve during training.

---

## Backpropagation in Neural Networks

Backpropagation is the mechanism through which neural networks learn.

### Key Learnings
- Error is computed at the output layer
- Gradients are propagated backward through the network
- Chain rule is used to update each weight
- Works in combination with gradient descent

This concept explained how deep networks efficiently adjust millions of parameters.

---

## Convolutional Neural Networks (CNNs)

Convolutional Neural Networks are specialized neural networks designed for image and spatial data.

### Core Components
- **Convolution Layers:** Extract spatial features
- **Filters/Kernels:** Learn local patterns
- **Pooling Layers:** Reduce dimensionality
- **Fully Connected Layers:** Perform final classification

CNNs demonstrated how hierarchical feature learning improves performance on visual tasks.
