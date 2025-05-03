# Mutual Fund Strategy Classification

This project analyzes how well AI and traditional machine learning models can classify mutual fund summaries into investment strategy categories.

## Objective

To categorize mutual fund summaries into three strategies:

* Balanced Fund (Low Risk)
* Fixed Income Long Only (Low Risk)
* Equity Long Only (Low Risk)

## Files

* `MutualFundLabels.csv`: Labeled fund strategies
* `/MutualFundSummary`: Text summaries from fund prospectuses
* `df_RAG_results.csv`: Predictions from the RAG model


## Methods

### Retrieval-Augmented Generation (RAG)

* Used OpenAI models with LangChain to classify each summary
* Results used as training targets for machine learning models

### Word Embedding

* Trained a skip-gram Word2Vec model on the training set
* Vectorized each summary for use in ML models

### Machine Learning Models

* **Cosine Similarity**: Used as a simple baseline
* **Support Vector Classifier (SVC)**: Performed well with RBF kernel
* **Naive Bayes**: Moderate performance, best on Fixed Income
* **Logistic Regression**: Performed poorly, unable to generalize

## Results

| Model               | Best F1 Score |
| ------------------- | ------------- |
| RAG (OpenAI)        | 0.86          |
| SVC (RBF Kernel)    | 0.80          |
| Naive Bayes         | 0.67          |
| Logistic Regression | 0.58          |
