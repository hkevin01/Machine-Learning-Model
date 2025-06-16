# Semi-Supervised Learning Algorithms

This directory contains documentation for semi-supervised learning algorithms - those that utilize both labeled and unlabeled data for training.

## Available Algorithms

- [**Label Propagation**](label_propagation.md) - Graph-based algorithm that propagates labels from labeled to unlabeled points
  - Utilizes similarity between data points
  - Works well when limited labeled data is available
  - Adaptable to manifold structure of data

- [**Semi-Supervised SVM (S3VM)**](semi_supervised_svm.md) - Extension of Support Vector Machines for semi-supervised learning
  - Finds decision boundary that passes through low-density regions
  - Incorporates unlabeled data into the optimization
  - Effective when the "cluster assumption" holds

## Common Interface

All semi-supervised learning algorithms implement this common interface:

```python
# Provide labeled and unlabeled data (unlabeled marked with -1)
y_train = [0, 1, -1, -1, -1, -1, 0, 1, -1, -1]

# Train the model
model.fit(X_train, y_train)

# Get transduction results (labels for unlabeled training data)
transduced_labels = model.transduction_

# Make predictions on new data
predictions = model.predict(X_test)
```

## When to Use

- **Label Propagation**: When data has clear manifold structure and you have limited labeled examples
- **Semi-Supervised SVM**: When classes are expected to be separated by low-density regions

## Practical Considerations

- Semi-supervised learning is most effective when:
  - Labeled data is expensive or limited
  - Unlabeled data is abundant
  - The distribution of unlabeled data provides useful information about class boundaries

- Be cautious when:
  - Class distributions are heavily imbalanced
  - Unlabeled data may come from different distributions than labeled data

## References

See individual algorithm pages for specific references and academic papers.
