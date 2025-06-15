# Text Datasets

This folder contains text datasets for natural language processing and semi-supervised learning.

## Available Datasets

### 1. Sample Newsgroups (`newsgroups/`)
- **File**: `sample_newsgroups.csv`
- **Samples**: 1,000 (sample from 20 Newsgroups)
- **Categories**: 3 (automotive, technology, politics)
- **Use Case**: Text classification, semi-supervised learning
- **Features**: Raw text, category labels

### 2. Semi-Supervised Splits (`semi_supervised/`)
- **Files**: `labeled_data.csv`, `unlabeled_data.csv`
- **Purpose**: Semi-supervised learning experiments
- **Format**: Split of newsgroups data (10% labeled, 90% unlabeled)

## Usage Examples

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load text data
newsgroups = pd.read_csv('data/raw/text/newsgroups/sample_newsgroups.csv')

# Convert text to features
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(newsgroups['text'])
y = newsgroups['target']

# Semi-supervised learning
labeled = pd.read_csv('data/raw/text/semi_supervised/labeled_data.csv')
unlabeled = pd.read_csv('data/raw/text/semi_supervised/unlabeled_data.csv')
```

## Text Processing Pipeline

1. **Text Cleaning**: Remove special characters, normalize case
2. **Tokenization**: Split text into words/tokens
3. **Feature Extraction**: TF-IDF, word embeddings, etc.
4. **Classification**: Naive Bayes, SVM, Neural Networks

## Semi-Supervised Learning Applications

- **Label Propagation**: Use labeled data to infer labels for unlabeled data
- **Self-Training**: Train classifier on labeled data, predict unlabeled, retrain
- **Co-Training**: Use multiple views of data for training
