# 🧠 Custom NLP Dataset & Visualization Toolkit

A modular and reusable NLP utility package for:

-   📊 Dataset inspection & cleaning\
-   🧹 Text preprocessing & feature engineering\
-   📈 Visualization & exploratory text analysis\
-   🤖 Single-text transformation for inference

This toolkit is designed to streamline NLP workflows for classification
tasks such as sentiment analysis, spam detection, review classification,
etc.

------------------------------------------------------------------------

## 🚀 Features

### 1️⃣ CustomNLPDatasetOp

Handles dataset-level operations:

-   Displays dataset information
-   Removes duplicate rows
-   Converts text to lowercase
-   Applies custom preprocessing
-   Returns fully transformed feature DataFrame

------------------------------------------------------------------------

### 2️⃣ CustomNLPPreprocessor

Performs advanced text preprocessing:

✔ URL & HTML removal\
✔ Special character cleaning\
✔ Smart stopword handling (keeps important words like *not, but,
however, no, yet*)\
✔ Lemmatization

Extracted Features:

-   clean_text
-   no_of_stopwords
-   word_count
-   punctuation_chars
-   char_count

Also supports **single text inference transformation**.

------------------------------------------------------------------------

### 3️⃣ CustomVisualizationHelper

Provides powerful visualizations:

-   Target label distribution
-   KDE plots (word count & stopword count)
-   Boxplots
-   Top 25 bigrams
-   WordCloud
-   Top N most frequent words

------------------------------------------------------------------------

# 📦 Installation

``` bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn wordcloud
```

Download required NLTK resources:

``` python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

------------------------------------------------------------------------

# 🛠️ Usage Example

``` python
import pandas as pd
from custom_nlp_module import (
    CustomNLPDatasetOp,
    CustomNLPPreprocessor,
    CustomVisualizationHelper
)

df = pd.read_csv("your_dataset.csv")

preprocessor = CustomNLPPreprocessor()

dataset_op = CustomNLPDatasetOp(
    df=df,
    text_col="review",
    target_col="sentiment",
    preprocessor=preprocessor
)

processed_df = dataset_op.run_dataset_operations(verbose=True)

visualizer = CustomVisualizationHelper(
    df=processed_df,
    target="sentiment",
    word_count="word_count",
    no_of_stopwords="no_of_stopwords",
    use_clean_text=True
)

visualizer.visualize()
```

------------------------------------------------------------------------

# 📊 Generated Features

  Feature Name        Description
  ------------------- ----------------------------------------------
  clean_text          Fully cleaned & lemmatized text
  no_of_stopwords     Count of stopwords (excluding key negations)
  word_count          Total number of words
  punctuation_chars   Number of punctuation characters
  char_count          Total characters after cleaning

------------------------------------------------------------------------

# 🎯 Ideal Use Cases

-   Sentiment Analysis
-   Spam Detection
-   Review Classification
-   Fake News Detection
-   Any Text Classification Problem

------------------------------------------------------------------------

# ⚠️ Notes

-   Make sure your dataset contains the specified text column.
-   Target column is optional but required for label-based
    visualizations.
-   Ensure NLTK resources are downloaded before running.

------------------------------------------------------------------------

# 👨‍💻 Author

Reusable NLP utility framework for efficient experimentation and rapid
prototyping.
