import re
import nltk
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collection.Collection import Counter
from sklearn.feature_extraction.text import CountVectorizer

class CustomNNLPDatasetOp:
    def __init__(self,df,text_col):
        nltk.download('all',quiet=True)
        self.df = df
        self.text_col = text_col

    def dataset_info(self):
        print("Dataset Information:")
        print(self.df.info())
        print("\nMissing Values per Column:")
        print(self.df.isnull().sum())
        print("\nDataset Description:")
        print(self.df.describe(include='all'))

    def remove_duplicates_to_lowercase(self, verbose=False):
        before = len(self.df)
        df = self.df.drop_duplicates().copy()
        after = len(self.df)

        if verbose:
            print(f"Removed {before - after} duplicate rows")

        df[self.text_col] = df[self.text_col].astype(str).str.lower()
        return df
    
    def run_dataset_operations(self, verbose=False):
        self.dataset_info()
        cleaned_df = self.remove_duplicates_to_lowercase(verbose=verbose)
        return cleaned_df

class CustomNLPPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.custom_stop_words = self.stop_words - {'not', 'but', 'however', 'no', 'yet'}
        self.lemmantizer = WordNetLemmatizer()

    def preprocess(self, text):
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def count_stopwords(self,text):
        return sum(1 for word in text.split() if word in self.custom_stop_words)

    def word_counts(self,text):
        return len(text.split())

    def count_punctuation_chars(self, text):
        punctuation = '.,!?;:"\'()[]{}-'
        return sum(1 for char in text if char in punctuation)

    def remove_special_characters(self, text):
        """
        Keeps only letters, numbers, spaces, and basic punctuation (! ? . ,)
        """
        text = str(text)
        return re.sub(r'[^A-Za-z0-9\s!?.,]', '', text)

    def remove_stopwords(self, text):
        text = ' '.join([word for word in text.split() if word not in self.custom_stop_words])
        return text

    def lemmantize_text(self, text):
        text = ' '.join([self.lemmantizer.lemmatize(word) for word in text.split()])
        return text
    
    # Main transformation function
    def transform_text(self, text):
        """
        Call this Main transformation function to get all features
        1. clean_text
        2. no_of_stopwords
        3. word_count
        4. punctuation_chars
        5. char_count
        Returns a dictionary with all features
        """
        text = str(text).lower()
        text = self.preprocess(text)
        text = self.remove_special_characters(text)

        stopword_count = self.count_stopwords(text)
        word_count = self.word_counts(text)
        punctuation_count = self.count_punctuation_chars(text)

        text = self.remove_stopwords(text)
        text = self.lemmantize_text(text)

        return {
            "clean_text": text,
            "no_of_stopwords": stopword_count,
            "word_count": word_count,
            "punctuation_chars": punctuation_count,
            "char_count": len(text)
        }

class CustomVisualizationHelper:
    def __init__(self,df,text,target,word_count,no_of_stopwords,clean_text):
        self.df = df
        self.text = text
        self.target = target
        self.word_count = word_count
        self.no_of_stopwords = no_of_stopwords
        self.clean_text = clean_text

    def basic_plots(self):

        sns.countplot(x=self.target, data=self.df)
        plt.show()

        sns.kdeplot(self.df[self.df[self.target] == 1][self.word_count], label='Positive', fill=True)
        sns.kdeplot(self.df[self.df[self.target] == 0][self.word_count], label='Negative', fill=True)
        plt.legend()
        plt.show()

        sns.boxplot(data=self.df,x=self.target,y=self.word_count)
        plt.show()

        sns.kdeplot(self.df[self.df[self.target] == 1][self.no_of_stopwords], label='Positive', fill=True)
        sns.kdeplot(self.df[self.df[self.target] == 0][self.no_of_stopwords], label='Negative', fill=True)
        plt.legend()
        plt.show()

    def get_top_ngrams(corpus, n=None):
        vec = CountVectorizer(ngram_range=(2, 2), stop_words='english').fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return words_freq[:n]

    top_25_bigrams = self.get_top_ngrams(self.df[self.clean_text], 25)
    top_25_bigrams_df = pd.DataFrame(top_25_bigrams, columns=['bigram', 'count'])

    plt.figure(figsize=(12, 8))
    sns.barplot(data=top_25_bigrams_df, x='count', y='bigram', palette='magma')
    plt.title('Top 25 Most Common Bigrams')
    plt.xlabel('Count')
    plt.ylabel('Bigram')
    plt.show()

    # --- IGNORE ---
    def generate_wordcloud():
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(text))
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
    # --- IGNORE ---

    def plot_top_n_words(n=20):
        """Plot the top N most frequent words in the dataset."""
        # Flatten all words in the content column
        words = ' '.join(self.df[self.clean_text]).split()

        counter = Counter(words)
        most_common_words = counter.most_common(n)

        # Split the words and their counts for plotting
        words, counts = zip(*most_common_words)

        # Plot the top N words
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(counts), y=list(words))
        plt.title(f'Top {n} Most Frequent Words')
        plt.xlabel('Frequency')
        plt.ylabel('Words')
        plt.show()
