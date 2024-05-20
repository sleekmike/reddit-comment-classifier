import pandas as pd
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
import csv
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
from itertools import groupby

stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()
#stop_words = set(stopwords.words('english'))
#stop_words.update(["wa", "gt", "amp", "u", "ha", "le", "doe", "don", 've', 'make', 'get'])
# Initialize NLP libraries and download necessary resources
#nltk.download('stopwords')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')
#============================================================>    Step 1: Data Prerpocessing and cleaning <============================================================# 
# Function to load data
def load_data(filepath):
    return pd.read_csv(filepath)

# Cleaning data
def clean_data(comment):
    comment = re.sub(r'<[^>]+>', ' ', comment)  # Remove HTML tags
    comment = re.sub(r"[^\x00-\x7F]+", " ", comment)
    comment = comment.lower()
    comment = re.sub(r"https?://\S+|www\.\S+", "", comment)
    comment = re.sub(r"[^a-z0-9\s]", " ", comment)
    comment = re.sub(r'\s+', ' ', comment).strip()
    return comment

def get_first_approx_words(text, max_words=200):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    word_count = 0
    selected_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        # If the current sentence exceeds max_words by itself and nothing has been added yet
        if word_count == 0 and len(words) > max_words:
            # Return the first max_words from this sentence
            return ' '.join(words[:max_words])
        # Check if adding this sentence would exceed the max_words limit
        if word_count + len(words) > max_words:
            break
        selected_sentences.append(sentence)
        word_count += len(words)
    # Join selected sentences to form the truncated text
    return ' '.join(selected_sentences)

def check_and_truncate(text, max_words=200):
    # Split text into words to count them
    words = text.split()
    if len(words) > max_words:
        truncated_text = get_first_approx_words(text, max_words)
        if not truncated_text.strip():  # Check if the result is empty
            #return "Text was truncated to empty, reverting to first 200 words."
            #return truncated_text
            return text
        else: 
            return truncated_text
    else:
        return text

def detect_redundancy(text):
    # Detect repeating patterns or duplicates within the text
    words = text.split()
    unique_words = set(words)
    redundancy_ratio = len(words) / (len(unique_words) + 1)  # Avoid division by zero
    if redundancy_ratio >= 2:  # Threshold for considering text as having redundancies
         return True
    else:
        print("redundancy_ratio", redundancy_ratio)
        #redundancy_ratio, "CULPRIT: ", text)
        return False

def duplicates_repetitions_cleaner(data):
    if isinstance(data, pd.DataFrame):
        # Log the number of initially empty comments
        initial_empty_count = data['comments'].isna().sum()
        print(f"Initial empty comments: {initial_empty_count}")
        # Processing
        data['comments'] = data['comments'].apply(lambda x: check_and_truncate(x) if (x and detect_redundancy(x)) else x)
        # Log the number of comments that are empty after processing
        final_empty_count = data['comments'].isna().sum()
        print(f"Empty comments after processing: {final_empty_count}")
        return data
    elif isinstance(data, str):
        if not data.strip():  # Check if the data is empty or only whitespace
            return "Received empty input text."
        return check_and_truncate(data) if detect_redundancy(data) else data
    else:
        raise ValueError("Input must be a pandas DataFrame or a string")



#def get_wordnet_pos(treebank_tag):
#    """Converts treebank tags to wordnet tags."""
#    if treebank_tag.startswith('J'):
#        return wordnet.ADJ
#    elif treebank_tag.startswith('V'):
#        return wordnet.VERB
#    elif treebank_tag.startswith('N'):
#        return wordnet.NOUN
#    elif treebank_tag.startswith('R'):
#        return wordnet.ADV
#    else:
#        return None

#def lemmatize_text(text):
#    lemmatizer = WordNetLemmatizer()
#    tokens = word_tokenize(text)
#    tagged_tokens = pos_tag(tokens)
#    lemmatized = []
#    for word, tag in tagged_tokens:
#        wn_tag = get_wordnet_pos(tag)
#        if wn_tag:
#            lemma = lemmatizer.lemmatize(word, pos=wn_tag)
#        else:
#            lemma = lemmatizer.lemmatize(word)
#        lemmatized.append(lemma)
#    return ' '.join(lemmatized)

## Stemming text
#def stem_text(comment):
#    return " ".join([stemmer.stem(word) for word in comment.split()])

# Main function to process data
def process_data(data, preprocess_steps):
    for step in preprocess_steps:
        #data = data.progress_map(step)
        data = data.map(step)
    return data

# Function to setup and process dataset
def cleaner(df):
    #comments_df = load_data('reddit_usernames_comments.csv')
    comments_df = df
    # Define preprocessing steps
    preprocessing_steps =  [clean_data, duplicates_repetitions_cleaner]
    # Apply preprocessing
    print("Cleaning and preprocessing the data...")
    comments_df['comments'] = process_data(comments_df['comments'], preprocessing_steps)
    return comments_df 


#scikit-learn imblearn
