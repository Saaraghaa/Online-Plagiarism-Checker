!pip install numpy pandas scikit-learn nltk requests beautifulsoup4
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import requests
from bs4 import BeautifulSoup
import nltk

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)
    
    # Convert to lowercase
    tokens = [token.lower() for token in tokens]
    
    # Remove stopwords
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

def scrape_web_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all('p')
    page_text = ' '.join([para.get_text() for para in paragraphs])
    return preprocess_text(page_text)


def cosine_sim(text1, text2):
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors)[0, 1]

def euclidean_dist(text1, text2):
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return euclidean_distances(vectors)[0, 1]

def jaccard_sim(text1, text2):
    set1 = set(text1.split())
    set2 = set(text2.split())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

def levenshtein_dist(text1, text2):
    if len(text1) < len(text2):
        return levenshtein_dist(text2, text1)

    if len(text2) == 0:
        return len(text1)

    previous_row = range(len(text2) + 1)
    for i, c1 in enumerate(text1):
        current_row = [i + 1]
        for j, c2 in enumerate(text2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def check_plagiarism(input_text, urls):
    preprocessed_input = preprocess_text(input_text)
    
    results = []
    for url in urls:
        web_text = scrape_web_page(url)
        cosine = cosine_sim(preprocessed_input, web_text)
        euclidean = euclidean_dist(preprocessed_input, web_text)
        jaccard = jaccard_sim(preprocessed_input, web_text)
        levenshtein = levenshtein_dist(preprocessed_input, web_text)
        
        results.append({
            'url': url,
            'cosine_similarity': cosine,
            'euclidean_distance': euclidean,
            'jaccard_similarity': jaccard,
            'levenshtein_distance': levenshtein
        })
    
    return pd.DataFrame(results)

input_text = "Oldest Language"
urls = [
    'https://www.scientificamerican.com/article/whats-the-worlds-oldest-language1/#:~:text=Historians%20and%20linguists%20generally%20agree,language%20to%20the%20next%20generation.',
    'https://timesofindia.indiatimes.com/readersblog/whatsup-university/oldest-language-of-the-world-19460/',
    'https://en.wikipedia.org/wiki/List_of_languages_by_first_written_account'
]

results = check_plagiarism(input_text, urls)
print(results)

