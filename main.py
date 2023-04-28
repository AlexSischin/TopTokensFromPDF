import os
import numpy as np
import pdfplumber
import spacy
from sklearn.feature_extraction.text import CountVectorizer

nlp = spacy.load('en_core_web_sm')


def find_files_recursively(directory):
    # Traverse directory and return a list of all files
    return [os.path.join(root, f) for (root, _, fs) in os.walk(directory) for f in fs]


def extract_texts(pdf_file_names):
    file_texts = []
    # Extract text from each PDF file
    for file_name in pdf_file_names:
        with pdfplumber.open(file_name) as pdf:
            # Extract text from each page of the PDF file
            page_texts = [page.extract_text(layout=True) for page in pdf.pages]
        file_texts.extend(page_texts)
    return file_texts


def filter_tokens(token):
    # Filter irrelevant tokens
    return (not token.is_stop
            and token.is_alpha
            and token.ent_type_ == ''
            and len(token.lemma_) >= 3)


def spacy_tokenizer(text):
    doc = nlp(text)
    # Process the text and return filtered tokens
    tokens = [token.lemma_.lower() for token in doc if filter_tokens(token)]
    return tokens


def main():
    directory = 'pdf_files'
    # Find all PDF files in the directory
    file_names = find_files_recursively(directory)
    pdf_file_names = [f for f in file_names if f.endswith('.pdf')]
    # Extract text from PDF files
    texts = extract_texts(pdf_file_names)

    # Initialize the CountVectorizer with a custom tokenizer
    vectorizer = CountVectorizer(tokenizer=spacy_tokenizer, lowercase=True)

    # Fit and transform the data
    X = vectorizer.fit_transform(texts)

    # Retrieve token frequencies
    token_frequencies = X.sum(axis=0).A1

    # Get the most frequent tokens
    n = 100
    top_token_indices = np.argsort(-token_frequencies)[:n]
    top_tokens = vectorizer.get_feature_names_out()[top_token_indices]
    top_token_frequencies = token_frequencies[top_token_indices]

    # Pair tokens with their frequencies
    most_frequent_tokens = list(zip(top_tokens, top_token_frequencies))

    # Print the results
    for token, freq in most_frequent_tokens:
        print(f"{token}: {freq}")


if __name__ == '__main__':
    main()
