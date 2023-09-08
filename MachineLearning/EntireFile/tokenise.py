import re
import numpy as np
from keras.preprocessing.text import Tokenizer
from gensim.models import Word2Vec
# Padding all questions with zeros
#from keras.preprocessing.sequence import pad_sequences
#from keras.preprocessing.sequence import pad_sequences


class processData:
    def __init__(filePath) :
        code=open(filePath).read()
        return code


    def tokenize_c_code(code):
        # Define the regular expression pattern to match tokens in the C code
        pattern = re.compile(r'\b\w+\b|[^\w\s]+')
        
        # Tokenize the code by splitting it into a list of strings using the pattern
        tokens = re.findall(pattern, code)
        
        return tokens

    # Example usage

    def build_vocabulary(tokens):
        # Create an empty set to store the unique tokens
        vocabulary = set()
        
        # Iterate over the list of tokens and add each unique token to the vocabulary
        for token in tokens:
            vocabulary.add(token)
        
        # Convert the vocabulary set to a list and return it
        return list(vocabulary)


    def map_tokens_to_integers(vocabulary):
        # Create a dictionary to store the mapping from tokens to integers
        token_to_int = {}
        
        # Iterate over the vocabulary and assign a unique integer to each token
        for i, token in enumerate(vocabulary):
            token_to_int[token] = i
        
        return token_to_int

    def words_to_int(tokens, vocab_dict):
    # use the vocab_dict to map each word to its corresponding integer
        int_representation = [vocab_dict[word] for word in tokens if word in vocab_dict]
        return int_representation

    def emmbbed_words(filePath):
        # Read the C code file into a list of strings, where each string is a line of code
        with open(filePath, 'r') as f:
            lines = f.readlines()

        # Tokenize each line of code into a list of words
        tokenized_lines = [line.split() for line in lines]

        # Train a word2vec model on the tokenized lines
        model = Word2Vec(tokenized_lines, vector_size=100, window=5, min_count=1, workers=4)
        return model




