import os
import re
from nltk.stem import PorterStemmer
import math
from collections import defaultdict

class TextTokenizer:
    def __init__(self):
        self.stemmer = PorterStemmer()

    def tokenize_text(self, document):
        # Remove standalone or surrounded digits with spaces
        document = re.sub(r"(^|\s|.|,|:)\d+($|\s|.|,|:|%)", r'\1\2', document)
        # Tokenize the document and convert to lowercase
        tokens = re.findall(r'\w+', document.lower())
        return tokens

class DocumentReader:
    def __init__(self, stop_words, tokenizer):
        self.stop_words = stop_words
        self.tokenizer = tokenizer

    def read_document(self, filename):
        documents_data = {}
        with open(filename, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
            # Retrieve individual documents from the content
            documents = re.findall(r'<DOC>.*?</DOC>', content, re.DOTALL)
            for doc in documents:
                # Extract document ID
                doc_name = re.findall(r'<DOCNO>(.*?)</DOCNO>', doc, re.DOTALL)
                doc_name = re.sub(r'\s', '', doc_name[0])

                # Extract text content
                text = re.search(r'<TEXT>(.*?)</TEXT>', doc, re.DOTALL).group(1).strip()

                # Tokenize and process words in the text
                words = self.tokenizer.tokenize_text(text)
                words = [self.tokenizer.stemmer.stem(word) for word in words if word not in self.stop_words]
                documents_data[doc_name] = words

        return documents_data

class QueryExtractor:
    def extract_queries(self, text):
        query_number = None
        query_title = None
        extracted_queries = {}

        # Extract query numbers and titles from TREC topics
        for line in text.split('\n'):
            if '<num>' in line:
                query_number = line.split(':')[-1].strip()
            elif '<title>' in line:
                query_title = line.split(':')[-1].strip()
            elif '</top>' in line:
                if query_number and query_title:
                    extracted_queries[query_number] = query_title
                query_number = None
                query_title = None

        return extracted_queries

class IndexHandler:
    def __init__(self, stop_words, tokenizer):
        self.stop_words = stop_words
        self.tokenizer = tokenizer
        self.stemmer = PorterStemmer()
        self.word_to_id = {}  # Initialize word_to_id mapping here

    def create_forward_index(self, documents):
        forward_index = {}
        current_id = 0

        # Generate forward index for each document
        for doc, text in documents.items():
            forward_index[doc] = []
            frequency_word = {}

            # Process each word in the document
            for word in text:
                if word not in self.word_to_id:
                    self.word_to_id[word] = current_id
                    current_id += 1
                word_id = self.word_to_id[word]
                if word_id not in frequency_word:
                    frequency_word[word_id] = 0
                frequency_word[word_id] += 1

            # Create forward index entries with word frequencies
            for word_id, freq in frequency_word.items():
                forward_index[doc].append(f"wordId{word_id}: {freq}")

        return forward_index

    def create_inverted_index(self, documents):
        inverted_index = {}
        # Generate inverted index for all words in documents
        for doc, words in documents.items():
            for word in words:
                word_id = "wordID" + str(self.word_to_id[word])
                if word_id not in inverted_index:
                    inverted_index[word_id] = {}

                if doc not in inverted_index[word_id]:
                    inverted_index[word_id][doc] = 0
                inverted_index[word_id][doc] += 1

        return inverted_index

    def perform_search(self, query, inverted_index, documents, query_num):
        query = [self.tokenizer.stemmer.stem(word) for word in self.tokenizer.tokenize_text(query) if word not in self.stop_words]

        scores = defaultdict(float)
        idf_values = {}
        for term in query:
            try:
                if term in self.word_to_id:
                    term_id = "wordID" + str(self.word_to_id[term])
                    n = len(inverted_index[term_id])
                    idf_values[term] = math.log(len(documents) / n)
            except:
                continue

        for term, idf in idf_values.items():
            term_id = "wordID" + str(self.word_to_id[term])
            postings = inverted_index.get(term_id, {})
            for doc, freq in postings.items():
                if doc in documents:
                    max_tf = max(postings.values())
                    tf = (1 + math.log10(freq)) / (1 + math.log10(max_tf))
                    scores[doc] += tf * idf
        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        output_string = ""
        for i, (doc_id, score) in enumerate(ranked_docs):
            doc = documents[doc_id]
            output_string += f"\n{query_num}    {doc_id}    {i+1}    {score:.6f}"

        return output_string.strip()


# Reading stop words from a file
file_stop_words = open('stopwordlist.txt', 'r')
stop_words = [word.lstrip(" ") for word in file_stop_words.read().split('\n')]
file_stop_words.close()

# Creating instances of the classes
tokenizer = TextTokenizer()
document_reader = DocumentReader(stop_words, tokenizer)
query_extractor = QueryExtractor()
index_handler = IndexHandler(stop_words, tokenizer)

# Reading TREC topics from a file
file_topics = open('topics.txt', 'r')
topics_content = file_topics.read()
extracted_queries = query_extractor.extract_queries(topics_content)
file_topics.close()

# Creating the output file
output_file = open('index_output.txt', 'w')

# Processing TREC files in the specified directory
directory = './ft911'
processed_data = {}

for filename in os.listdir(directory):
    if filename.startswith('ft911'):
        documents = document_reader.read_document(os.path.join(directory, filename))
        forward_index = index_handler.create_forward_index(documents)
        inverted_index = index_handler.create_inverted_index(documents)

        # Extracting the query and performing the search
        for query_num, query in extracted_queries.items():
            result = index_handler.perform_search(query, inverted_index, documents, query_num)
            output_file.write(result)
            output_file.write("\n")

# Closing the output file
output_file.close()
