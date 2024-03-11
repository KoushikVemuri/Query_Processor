# Query Processor

This Python program processes TREC documents and topics to perform information retrieval tasks using inverted indexing.

## Description

This code implements a simple query processing system using a collection of TREC documents and topics. It includes several components:

- **TextTokenizer**: Tokenizes and preprocesses text data by removing stop words, stemming, and generating tokens.
- **DocumentReader**: Reads TREC documents, extracts content, and tokenizes it for indexing.
- **QueryExtractor**: Extracts queries from TREC topics for information retrieval.
- **IndexHandler**: Generates forward and inverted indices, calculates TF-IDF scores, and ranks documents based on query relevance.

## Usage

1. Ensure Python and required libraries (NLTK) are installed.
2. Prepare input files:
   - `stopwordlist.txt`: Contains a list of stop words.
   - `topics.txt`: Holds TREC topics for queries.
   - TREC documents in the specified directory structure (e.g., `./ft911`).
3. Modify filenames and directories in the code if needed (`stopwordlist.txt`, `topics.txt`, `./ft911`).
4. Execute the script.

## File Structure

- `query_processor.py`: Main Python script containing classes for processing TREC documents and topics.
- `stopwordlist.txt`: File with a list of stop words.
- `topics.txt`: File containing TREC topics/queries.
- `index_output.txt`: Output file storing search results.

## Dependencies

- NLTK library: Install using `pip install nltk`.
