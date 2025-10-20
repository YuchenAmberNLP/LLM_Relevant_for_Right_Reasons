import json
import argparse
from collections import defaultdict
from pathlib import Path
import re

def is_valid_id_start(line):
    """Check if line starts with Q or D followed by numbers."""
    return bool(re.match(r'^[QD]\d+', line))

def read_queries(query_file):
    """Read queries from TSV file and return a dictionary of qid to query text."""
    queries = {}  # {qid: query} mapping
    current_line = ""

    with open(query_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if is_valid_id_start(line):
                # If we have a previous line, process it
                if current_line:
                    parts = current_line.split("\t", 1)
                    if len(parts) == 2:
                        qid, query = parts
                        query = query.strip()
                        queries[qid] = query
                # Start new line
                current_line = line
            else:
                # Append to current line
                current_line += "\n" + line

        # Process the last line
        if current_line:
            parts = current_line.split("\t", 1)
            if len(parts) == 2:
                qid, query = parts
                query = query.strip()
                queries[qid] = query

    return queries

def read_documents(doc_file):
    """Read documents from TSV file and return a dictionary of did to document text."""
    documents = {}
    current_line = ""

    with open(doc_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if is_valid_id_start(line):
                # If we have a previous line, process it
                if current_line:
                    parts = current_line.split("\t", 1)
                    if len(parts) == 2:
                        did, doc = parts
                        doc = doc.strip()
                        documents[doc] = did
                # Start new line
                current_line = line
            else:
                # Append to current line
                current_line += "\n" + line

        # Process the last line
        if current_line:
            parts = current_line.split("\t", 1)
            if len(parts) == 2:
                did, doc = parts
                doc = doc.strip()
                documents[doc] = did

    return documents

def generate_qrels(data_file):
    # Get language code from input file path
    lang = Path(data_file).stem

    # Construct paths for query and document files
    base_dir = Path("data/xorqa-r")
    query_file = base_dir / f"{lang}_queries.tsv"
    doc_file = base_dir / f"{lang}_sentences.tsv"

    # Read queries and documents
    queries = read_queries(query_file)
    documents = read_documents(doc_file)

    # Load the data
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Initialize qrels dictionary
    qrels = defaultdict(list)

    # Process all documents in the data
    for doc in data["data"]:
        # Process each paragraph
        for paragraph in doc["paragraphs"]:
            sentences = paragraph["sentences"]
            sentence_spans = paragraph["sentence_breaks"]

            # Process each QA pair
            for qa in paragraph["qas"]:
                question = qa["question"].strip()

                # Find matching qid
                if question in queries.values():
                    # Find qid for this question
                    qid = next(qid for qid, q in queries.items() if q == question)

                    # Process each answer
                    for answer in qa["answers"]:
                        answer_start = answer["answer_start"]
                        answer_end = answer_start + len(answer["text"])

                        # Find which sentence contains the answer
                        for sid, (sent_start, sent_end) in enumerate(sentence_spans):
                            # Check if answer is within sentence span, including punctuation
                            if sent_start <= answer_start and answer_end <= sent_end:
                                # Find matching did for the sentence
                                sentence = sentences[sid].strip()
                                if sentence in documents:
                                    did = documents[sentence]
                                    # Only add if not already in the list
                                    if did not in qrels[qid]:
                                        qrels[qid].append(did)

    return qrels

def save_qrels(qrels, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for qid, dids in qrels.items():
            for did in dids:
                f.write(f"{qid} 0 {did} 1\n")

def get_output_path(input_path):
    """Generate output path based on input path."""
    input_path = Path(input_path)
    output_filename = f"{input_path.stem}_qrels_new.tsv"
    return "data/xorqa-r/"+output_filename

def main():
    parser = argparse.ArgumentParser(description='Generate qrels file from LAREQA dataset')
    parser.add_argument('--input_file', type=str, help='Path to the input JSON file')

    args = parser.parse_args()

    # Generate output path if not provided
    args.output_file = get_output_path(args.input_file)

    # Generate qrels
    qrels = generate_qrels(args.input_file)

    # Save qrels
    save_qrels(qrels, args.output_file)

    print(f"Output saved to: {args.output_file}")

if __name__ == "__main__":
    main()

