import json
import random
from pathlib import Path
from collections import defaultdict
import argparse
random.seed(42)

def load_queries(queries_file):
    """Load queries from TSV file."""
    queries = {}
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            try:
                parts = line.split('\t')
                if len(parts) != 2:
                    print(f"Warning: Line {line_num} in {queries_file} has incorrect format: {line}")
                    continue
                qid, query = parts
                queries[qid] = query
            except Exception as e:
                print(f"Error processing line {line_num} in {queries_file}: {e}")
                continue
    return queries

def load_documents(docs_file):
    """Load documents from TSV file."""
    documents = {}
    with open(docs_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            try:
                parts = line.split('\t')
                if len(parts) != 2:
                    print(f"Warning: Line {line_num} in {docs_file} has incorrect format: {line}")
                    continue
                pid, passage = parts
                # Replace all newlines with spaces and normalize whitespace
                passage = ' '.join(passage.split())
                documents[pid] = passage
            except Exception as e:
                print(f"Error processing line {line_num} in {docs_file}: {e}")
                continue
    return documents

def load_qrels(qrels_file):
    """Load qrels from TSV file."""
    qrels = defaultdict(list)
    with open(qrels_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            try:
                parts = line.split()
                if len(parts) != 4:
                    print(f"Warning: Line {line_num} in {qrels_file} has incorrect format: {line}")
                    continue
                qid, _, pid, _ = parts
                qrels[qid].append(pid)
            except Exception as e:
                print(f"Error processing line {line_num} in {qrels_file}: {e}")
                continue
    return qrels

def generate_training_data(queries, documents, qrels, output_file, neg_samples=5):
    """Generate training data with positive and negative samples."""
    # Get all document IDs for negative sampling
    all_doc_ids = list(documents.keys())
    
    # Generate training examples
    training_data = []
    
    for qid, query in queries.items():
        # Get relevant documents for this query
        relevant_docs = qrels[qid]
        
        # Skip if no relevant documents
        if not relevant_docs:
            print(f"Warning: Query {qid} has no relevant documents")
            continue
        
        # For each relevant document, create a positive example
        for rel_doc_id in relevant_docs:
            # Skip if relevant document not found in documents
            if rel_doc_id not in documents:
                print(f"Warning: Relevant document {rel_doc_id} not found in documents")
                continue
                
            # Positive example
            training_data.append({
                "qid": qid,
                "query": query,
                "pid": rel_doc_id,
                "passage": documents[rel_doc_id],
                "label": 1
            })
            
            # Generate negative examples
            # Get all documents except the relevant ones
            negative_candidates = [doc_id for doc_id in all_doc_ids 
                                if doc_id not in relevant_docs]
            
            # Randomly sample negative examples
            if len(negative_candidates) >= neg_samples:
                neg_docs = random.sample(negative_candidates, neg_samples)
            else:
                neg_docs = negative_candidates
            
            # Create negative examples
            for neg_doc_id in neg_docs:
                training_data.append({
                    "qid": qid,
                    "query": query,
                    "pid": neg_doc_id,
                    "passage": documents[neg_doc_id],
                    "label": 0
                })
    
    # Shuffle the training data
    random.shuffle(training_data)
    
    # Write to jsonl file
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in training_data:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    # Print statistics
    pos_count = sum(1 for ex in training_data if ex["label"] == 1)
    neg_count = sum(1 for ex in training_data if ex["label"] == 0)
    print(f"\nGenerated training data:")
    print(f"Total examples: {len(training_data)}")
    print(f"Positive examples: {pos_count}")
    print(f"Negative examples: {neg_count}")

def main():
    parser = argparse.ArgumentParser(description='Generate training jsonl file from queries, documents and qrels')
    parser.add_argument('--queries_file', type=str, required=True, help='Path to queries TSV file')
    parser.add_argument('--docs_file', type=str, required=True, help='Path to documents TSV file')
    parser.add_argument('--qrels_file', type=str, required=True, help='Path to qrels TSV file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to output jsonl file')
    parser.add_argument('--neg_samples', type=int, default=5, help='Number of negative samples per positive example')
    
    args = parser.parse_args()
    
    # Load data
    print("Loading queries...")
    queries = load_queries(args.queries_file)
    print(f"Loaded {len(queries)} queries")
    
    print("\nLoading documents...")
    documents = load_documents(args.docs_file)
    print(f"Loaded {len(documents)} documents")
    
    print("\nLoading qrels...")
    qrels = load_qrels(args.qrels_file)
    print(f"Loaded qrels for {len(qrels)} queries")
    
    # Generate training data
    print("\nGenerating training data...")
    generate_training_data(queries, documents, qrels, args.output_file, args.neg_samples)

if __name__ == "__main__":
    main() 