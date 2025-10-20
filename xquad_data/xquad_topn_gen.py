import argparse
from pathlib import Path
import random
from collections import defaultdict
import re

# Set random seed for reproducibility
random.seed(42)

def is_valid_id_start(line):
    """Check if line starts with Q or D followed by numbers."""
    return bool(re.match(r'^[QD]\d+', line))

def read_queries(query_file):
    """Read queries from TSV file and return a dictionary of qid to query text."""
    queries = {}  # Changed to {qid: query} mapping
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
                        # Strip whitespace from query text
                        query = query.strip()
                        queries[qid] = query  # Changed to store qid as key
                    else:
                        print(f"Warning: Invalid line format in query file: {current_line}")
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
                # Strip whitespace from query text
                query = query.strip()
                queries[qid] = query  # Changed to store qid as key
            else:
                print(f"Warning: Invalid line format in query file: {current_line}")
    return queries

def read_documents(doc_file):
    """Read documents from TSV file and return a dictionary of did to document text."""
    documents = {}  # Changed to {did: doc} mapping
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
                        documents[did] = doc  # Changed to store did as key
                    else:
                        print(f"Warning: Invalid line format in document file: {current_line}")
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
                documents[did] = doc  # Changed to store did as key
            else:
                print(f"Warning: Invalid line format in document file: {current_line}")
    return documents

def read_qrels(qrels_file):
    """Read qrels file and return a dictionary of qid to list of relevant dids."""
    qrels = defaultdict(list)
    with open(qrels_file, 'r', encoding='utf-8') as f:
        for line in f:
            qid, _, did, _ = line.strip().split()
            qrels[qid].append(did)
    return qrels

def generate_topn_candidates(queries, documents, qrels, n=100, output_file=None):
    """Generate top-n candidate sentences for each query."""
    # Create a list of all document IDs for random selection
    all_dids = list(documents.keys())
    
    # Track statistics
    total_candidates = 0
    
    with open(output_file, 'w', encoding='utf-8') as fout:
        for qid, query in queries.items():
            # Clean query text
            clean_query = query.replace("\n", " ").strip()
            
            # Get relevant documents for this query
            relevant_dids = qrels[qid]
            candidates_count = 0
            candidate_pairs = []  # Store (did, doc) pairs for shuffling
            
            # First add all relevant documents
            for did in relevant_dids:
                # Clean document text
                clean_doc = documents[did].replace("\n", " ").strip()
                candidate_pairs.append((did, clean_doc))
                candidates_count += 1
            
            # Calculate how many more documents we need
            remaining = n - candidates_count
            if remaining > 0:
                # Get all non-relevant documents
                non_relevant_dids = [did for did in all_dids if did not in relevant_dids]
                # Randomly select remaining documents
                selected_dids = random.sample(non_relevant_dids, min(remaining, len(non_relevant_dids)))
                # Add selected documents
                for did in selected_dids:
                    # Clean document text
                    clean_doc = documents[did].replace("\n", " ").strip()
                    candidate_pairs.append((did, clean_doc))
                    candidates_count += 1
            
            # Shuffle the candidate pairs
            random.shuffle(candidate_pairs)
            
            # Write shuffled candidates to file
            for did, doc in candidate_pairs:
                fout.write(f"{qid}\t{did}\t{clean_query}\t{doc}\n")
            
            total_candidates += candidates_count
    
    print(f"\nTotal query-doc pairs saved: {total_candidates}")

def main():
    parser = argparse.ArgumentParser(description='Generate top-n candidate sentences for each query')
    parser.add_argument('--query_lang', type=str, required=True, choices=['en', 'de', 'ar', 'ru'],
                      help='Language code for queries (en, de, ar, ru)')
    parser.add_argument('--sent_lang', type=str, required=True, choices=['en', 'de', 'ar', 'ru'],
                      help='Language code for documents (en, de, ar, ru)')
    parser.add_argument('--n', type=int, default=100, help='Number of candidate sentences per query')
    
    args = parser.parse_args()
    
    # Construct file paths
    base_dir = Path("data/xquad-r/")
    query_file = base_dir / f"{args.query_lang}_queries.tsv"
    doc_file = base_dir / f"{args.sent_lang}_sentences.tsv"
    qrels_file = base_dir / f"{args.sent_lang}_qrels.tsv"
    
    # Create output directory if it doesn't exist
    output_dir = Path("data/xquad-topn")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{args.query_lang}-{args.sent_lang}_top{args.n}_shuffled.dev"
    
    # Read input files
    print("Reading queries...")
    queries = read_queries(query_file)
    print(f"Loaded {len(queries)} queries")
    
    print("Reading sentences...")
    documents = read_documents(doc_file)
    print(f"Loaded {len(documents)} sentences")
    
    print("Reading qrels...")
    qrels = read_qrels(qrels_file)
    print(f"Loaded relevance judgments for {len(qrels)} queries")
    
    # Generate candidates
    print(f"\nGenerating top-{args.n} candidates...")
    generate_topn_candidates(queries, documents, qrels, args.n, output_file)
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    main()

