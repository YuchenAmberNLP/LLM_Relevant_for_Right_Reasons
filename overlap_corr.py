import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple
import re
from tqdm import tqdm
import os
import pandas as pd
from scipy import stats


lang_pair_to_folder = {
    "en-en": "english-english",
    "de-de": "german-german",
    "ar-ar": "arabic-arabic",
    "it-it": "italian-italian",
    "ru-ru": "russian-russian",
    "en-de": "english-german",
    "en-ar": "english-arabic",
    "en-it": "english-italian",
    "de-it": "german-italian",
    "de-ru": "german-russian",
    "ar-it": "arabic-italian",
    "ar-ru": "arabic-russian"
}

def preprocess_text(text: str) -> Dict[str, int]:
    """Count token frequencies in text, converting to lowercase."""
    tokens = text.lower().split()
    return {token: tokens.count(token) for token in set(tokens)}

def lexical_overlap_odds(query: str, pos_docs: List[str], neg_docs: List[str]) -> Tuple[float, float]:
    """
    Calculate average lexical overlap between query and positive/negative documents
    based on token frequencies.
    
    Args:
        query: The query text
        pos_docs: List of relevant document texts
        neg_docs: List of irrelevant document texts
    
    Returns:
        Tuple of (avg_overlap_with_pos, avg_overlap_with_neg)
    """
    # Method 1: Based on token frequencies
    """
    query_freq = preprocess_text(query)
    
    def calculate_overlap(doc):
        doc_freq = preprocess_text(doc)
        overlap = sum(min(query_freq.get(token, 0), doc_freq.get(token, 0)) 
                     for token in set(query_freq.keys()) | set(doc_freq.keys()))
        return overlap
    
    pos_overlaps = [calculate_overlap(doc) for doc in pos_docs]
    neg_overlaps = [calculate_overlap(doc) for doc in neg_docs]
    
    avg_pos_overlap = np.mean(pos_overlaps) if pos_overlaps else 0.0
    avg_neg_overlap = np.mean(neg_overlaps) if neg_overlaps else 0.0
    """
    
    # Method 2: Based on token intersection
    query_tokens = set(query.lower().split())
    
    def calculate_overlap(doc):
        doc_tokens = set(doc.lower().split())
        if not doc_tokens:
            return 0.0
        overlap = len(query_tokens.intersection(doc_tokens))
        return overlap
    
    pos_overlaps = [calculate_overlap(doc) for doc in pos_docs]
    neg_overlaps = [calculate_overlap(doc) for doc in neg_docs]
    
    avg_pos_overlap = np.mean(pos_overlaps) if pos_overlaps else 0.0
    avg_neg_overlap = np.mean(neg_overlaps) if neg_overlaps else 0.0
    
    return avg_pos_overlap, avg_neg_overlap

def calculate_ap(ranked_docs: List[str], relevant_docs: set, top_k: int = 10) -> float:
    """
    Calculate Average Precision (AP) for a ranked list of documents.
    Only considers the top k documents.
    
    Args:
        ranked_docs: List of document IDs in ranked order
        relevant_docs: Set of relevant document IDs
        top_k: Number of top documents to consider (default: 10)
    
    Returns:
        Average Precision score
    """
    if not relevant_docs:
        return 0.0
    
    ap = 0.0
    num_relevant = 0
    
    # Only consider top k documents
    for i, doc_id in enumerate(ranked_docs[:top_k]):
        if doc_id in relevant_docs:
            num_relevant += 1
            precision = num_relevant / (i + 1)
            ap += precision
    
    # If there are no relevant documents in top k, return 0
    if num_relevant == 0:
        return 0.0
        
    return ap / num_relevant

def odds_ap_corr(top100_file: str, rank_result_path: str, qrel_path: str, output_file: str, top_k: int = 10):
    """
    Calculate lexical overlap odds and AP for each query and save results.
    
    Args:
        top100_file: Path to the top100.dev file
        rank_result_path: Path to the reranking results file
        qrel_path: Path to the qrels file
        output_file: Path to save the results
        top_k: Number of top documents to consider for AP calculation (default: 10)
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Read qrels to get relevant documents
    qrels = defaultdict(set)
    with open(qrel_path, 'r') as f:
        for line in f:
            # qrels format: queryID ignore passageID ignore
            qid, _, docid, _ = line.strip().split()
            qrels[qid].add(docid)
    
    # Read reranking results
    ranked_docs = defaultdict(list)
    with open(rank_result_path, 'r') as f:
        for line in f:
            # reranking results format: queryID ignored docID rank score runid
            qid, _, docid, _, score, _ = line.strip().split()
            ranked_docs[qid].append((docid, float(score)))
    
    # Sort documents by score in descending order
    for qid in ranked_docs:
        ranked_docs[qid].sort(key=lambda x: x[1], reverse=True)
        ranked_docs[qid] = [docid for docid, _ in ranked_docs[qid]]
    
    # Process top100 file and calculate metrics
    results = []
    current_qid = None
    current_query = None
    current_docs = []
    
    with open(top100_file, 'r') as f:
        for line in tqdm(f):
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                qid, docid, query, *doc_text = parts
                doc_text = ' '.join(doc_text)
                
                if qid != current_qid:
                    if current_qid is not None:
                        # Process previous query
                        pos_docs = [doc for doc, is_rel in current_docs if is_rel]
                        neg_docs = [doc for doc, is_rel in current_docs if not is_rel]
                        pos_overlap, neg_overlap = lexical_overlap_odds(current_query, pos_docs, neg_docs)
                        ap = calculate_ap(ranked_docs[current_qid], qrels[current_qid], top_k)
                        results.append((current_qid, pos_overlap, neg_overlap, ap))
                    
                    current_qid = qid
                    current_query = query
                    current_docs = []
                
                is_relevant = docid in qrels[qid]
                current_docs.append((doc_text, is_relevant))
    
    # Process the last query
    if current_qid is not None:
        pos_docs = [doc for doc, is_rel in current_docs if is_rel]
        neg_docs = [doc for doc, is_rel in current_docs if not is_rel]
        pos_overlap, neg_overlap = lexical_overlap_odds(current_query, pos_docs, neg_docs)
        ap = calculate_ap(ranked_docs[current_qid], qrels[current_qid], top_k)
        results.append((current_qid, pos_overlap, neg_overlap, ap))
    
    # Save results
    with open(output_file, 'w') as f:
        f.write('qid\tpos_overlap\tneg_overlap\toverlap_odds\toverlap_diff\tap\n')
        for qid, pos_overlap, neg_overlap, ap in results:
            # Calculate odds, avoid division by zero
            overlap_odds = pos_overlap / neg_overlap if neg_overlap > 0 else float('inf')
            # Calculate difference
            overlap_diff = pos_overlap - neg_overlap
            f.write(f'{qid}\t{pos_overlap}\t{neg_overlap}\t{overlap_odds}\t{overlap_diff}\t{ap}\n')
    
    print(f"Results saved to {output_file}")

def analyze_results():
    """
    Analyze the results for each model and language pair combination.
    """
    # Define models and language pairs
    models = [
        "Llama-3.1-8B-Instruct",
        "model_output_enen_checkpoint-7812",
        "model_output_cs_checkpoint-7812",
        "model_output_xxxx_checkpoint-7812"
    ]
    
    language_pairs = [
        "en-en",
        "de-de",
        "ar-ar",
        "it-it",
        "ru-ru",
        "en-de",
        "en-ar",
        "en-it",
        "de-it",
        "de-ru",
        "ar-it",
        "ar-ru"
    ]
    
    # Define language groups
    monolingual_pairs = ["en-en", "de-de", "ar-ar", "it-it", "ru-ru"]
    cl_pairs = [pair for pair in language_pairs if pair not in monolingual_pairs]
    
    # Initialize results storage
    results = defaultdict(dict)
    group_data = defaultdict(lambda: defaultdict(lambda: {'overlap_diff': [], 'ap': []}))
    
    # Process each combination
    for lang_pair in language_pairs:
        folder_name = lang_pair_to_folder[lang_pair]
        for model in models:
            print(f"\nProcessing {lang_pair} with {model}")
            input_file = f"overlap_ap_corr/mmarco_shuffle/top100_{folder_name}/{model}/query_metrics.tsv"
            
            # Read the metrics file
            df = pd.read_csv(input_file, sep='\t')
            
            # Calculate statistics
            avg_overlap_diff = df['overlap_diff'].mean()
            pearson_corr = stats.pearsonr(df['overlap_diff'], df['ap'])[0]
            spearman_corr = stats.spearmanr(df['overlap_diff'], df['ap'])[0]
            kendall_corr = stats.kendalltau(df['overlap_diff'], df['ap'])[0]
            
            # Store results
            results[model][lang_pair] = {
                'avg_overlap_diff': avg_overlap_diff,
                'pearson_corr': pearson_corr,
                'spearman_corr': spearman_corr,
                'kendall_corr': kendall_corr
            }
            
            # Store data for group analysis
            group = 'monolingual' if lang_pair in monolingual_pairs else 'crosslingual'
            group_data[model][group]['overlap_diff'].extend(df['overlap_diff'].tolist())
            group_data[model][group]['ap'].extend(df['ap'].tolist())
    
    # Calculate group statistics
    group_results = defaultdict(dict)
    for model in models:
        for group in ['monolingual', 'crosslingual']:
            print(f"\nProcessing {group} group with {model}")
            overlap_diffs = group_data[model][group]['overlap_diff']
            aps = group_data[model][group]['ap']
            
            group_results[model][group] = {
                'avg_overlap_diff': np.mean(overlap_diffs),
                'pearson_corr': stats.pearsonr(overlap_diffs, aps)[0],
                'spearman_corr': stats.spearmanr(overlap_diffs, aps)[0],
                'kendall_corr': stats.kendalltau(overlap_diffs, aps)[0]
            }
    
    # Save detailed results
    with open('overlap_ap_corr/mmarco_shuffle_analysis_results.tsv', 'w') as f:
        f.write('model\tlanguage_pair\tavg_overlap_diff\tpearson_corr\tspearman_corr\tkendall_corr\n')
        for model in models:
            for lang_pair in language_pairs:
                f.write(f'{model}\t{lang_pair}\t{results[model][lang_pair]["avg_overlap_diff"]}\t'
                       f'{results[model][lang_pair]["pearson_corr"]}\t'
                       f'{results[model][lang_pair]["spearman_corr"]}\t'
                       f'{results[model][lang_pair]["kendall_corr"]}\n')
    
    # Save group results
    with open('overlap_ap_corr/mmarco_shuffle_group_analysis_results.tsv', 'w') as f:
        f.write('model\tgroup\tavg_overlap_diff\tpearson_corr\tspearman_corr\tkendall_corr\n')
        for model in models:
            for group in ['monolingual', 'crosslingual']:
                f.write(f'{model}\t{group}\t{group_results[model][group]["avg_overlap_diff"]}\t'
                       f'{group_results[model][group]["pearson_corr"]}\t'
                       f'{group_results[model][group]["spearman_corr"]}\t'
                       f'{group_results[model][group]["kendall_corr"]}\n')
    
    print("Analysis results saved to 'overlap_ap_corr/mmarco_shuffle_analysis_results.tsv' and 'overlap_ap_corr/mmarco_shuffle_group_analysis_results.tsv'")

def main():
    # Define models and language pairs
    models = [
        "Llama-3.1-8B-Instruct",
        "model_output_enen_checkpoint-7812",
        "model_output_cs_checkpoint-7812",
        "model_output_xxxx_checkpoint-7812"
    ]
    
    language_pairs = [
        "en-en",
        "de-de",
        "ar-ar",
        "it-it",
        "ru-ru",
        "en-de",
        "en-ar",
        "en-it",
        "de-it",
        "de-ru",
        "ar-it",
        "ar-ru"
    ]
    
    # Process each combination
    for lang_pair in language_pairs:
        # Get the second language for qrels path
        second_lang = lang_pair.split('-')[1]
        qrel_path = f"data/qrels.dev.tsv"
        
        folder_name = lang_pair_to_folder[lang_pair]
        # Update top100 file path
        top100_file = f"data/cl_topn/top100_shuffle_{folder_name}.dev"
        
        for model in models:
            # Update reranking results path
            rank_result_path = f"results/top100_shuffle_{folder_name}/{model}/full_rerank_results.tsv"
            
            # Update metrics output path
            output_file = f"overlap_ap_corr/mmarco_shuffle/top100_{folder_name}/{model}/query_metrics.tsv"
            
            print(f"\nProcessing results of {lang_pair} on {model}")
            odds_ap_corr(top100_file, rank_result_path, qrel_path, output_file, top_k=10)

if __name__ == "__main__":
    main()
    analyze_results()

