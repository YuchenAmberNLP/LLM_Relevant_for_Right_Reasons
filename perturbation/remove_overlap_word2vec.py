import os
import glob
import json
import nltk
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
from ftfy import fix_text

nltk.download('punkt_tab')


print("Loading Word2Vec model...")
embedding_path = "word2vec/GoogleNews-vectors-negative300.bin.gz"
w2v = KeyedVectors.load_word2vec_format(embedding_path, binary=True)
print("Embedding loaded.")

def find_similar_word(word, exclude_set):
    try:
        similar = w2v.most_similar(word, topn=10)
        for candidate, _ in similar:
            clean = candidate.lower()
            if clean not in exclude_set and clean.isalpha():
                return candidate
    except KeyError:
        pass
    return word

def replace_overlap(query, passage, query_token_set):
    tokens = word_tokenize(passage)
    new_tokens = []
    for token in tokens:
        if token.lower() in query_token_set and token.isalpha():
            new_token = find_similar_word(token.lower(), query_token_set)
            new_tokens.append(new_token)
        else:
            new_tokens.append(token)
    return ' '.join(new_tokens)



def corrupt_file(input_path):
    output_path = input_path.replace(".jsonl", "_corrupted.jsonl")
    with open(input_path, 'r', encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
        for lineno, raw in enumerate(tqdm(fin, desc="Corrupting passages"), 1):
            raw = raw.strip()
            if not raw:
                continue

            try:
                line = fix_text(raw)
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"JSON decode error at line {lineno}: {e}")
                print(f"Raw repr: {repr(raw)}")
                continue

            query = data['query']
            model_output_key = 'passage_A' if data['model_output'] == 'Passage A' else 'passage_B'
            original_passage = data[model_output_key]['passage']
            query_tokens = set(word_tokenize(query.lower()))
            corrupted_passage = replace_overlap(query, original_passage, query_tokens)
            data[model_output_key]['passage'] = corrupted_passage
            fout.write(json.dumps(data, ensure_ascii=False) + '\n')
    print(f"\nDone. Corrupted file saved as: {output_path}")


if __name__ == "__main__":
    # corrupt_file("classification_results/classification_range/english-german_conflict_irrel_ge3_model_output_enen_errors.jsonl")
    pattern = "classification_results/classification_range/**/*conflict_irrel_ge3*_errors.jsonl"
    matching_files = glob.glob(pattern, recursive=True)
    print(f"Found {len(matching_files)} matching files.")

    for input_file in matching_files:
        print(f"\nProcessing: {input_file}")
        corrupt_file(input_file)