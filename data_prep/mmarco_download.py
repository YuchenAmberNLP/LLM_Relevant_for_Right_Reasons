from huggingface_hub import hf_hub_download
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import shutil

BASE_REPO = "unicamp-dl/mmarco"   # HF dataset repo
REPO_TYPE = "dataset"

SHORT2FULL = {
    "en": "english",
    "de": "german",
    "ar": "arabic",
    "it": "italian",
    "ru": "russian",
}

def read_lang_sets(pairs_file: str):
    langs = set()
    with open(pairs_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 2:
                langs.update(parts)
    return sorted(langs)

def short2full_or_raise(code: str) -> str:
    if code not in SHORT2FULL:
        raise ValueError(f"Language code '{code}' not found in SHORT2FULL mapping.")
    return SHORT2FULL[code]

def download_one(remote_path: str, save_dir: Path) -> bool:
    save_dir.mkdir(parents=True, exist_ok=True)
    try:
        print(f"Downloading: {remote_path}")
        local_tmp = hf_hub_download(
            repo_id=BASE_REPO,
            filename=remote_path,
            repo_type=REPO_TYPE,
            local_dir=str(save_dir),
        )
        target = save_dir / Path(remote_path).name
        if Path(local_tmp) != target:
            shutil.move(local_tmp, target)
        print(f"Saved to {target}")
        return True
    except Exception as e:
        print(f"[ERROR] {remote_path}: {e}")
        return False

def main(args):
    langs = read_lang_sets(args.lang_pairs)

    collections_home = Path(args.collections_home)
    queries_home = Path(args.queries_home) / "dev"

    tasks = []
    ok = fail = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for code in langs:
            lang_full = short2full_or_raise(code)

            # collection
            rp_col = f"data/google/collections/{lang_full}_collection.tsv"
            tasks.append(ex.submit(download_one, rp_col, collections_home))

            # queries (dev)
            rp_q = f"data/google/queries/dev/{lang_full}_queries.dev.tsv"
            tasks.append(ex.submit(download_one, rp_q, queries_home))

        for fut in as_completed(tasks):
            if fut.result():
                ok += 1
            else:
                fail += 1

    print("\n=== Summary ===")
    print(f"success: {ok}, failed: {fail}")
    print(f"Collections saved under: {collections_home}")
    print(f"Queries saved under:     {queries_home}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Download mMARCO collections/queries (dev only) using full language names.")
    p.add_argument("--lang_pairs", default="data/mmarco_langpair.txt", help="Path to language pairs file, two cols per line, e.g., 'en de'.")
    p.add_argument("--collections_home", default="data/mmarco/collections", help="Where to save collection TSVs.")
    p.add_argument("--queries_home", default="data/mmarco/queries", help="Where to save query TSVs.")
    p.add_argument("--workers", type=int, default=5)
    args = p.parse_args()
    main(args)