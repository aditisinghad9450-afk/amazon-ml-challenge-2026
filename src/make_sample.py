"""Writes small samples of train source2/source3 (first 300k rows) to dataset/sample/
so they can be shared for error analysis. Run from project root: python src/make_sample.py"""
import os
os.makedirs("dataset/sample", exist_ok=True)
for s in (2, 3):
    src = f"dataset/train/train_source{s}.tsv"
    dst = f"dataset/sample/train_source{s}_sample.tsv"
    with open(src, encoding="utf-8") as a, open(dst, "w", encoding="utf-8") as b:
        for n, line in enumerate(a):
            if n > 300_000:
                break
            b.write(line)
    print("wrote", dst, os.path.getsize(dst) // 1_000_000, "MB")
