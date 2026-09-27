"""Optional: make gzip copies of the big source files (pipeline reads .tsv or .tsv.gz).
Run from the project root:  python src/compress_data.py"""
import glob, gzip, shutil
for f in glob.glob("dataset/*/*source[23].tsv"):
    print("compressing", f)
    with open(f, "rb") as a, gzip.open(f + ".gz", "wb", 3) as b:
        shutil.copyfileobj(a, b)
print("done")
