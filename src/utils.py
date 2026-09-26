import pandas as pd

def load_train_data():
    """Load all training data"""
    s1 = pd.read_csv('dataset/train/train_source1.tsv', sep='\t')
    s2 = pd.read_csv('dataset/train/train_source2.tsv', sep='\t')
    s3 = pd.read_csv('dataset/train/train_source3.tsv', sep='\t')
    gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t')
    
    return s1, s2, s3, gt

if __name__ == '__main__':
    print("Loading data...")
    s1, s2, s3, gt = load_train_data()
    
    print(f"S1: {len(s1)} records")
    print(f"S2: {len(s2)} records")
    print(f"S3: {len(s3)} records")
    print(f"Ground truth: {len(gt)} records")