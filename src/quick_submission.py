import pandas as pd
import numpy as np

print("Loading S1...")
s1 = pd.read_csv('dataset/train/train_source1.tsv', sep='\t')

print("Sampling S2 and S3...")
# Read only first 50K records from S2/S3 (memory efficient)
s2 = pd.read_csv('dataset/train/train_source2.tsv', sep='\t', nrows=50000)
s3 = pd.read_csv('dataset/train/train_source3.tsv', sep='\t', nrows=50000)

print(f"S1: {len(s1)}, S2 (sample): {len(s2)}, S3 (sample): {len(s3)}")

candidates = []

# Simple: take top 20 matches per S1 entity (random for now)
for idx, s1_row in s1.iterrows():
    # Random S2 matches
    if len(s2) > 0:
        s2_sample = np.random.choice(s2['entity_id'].values, min(10, len(s2)), replace=False)
        for s2_id in s2_sample:
            candidates.append({'source1_entity_id': s1_row['entity_id'], 'candidate_entity_id': s2_id})
    
    # Random S3 matches
    if len(s3) > 0:
        s3_sample = np.random.choice(s3['entity_id'].values, min(10, len(s3)), replace=False)
        for s3_id in s3_sample:
            candidates.append({'source1_entity_id': s1_row['entity_id'], 'candidate_entity_id': s3_id})
    
    if (idx + 1) % 5000 == 0:
        print(f"  Processed {idx+1}/{len(s1)}")

candidates_df = pd.DataFrame(candidates)
print(f"\n✅ Generated {len(candidates_df):,} candidates")

candidates_df.to_csv('output/candidate_pairs.tsv', sep='\t', index=False)
print("Saved to output/candidate_pairs.tsv!")