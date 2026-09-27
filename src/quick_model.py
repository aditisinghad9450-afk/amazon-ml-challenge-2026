import pandas as pd
import numpy as np

print("Loading candidates...")
candidates = pd.read_csv('output/candidate_pairs.tsv', sep='\t')

print("Loading S1...")
s1 = pd.read_csv('dataset/train/train_source1.tsv', sep='\t')

print("Generating matching results...")

results = []
for s1_id in s1['entity_id']:
    cand_ids = candidates[candidates['source1_entity_id'] == s1_id]['candidate_entity_id'].tolist()
    # Keep top 5 candidates per S1
    results.append({
        'source1_entity_id': s1_id,
        'matched_entity_ids': ','.join(cand_ids[:5]) if cand_ids else ''
    })

results_df = pd.DataFrame(results)
results_df.to_csv('output/matching_results.tsv', sep='\t', index=False)
print(f"✅ Generated {len(results_df)} matching results")
print("Saved to output/matching_results.tsv!")