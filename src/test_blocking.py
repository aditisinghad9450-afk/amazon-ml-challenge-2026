import pandas as pd
from utils import BlockingPipeline

def load_test_data():
    """Load test data"""
    s1 = pd.read_csv('dataset/test/test_source1.tsv', sep='\t')
    s2 = pd.read_csv('dataset/test/test_source2.tsv', sep='\t')
    s3 = pd.read_csv('dataset/test/test_source3.tsv', sep='\t')
    return s1, s2, s3

if __name__ == '__main__':
    print("Loading test data...")
    s1_test, s2_test, s3_test = load_test_data()
    
    print(f"S1 test: {len(s1_test)} records")
    print(f"S2 test: {len(s2_test)} records")
    print(f"S3 test: {len(s3_test)} records")
    
    print("\nGenerating test candidates (threshold=0.3)...")
    blocker = BlockingPipeline(threshold=0.3)
    test_candidates = blocker.get_candidate_pairs(s1_test, s2_test, s3_test)
    
    print(f"Generated {len(test_candidates):,} test candidate pairs")
    
    # Save
    print("Saving to output/candidate_pairs.tsv...")
    
    # Format for submission: group by source1_entity_id
    output = []
    for s1_id in s1_test['entity_id']:
        cand_ids = test_candidates[test_candidates['source1_entity_id'] == s1_id]['candidate_entity_id'].tolist()
        output.append({
            'source1_entity_id': s1_id,
            'candidate_entity_ids': ','.join(cand_ids) if cand_ids else ''
        })
    
    output_df = pd.DataFrame(output)
    output_df.to_csv('output/candidate_pairs.tsv', sep='\t', index=False)
    print("✅ Done!")