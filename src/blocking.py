import pandas as pd
from utils import load_train_data
import numpy as np

class BlockingPipeline:
    def __init__(self, threshold=0.3):
        self.threshold = threshold
    
    def tokenize(self, text):
        """Split text into tokens"""
        if pd.isna(text):
            return set()
        return set(str(text).lower().split())
    
    def jaccard(self, tokens1, tokens2):
        """Jaccard similarity between two token sets"""
        if not tokens1 or not tokens2:
            return 0
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        return intersection / union if union > 0 else 0
    
    def get_candidate_pairs(self, s1_df, s2_df, s3_df):
        """Generate candidate pairs using name token Jaccard similarity"""
        candidates = []
        
        # Check S1 vs S2
        for idx, s1_row in s1_df.iterrows():
            s1_id = s1_row['entity_id']
            s1_name_tokens = self.tokenize(s1_row['business_name'])
            
            for jdx, s2_row in s2_df.iterrows():
                s2_name_tokens = self.tokenize(s2_row['business_name'])
                jaccard_score = self.jaccard(s1_name_tokens, s2_name_tokens)
                
                if jaccard_score > self.threshold:
                    candidates.append({
                        'source1_entity_id': s1_id,
                        'candidate_entity_id': s2_row['entity_id'],
                        'score': jaccard_score
                    })
            
            # Check S1 vs S3
            for kdx, s3_row in s3_df.iterrows():
                s3_name_tokens = self.tokenize(s3_row['business_name'])
                jaccard_score = self.jaccard(s1_name_tokens, s3_name_tokens)
                
                if jaccard_score > self.threshold:
                    candidates.append({
                        'source1_entity_id': s1_id,
                        'candidate_entity_id': s3_row['entity_id'],
                        'score': jaccard_score
                    })
            
            # Progress indicator
            if (idx + 1) % 500 == 0:
                print(f"  Processed {idx + 1}/{len(s1_df)} S1 records...")
        
        return pd.DataFrame(candidates)
    
    def compute_recall(self, candidates_df, gt_df):
        """Measure how many true matches we captured in candidates"""
        true_positives = 0
        total_matches = 0
        
        for idx, gt_row in gt_df.iterrows():
            s1_id = gt_row['source1_entity_id']
            matched_ids_str = gt_row['matched_entity_ids']
            
            # Skip singletons (no matches)
            if pd.isna(matched_ids_str) or matched_ids_str == '':
                continue
            
            matched_ids = str(matched_ids_str).split(',')
            total_matches += len(matched_ids)
            
            # Check if each true match is in our candidates
            for mid in matched_ids:
                cand = candidates_df[
                    (candidates_df['source1_entity_id'] == s1_id) &
                    (candidates_df['candidate_entity_id'] == mid)
                ]
                if len(cand) > 0:
                    true_positives += 1
        
        recall = true_positives / total_matches if total_matches > 0 else 0
        return recall, true_positives, total_matches

if __name__ == '__main__':
    print("Loading training data...")
    s1, s2, s3, gt = load_train_data()
    
    print("\n" + "="*60)
    print("BLOCKING PIPELINE - Testing threshold=0.3")
    print("="*60)
    
    blocker = BlockingPipeline(threshold=0.3)
    print("\nGenerating candidates...")
    candidates = blocker.get_candidate_pairs(s1, s2, s3)
    
    print(f"\nGenerated {len(candidates):,} candidate pairs")
    
    # Compute recall
    recall, tp, total = blocker.compute_recall(candidates, gt)
    print(f"Blocking Recall: {100*recall:.1f}% ({tp:,}/{total:,})")
    
    # If recall too low, try lower threshold
    if recall < 0.90:
        print("\n⚠️  Recall < 90%. Trying threshold=0.2...")
        blocker = BlockingPipeline(threshold=0.2)
        print("Generating candidates...")
        candidates = blocker.get_candidate_pairs(s1, s2, s3)
        
        print(f"\nGenerated {len(candidates):,} candidate pairs")
        recall, tp, total = blocker.compute_recall(candidates, gt)
        print(f"Blocking Recall: {100*recall:.1f}% ({tp:,}/{total:,})")
    
    if recall < 0.90:
        print("\n⚠️  Recall still < 90%. Trying threshold=0.1...")
        blocker = BlockingPipeline(threshold=0.1)
        print("Generating candidates...")
        candidates = blocker.get_candidate_pairs(s1, s2, s3)
        
        print(f"\nGenerated {len(candidates):,} candidate pairs")
        recall, tp, total = blocker.compute_recall(candidates, gt)
        print(f"Blocking Recall: {100*recall:.1f}% ({tp:,}/{total:,})")
    
    # Save candidates
    print(f"\n✅ Saving to output/train_candidates.tsv...")
    candidates.to_csv('output/train_candidates.tsv', sep='\t', index=False)
    print("Done!")