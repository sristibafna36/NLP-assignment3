#!/usr/bin/env python3
"""
Find the minimum prior probability p(gen) where all dev files are classified as spam.
Usage: python find_threshold.py gen.model spam.model
"""

import sys
import math
from pathlib import Path
from probs import LanguageModel

def classify_files(gen_model, spam_model, prior_gen, file_paths):
    """Classify files and return (gen_count, spam_count)."""
    gen_count = 0
    spam_count = 0
    
    log_prior_gen = math.log(prior_gen) if prior_gen > 0 else float('-inf')
    log_prior_spam = math.log(1 - prior_gen) if prior_gen < 1 else float('-inf')
    
    for file_path in file_paths:
        # Get log probabilities from models
        log_prob_gen = gen_model.file_log_prob(file_path)
        log_prob_spam = spam_model.file_log_prob(file_path)
        
        # Add log priors to get log posteriors
        log_posterior_gen = log_prob_gen + log_prior_gen
        log_posterior_spam = log_prob_spam + log_prior_spam
        
        # MAP classification
        if log_posterior_gen > log_posterior_spam:
            gen_count += 1
        else:
            spam_count += 1
    
    return gen_count, spam_count

def binary_search_threshold(gen_model, spam_model, file_paths, tolerance=1e-8):
    """Binary search to find minimum p(gen) where all files are spam."""
    low = 0.0
    high = 1.0
    
    # Verify that high doesn't classify all as spam already
    gen_count, spam_count = classify_files(gen_model, spam_model, high, file_paths)
    print(f"At p(gen)={high}: {gen_count} gen, {spam_count} spam")
    
    if spam_count == len(file_paths):
        print("All files already classified as spam at p(gen)=1.0")
        return 1.0
    
    print("\nStarting binary search...\n")
    
    while high - low > tolerance:
        mid = (low + high) / 2
        gen_count, spam_count = classify_files(gen_model, spam_model, mid, file_paths)
        
        print(f"p(gen)={mid:.10f}: {gen_count} gen, {spam_count} spam")
        
        if gen_count == 0:
            # All classified as spam, we can go higher
            low = mid
        else:
            # Some still gen, need to go lower
            high = mid
    
    return high

def main():
    if len(sys.argv) != 3:
        print("Usage: python find_threshold.py gen.model spam.model")
        sys.exit(1)
    
    gen_model_path = Path(sys.argv[1])
    spam_model_path = Path(sys.argv[2])
    
    print("Loading models...")
    gen_model = LanguageModel.load(gen_model_path)
    spam_model = LanguageModel.load(spam_model_path)
    
    # Collect all dev files
    dev_dir = Path('data/gen_spam/dev')
    gen_files = sorted((dev_dir / 'gen').glob('*'))
    spam_files = sorted((dev_dir / 'spam').glob('*'))
    all_files = gen_files + spam_files
    
    print(f"Found {len(gen_files)} gen files and {len(spam_files)} spam files")
    print(f"Total: {len(all_files)} dev files\n")
    
    # Find threshold
    threshold = binary_search_threshold(gen_model, spam_model, all_files)
    
    print(f"\n{'='*60}")
    print(f"RESULT: Minimum p(gen) for all-spam classification:")
    print(f"p(gen) < {threshold:.10f}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()