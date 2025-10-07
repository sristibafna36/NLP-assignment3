#!/usr/bin/env python3
"""
Text categorization using Bayes' Theorem with two language models.
"""
import argparse
import logging
import math
from pathlib import Path
import torch

from probs import LanguageModel, read_trigrams

log = logging.getLogger(Path(__file__).stem)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model1",
        type=Path,
        help="path to first trained model",
    )
    parser.add_argument(
        "model2",
        type=Path,
        help="path to second trained model",
    )
    parser.add_argument(
        "prior",
        type=float,
        help="prior probability of first category",
    )
    parser.add_argument(
        "test_files",
        type=Path,
        nargs="*"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=['cpu','cuda','mps'],
        help="device to use for PyTorch"
    )
    
    parser.set_defaults(logging_level=logging.INFO)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose", dest="logging_level", action="store_const", const=logging.DEBUG
    )
    verbosity.add_argument(
        "-q", "--quiet", dest="logging_level", action="store_const", const=logging.WARNING
    )
    
    return parser.parse_args()


def file_log_prob(file: Path, lm: LanguageModel) -> float:
    """Return total log-probability of file under the language model."""
    log_prob = 0.0
    for (x, y, z) in read_trigrams(file, lm.vocab):
        log_prob += lm.log_prob(x, y, z)
        if log_prob == -math.inf:
            break
    return log_prob


def main():
    args = parse_args()
    logging.basicConfig(level=args.logging_level)
    
    if args.device == 'mps':
        if not torch.backends.mps.is_available():
            logging.critical("MPS not available")
            exit(1)
    torch.set_default_device(args.device)
    
    # Load both models
    log.info("Loading models...")
    lm1 = LanguageModel.load(args.model1, device=args.device)
    lm2 = LanguageModel.load(args.model2, device=args.device)
    
    # Extract model names from file paths
    model1_name = args.model1.stem
    model2_name = args.model2.stem
    
    # Determine which corpus each model was trained on
    model1_corpus = 'gen' 
    model2_corpus = 'spam' 
    
    # Classify each file
    count1 = 0
    count2 = 0
    gen_count = 0
    spam_count = 0
    gen_errors = 0
    spam_errors = 0
    
    for file in args.test_files:
        # Determine true class from file path
        file_str = str(file)
        true_class = 'gen' if '/gen/' in file_str or '\\gen\\' in file_str else 'spam'
        
        # Compute log probabilities
        log_prob1 = file_log_prob(file, lm1)
        log_prob2 = file_log_prob(file, lm2)
        
        # Add log priors (Bayes' Theorem)
        log_posterior1 = log_prob1 + math.log(args.prior)
        log_posterior2 = log_prob2 + math.log(1 - args.prior)
        
        # Classify by MAP (maximum a posteriori)
        if log_posterior1 > log_posterior2:
            predicted_corpus = model1_corpus
            print(f"{model1_name} {file}")
            count1 += 1
        else:
            predicted_corpus = model2_corpus
            print(f"{model2_name} {file}")
            count2 += 1
        
        # Track errors
        if true_class == 'gen':
            gen_count += 1
            if predicted_corpus != 'gen':
                gen_errors += 1
        else:
            spam_count += 1
            if predicted_corpus != 'spam':
                spam_errors += 1
    
    # Print summary
    total = count1 + count2
    if total > 0:
        pct1 = 100.0 * count1 / total
        pct2 = 100.0 * count2 / total
        print(f"{count1} files were more probably from {model1_name} ({pct1:.2f}%)")
        print(f"{count2} files were more probably from {model2_name} ({pct2:.2f}%)")
    
    # Print error analysis
    total_errors = gen_errors + spam_errors
    total_files = gen_count + spam_count
    if total_files > 0:
        error_rate = 100.0 * total_errors / total_files
        print(f"\nError Analysis:")
        print(f"Gen files misclassified as spam: {gen_errors}/{gen_count}")
        print(f"Spam files misclassified as gen: {spam_errors}/{spam_count}")
        print(f"Total errors: {total_errors}/{total_files}")
        print(f"Error rate: {error_rate:.2f}%")


if __name__ == "__main__":
    main()