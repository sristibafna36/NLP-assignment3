# Question 8 - Speech Recognition

import argparse
import logging
import math
import re
from pathlib import Path
import torch

from probs import LanguageModel, BOS, EOS, read_tokens

log = logging.getLogger(Path(__file__).stem)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        type=Path,
        help="path to the trained language model",
    )
    parser.add_argument(
        "test_files",
        type=Path,
        nargs="*",
        help="utterance files to process"
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


def read_utterance_file(file: Path):
    """Read an utterance file and return candidates with their info."""
    candidates = []
    true_transcription = None
    
    with open(file) as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            
            if i == 0:
                # First line: true transcription (just store length for reference)
                num_words = int(parts[0])
                true_transcription = (num_words, parts[1:])
            else:
                # Candidate lines: word_error_rate acoustic_log_prob num_words transcription...
                wer = float(parts[0])
                acoustic_log_prob = float(parts[1])
                num_words = int(parts[2])
                transcription = parts[3:]  # All remaining words
                
                candidates.append({
                    'wer': wer,
                    'acoustic': acoustic_log_prob,
                    'num_words': num_words,
                    'transcription': transcription
                })
    
    return true_transcription, candidates


def compute_lm_log_prob(transcription: list, lm: LanguageModel) -> float:
    """Compute language model log probability for a transcription."""
    # Add BOS context and EOS at end
    x, y = BOS, BOS
    log_prob = 0.0
    
    for word in transcription:
        # Map to vocab (handle OOV)
        if word in lm.vocab:
            z = word
        else:
            z = "OOV"
        
        log_prob += lm.log_prob(x, y, z)
        x, y = y, z
    
    # Add EOS
    log_prob += lm.log_prob(x, y, EOS)
    
    return log_prob


def main():
    args = parse_args()
    logging.basicConfig(level=args.logging_level)
    
    if args.device == 'mps':
        if not torch.backends.mps.is_available():
            logging.critical("MPS not available")
            exit(1)
    torch.set_default_device(args.device)
    
    # Load language model
    log.info("Loading language model...")
    lm = LanguageModel.load(args.model, device=args.device)
    
    total_errors = 0
    total_words = 0
    
    for file in args.test_files:
        true_trans, candidates = read_utterance_file(file)
        
        # Skip if file couldn't be parsed
        if true_trans is None or not candidates:
            log.warning(f"Skipping malformed file: {file}")
            continue
            
        true_num_words = true_trans[0]
        
        # Find best candidate using Bayes' Theorem
        best_score = -math.inf
        best_candidate = None
        
        for cand in candidates:
            # Compute language model log prob (in natural log)
            lm_log_prob = compute_lm_log_prob(cand['transcription'], lm)
            
            # Convert to log base 2 for consistency with acoustic scores
            lm_log_prob_base2 = lm_log_prob / math.log(2)
            
            # Total score: log p(u|w) + log p(w)
            score = cand['acoustic'] + lm_log_prob_base2
            
            if score > best_score:
                best_score = score
                best_candidate = cand
        
        if best_candidate is None:
            log.warning(f"No valid candidates in file: {file}")
            continue
        
        # Report WER for this file
        wer = best_candidate['wer']
        print(f"{wer:.3f}\t{file.name}")
        
        # Accumulate errors
        num_errors = int(wer * true_num_words)
        total_errors += num_errors
        total_words += true_num_words
    
    # Report overall WER
    overall_wer = total_errors / total_words if total_words > 0 else 0
    print(f"{overall_wer:.3f}\tOVERALL")


if __name__ == "__main__":
    main()