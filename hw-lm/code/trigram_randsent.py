#!/usr/bin/env python3
"""
Sample random sentences from a trained language model.
"""
import argparse
import logging
from pathlib import Path
import torch

from probs import LanguageModel, BOS, EOS

log = logging.getLogger(Path(__file__).stem)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        type=Path,
        help="path to the trained model",
    )
    parser.add_argument(
        "num_sentences",
        type=int,
        help="number of sentences to generate",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=20,
        help="maximum length of generated sentences",
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


def sample_sentence(lm: LanguageModel, max_length: int) -> list:
    """Sample one sentence from the language model."""
    x, y = BOS, BOS
    sentence = []
    
    for _ in range(max_length):
        # Get probabilities for all possible next words
        probs = []
        words = []
        for z in lm.vocab:
            prob = lm.prob(x, y, z) if hasattr(lm, 'prob') else torch.exp(lm.log_prob(x, y, z))
            probs.append(prob)
            words.append(z)
        
        # Convert to tensor and sample
        probs_tensor = torch.tensor(probs)
        idx = torch.multinomial(probs_tensor, 1).item()
        z = words[idx]
        
        if z == EOS:
            break
        
        sentence.append(z)
        x, y = y, z
    
    # Truncate if too long
    if len(sentence) >= max_length:
        sentence.append("...")
    
    return sentence


def main():
    args = parse_args()
    logging.basicConfig(level=args.logging_level)
    
    if args.device == 'mps':
        if not torch.backends.mps.is_available():
            logging.critical("MPS not available")
            exit(1)
    torch.set_default_device(args.device)
    
    log.info("Loading model...")
    lm = LanguageModel.load(args.model, device=args.device)
    
    log.info(f"Sampling {args.num_sentences} sentences...")
    for i in range(args.num_sentences):
        sentence = sample_sentence(lm, args.max_length)
        print(" ".join(sentence))


if __name__ == "__main__":
    main()