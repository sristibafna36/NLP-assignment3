#file for dealing with the graphing for questions 3f and 3h

import matplotlib.pyplot as plt
import re
from pathlib import Path
import math
import os
from probs import LanguageModel, num_tokens
from fileprob import file_log_prob

gen = LanguageModel.load('gen0.005.model')
spam = LanguageModel.load('spam0.005.model')

length = []
cross_entropy = []

for file in list(Path('../data/gen_spam/dev/gen/').iterdir()) + list(Path('../data/gen_spam/dev/spam/').iterdir()):
    if 'gen' in str(file):
        model = gen
    else:
        model = spam
    length_val = int(re.search(r'\.(\d+)\.', file.name).group(1))
    cross_entropy_val = -file_log_prob(file, model) / (num_tokens(file) * math.log(2))
    length.append(length_val)
    cross_entropy.append(cross_entropy_val)

    plt.scatter(length, cross_entropy, alpha=0.6)
    plt.xlabel('Document Length (words)')
    plt.ylabel('Cross-entropy (bits/token)')
    plt.title('Performance vs Document Length (λ=0.005)')
    plt.savefig('length_graph.png', dpi=150)