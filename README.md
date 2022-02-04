# Meta-Embedding-Framework

## Problem Definition
Word-level meta-embedding learning consider the problem of learning a single embedding (meta-embedding) from a given set of pretrained input embeddings (source embeddings). In particular, in meta-embedding learning we 
(a) do *not* assume the availability of the training resources that were used to train the source emebddings (such resources might not be publicly available due to copyright or licensing restrictions) and
(b) do *not* retrain the source embeddings (which might be computationally costly to do so, and impossible if the training resources are not made available).
In particular, meta-embedding learning methods must work with source embeddings with *different* input dimensionalities in general.

This framework contains implementations of various exising word-level meta-embedding learning methods such as
- Concatenation (CONC)
- Averaging (AVG)
- Singular Value Decomposition of the concatenated source embeddings (SVD)
- Globally Linear Meta-Embeddings (Yin and Schutze, 2016)
- Locally Linear Meta-Embeddings (Bollegala et al., 2018)
and external support to autoencoder-based meta-embeddings (AEME)[https://github.com/LivNLP/AEME]

Moreover, the framework provides a single evaluation interface to meta word embeddings on a broad range of
tasks such as word similarity, word analogy, part-of-speech tagging, relation classification, sentiment classification,
psycholinguistic score prediction and sentence textual similarity (STS) evaluations.
For STS evaluations this framework relies on (SentEval)[https://github.com/facebookresearch/SentEval] and for all other word-level evaluations it uses (RepsEval)[https://github.com/Bollegala/repseval]
