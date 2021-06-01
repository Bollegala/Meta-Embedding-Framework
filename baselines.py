"""
Implement baseline methods such as averaging and SVD.
"""

import argparse
import numpy as np
import time
import sys
from wordreps import WordReps
from sklearn.utils.extmath import randomized_svd

def save_model(embed_matrix, ix_to_word, fname):
    """
    Write the word embeddings in the embed_matrix to fname.
    """
    with open(fname, 'w') as F:
        F.write("%d\t%d\n" % embed_matrix.shape)
        for i in range(embed_matrix.shape[0]):
            F.write("%s %s\n" % (ix_to_word[i], " ".join([str(x) for x in embed_matrix[i,:]])))

def command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", nargs="+", help="source embedding files")
    #parser.add_argument("-d", nargs="+", type=int, help="dimensionalities of the source embeddings [comma separated and ordered in the files specificed by option -i")
    parser.add_argument("-o", type=str, help="output file name.")
    parser.add_argument("-k", type=int, help="Dimensionality for SVD")
    parser.add_argument("-m", choices=['svd', 'avg'], required=True, help="use svd or avg respectively to perform SVD on the concatenated sources or to use their average")
    args = parser.parse_args()
    sources, word_to_ix, ix_to_word = load_source_embeddings(args.i)
    if args.m.lower() == "avg":
        ME = avg_baseline(sources, word_to_ix)
    elif args.m.lower() == "svd":
        ME = svd_baseline(sources, word_to_ix, args.k)
    else:
        raise "Invalid argument %s" % args.m
    save_model(ME, ix_to_word, args.o)
    pass

def svd_baseline(sources, word_to_ix, k):
    """
    Concatenate all sources and apply SVD to reduce dimensionality to k.
    """
    M = np.concatenate(sources, axis=1)
    print("concatenated M = ", M.shape)
    U, A, VT = randomized_svd(M, n_components=k, random_state=None)
    #print(U.shape)
    #print(A.shape)
    #print(VT.shape)
    return U[:,:k] @ np.diag(A[:k])

def avg_baseline(sources, word_to_ix):
    """
    Average the source embeddings.
    """
    n_max = np.max([s.shape[1] for s in sources])
    M = np.zeros((len(word_to_ix), n_max))
    for s in sources:
        M += np.pad(s, ((0,0),(0, n_max - s.shape[1])), 'constant')
    return M

def load_source_embeddings(sources):
    """
    Load all source embeddings and compute the union of all vocabularies.
    """
    embeddings = []
    vocab = set()
    for embd_fname in sources:
        start_time = time.process_time()
        sys.stdout.write("Loading %s ..." % embd_fname)
        sys.stdout.flush()        
        WR = WordReps()
        WR.read_model(embd_fname)
        end_time = time.process_time()
        sys.stdout.write("\nDone. took %s seconds\n" % str(end_time - start_time))
        sys.stdout.flush()
        embeddings.append(WR)
        vocab = vocab.union(WR.vocab)
    
    # Align all sources in the same vocabulary. Add zero embeddings for the missing sources
    sources = []
    vocab = list(vocab)
    word_to_ix = {}
    for word in vocab:
        word_to_ix[word] = len(word_to_ix)
    ix_to_word = {value:key for (key,value) in word_to_ix.items()}

    for emb in embeddings:
        M = np.zeros((len(vocab), emb.dim))
        for word in emb.vects:
            M[word_to_ix[word],:] = emb.vects[word]
        sources.append(M)
    return sources, word_to_ix, ix_to_word

if __name__ == "__main__":
    command_line()