"""
Danushka Bollegala.
Locally Linear Meta Embedding Learning.
"""

import numpy

from sklearn import manifold
from sklearn.utils.extmath import randomized_svd
from sklearn.neighbors import NearestNeighbors


from wordreps import WordReps
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg.eigen.arpack.arpack import ArpackNoConvergence

from scipy.io import savemat, loadmat

import sys
import time
import argparse
import collections
from itertools import combinations


class MetaEmbed():

    def __init__(self, wordreps, words):
        self.words = words
        self.ids = {}
        for (i,word) in enumerate(words):
            self.ids[word] = i

        self.reps = wordreps
        self.dims = [x.dim for x in self.reps]
        self.embeds = []
        N = len(words)

        # Create the source embedding matrices
        write("Creating source embedding matrices...")
        for i in range(len(self.reps)):
            M = numpy.zeros((self.reps[i].dim, N), dtype=numpy.float64)
            for j in range(N):
                M[:,j] = self.reps[i].vects[self.words[j]]
            self.embeds.append(M)
        write("done\n")
        pass

    def compute_neighbours(self, nns):
        """
        Compute the nearest neighbours for each embedding.
        """
        self.NNS = []
        for i in range(len(self.embeds)):
            start_time = time.process_time()
            write("Computing nearest neighbours for embedding no = %d ..." % i)
            nbrs = NearestNeighbors(n_neighbors=nns, algorithm='ball_tree').fit(self.embeds[i].T)
            distances, indices = nbrs.kneighbors(self.embeds[i].T)
            self.NNS.append(indices[:,1:])
            end_time = time.process_time()
            write("Done (%s sec.)\n" % str(end_time - start_time))
        pass

    def show_nns(self, word ,nns):
        """
        Print nearest neigbours for a word in different embeddings.
        """
        for i in range(len(self.embeds)):
            print("Showing nearest neighbours for = %s" % word)
            print("\nEmbedding no = %d" % i)
            for s in self.NNS[i][self.ids[word], :][:nns]:
                print(self.words[s])
        pass

    def compute_weights(self):
        """
        Computes the reconstruction weights.
        """
        start_time = time.process_time()
        T = 10  # no. of iterations.
        alpha = 0.01  # learning rate.
        N = len(self.words)
        self.W = numpy.zeros((N, N), dtype=numpy.float64)

        # initialise the weights.
        for i in range(N):
            nns = set()
            for j in range(len(self.embeds)):
                for x in self.NNS[j][i,:]:
                    nns.add(x)
            val = 1.0 / float(len(nns))
            for j in nns:
                self.W[i,j] = val

        # iterate
        for i in range(N):
            write("\x1b[2K\rLearning weights for (%d of %d) = %s" % (i, N, self.words[i]))
            for t in range(T):                       
                d = [self.embeds[j][:,i] - numpy.sum([self.W[i,k] * self.embeds[j][:,k] for k in self.NNS[j][i,:]], axis=0) for j in range(len(self.embeds))]
                #for j in range(len(self.embeds)):
                #    d.append(self.embeds[j][:,i] - numpy.sum([self.W[i,k] * self.embeds[j][:,k] for k in self.NNS[j][i,:]], axis=0))
                
                grad = numpy.zeros(N, dtype=numpy.float64)
                for j in range(len(self.embeds)):
                    for k in self.NNS[j][i,:]:
                        grad[k] += -2.0 * numpy.dot(d[j], self.embeds[j][:,k])
        
                self.W[i,:] -= (alpha * grad)
        
            total = numpy.sum(self.W[i,:])
            if total != 0:
                self.W[i,:] = self.W[i,:] / total
        write("\n")
        end_time = time.process_time()
        write("Done (took %s seconds)\n" % str(end_time - start_time))
        pass

    def save_weights(self, fname):
        """
        Save the weight matrix to a disk file.
        """
        savemat(fname, {"W":self.W})
        pass

    def load_weights(self, fname):
        """
        Load the weight matrix from a disk file.
        """
        self.W = loadmat(fname)["W"]
        pass

    def test_compute_weights(self):
        """
        Check whether the weights are computed correctly
        """
        N = len(self.words)
        # Check whether non-neighbours have weights equal to zero.
        write("Checking whether non-neighbours have zero weights...\n")
        for i in range(N):
            pred_nns = set(numpy.where(self.W[i,:] != 0)[0])
            nns = set()
            for j in range(len(self.embeds)):
                nns = nns.union(set(self.NNS[j][i,:]))
            assert(pred_nns == nns)

        # Check whether reconstruction weights add upto one.
        write("Checking whether weights add to 1...\n")
        for i in range(N):
            assert(numpy.allclose(numpy.sum(self.W[i,:]), 1)) 

        # print nearest neighbours and their weights
        nn_file = open("./work/nn.csv", 'w')
        for i in range(N):
            nn_file.write("%s, " % self.words[i])
            L = []
            for j in range(N):
                if self.W[i,j] != 0:
                    L.append((self.words[j], self.W[i,j]))
            L.sort(lambda x, y: -1 if x[1] > y[1] else 1)
            for (w, val) in L:
                nn_file.write("%s, %f, " % (w, val))
            nn_file.write("\n")
        nn_file.close()
        pass

    def compute_M(self):
        """
        Compute the smallest eigenvectors of M = (I - W')\T(I - W').
        """
        # Building W'
        N = len(self.words)
        start_time = time.process_time()
        write("Computing W'...")
        for i in range(N):
            z = numpy.zeros(N)
            write("Completed %d of %d\r" % (i, N))
            for nns in self.NNS:
                z[nns[i,:]] += 1
            self.W[i,:] = z * self.W[i,:]
        end_time = time.process_time()
        write("Done (took %s seconds)\n" % str(end_time - start_time))

        # Computing M.
        start_time = time.process_time()
        write("Computing M....")
        self.W = csr_matrix(self.W)
        M = eye(N, format=self.W.format) - self.W
        M = (M.T * M).tocsr()
        end_time = time.process_time()
        write("Done (took %s seconds)\n" % str(end_time - start_time))
        return M

    def compute_embeddings(self, k, M, embed_fname):
        """
        Perform eigen decomposition.
        """
        N = len(self.words)
        start_time = time.process_time()
        write("Computing Eigen decomposition...")
        s, V =  eigsh(M, k+1, tol=1E-6, which="SA", maxiter=100)
        end_time = time.process_time()
        write("Done (took %s seconds)\n" % str(end_time - start_time))
        P = V[:, 1:]
        err = numpy.sum(s[1:])
        write("Projection error = %f\n" % err)
        
        write("Writing embeddings to file...")
        # Write embeddings to file.
        with open(embed_fname, 'w') as embed_file:
            for i in range(N):
                embed_file.write("%s %s\n" % (self.words[i], " ".join([str(x) for x in P[i,:]])))
        write("Done\n")
        pass


def write(msg):
    sys.stdout.write(msg)
    sys.stdout.flush()
    pass


def meta_embed(embeddings, words, nns, comps, embed_path):
    """
    Perform meta-embedding using LLE.
    """
    ME = MetaEmbed(embeddings, words)
    ME.compute_neighbours(nns)
    #ME.show_nns("king", 5)

    #ME.compute_weights_parallel()
    ME.compute_weights()

    #ME.save_weights("../work/weights_%d" % nns)
    #ME.load_weights("../work/weights+n=%d.meta" % nns)
    #ME.test_compute_weights()
    M = ME.compute_M()
    for k in comps:
        embed_fname = "%s_n=%d_k=%d" % (embed_path, nns, k)
        write("Embedding NNS = %d, Components (k) = %d\n" % (nns, k))
        try:
            ME.compute_embeddings(k, M, embed_fname)
        except ArpackNoConvergence as e:
            print(e)
    return ME


def baseline_concatenate(embeddings, words, embed_fname):
    """
    Concatenate embeddings to create co-embeddings.
    """
    dim = sum([x.dim for x in embeddings])

    print("Concatenation dimension =", dim)
    # concatenate the vectors.
    with open(embed_fname, 'w') as embed_file:
        for (i,word) in enumerate(words):
            L = []
            for x in embeddings:
                w = 8 if x.dim == 300 else 1
                #w = 1
                L.append(w * x.vects[word])
                
            z = numpy.concatenate(L)            
            embed_file.write("%s %s\n" % (word, " ".join([str(x) for x in z])))
    pass

def get_common_words(embeddings):
    words = set(embeddings[0].vocab)
    for i in range(1, len(embeddings)):
        words = words.intersection(set(embeddings[i].vocab))
    return words


def get_selected_words(fname):
    words = []
    with open(fname, encoding='utf-8', errors='ignore') as F:
        for line in F:
            words.append(line.strip())
    return words

def perform_embedding(sources, dims, nns, comps, output_fname):
    print("Neigbourhood size = %d" % nns)    
    embeddings = []
    for (embd_fname, dim) in zip(sources, dims):
        start_time = time.process_time()
        sys.stdout.write("Loading %s -- (%d dim) ..." % (embd_fname, dim))
        sys.stdout.flush()        
        WR = WordReps()
        WR.read_model(embd_fname, dim)
        end_time = time.process_time()
        sys.stdout.write("\nDone. took %s seconds\n" % str(end_time - start_time))
        sys.stdout.flush()
        embeddings.append(WR)

    common_words = get_common_words(embeddings)
    selected_words = get_selected_words("./data/selected-words")
    words = []
    for word in selected_words:
        if word in common_words and word not in words:
            words.append(word)
    print("No. of common words =", len(common_words))
    print("Vocabulary size =", len(words))
    ME = meta_embed(embeddings, words, nns, comps, output_fname)
    pass

def save_embedding(words, WR, fname):
    F = open(fname, 'w')
    F.write("%d %d\n" % (len(WR.vects), WR.dim))
    for w in words:
        if w in WR.vects:
            F.write("%s " % w)
            F.write("%s\n" % " ".join([str(x) for x in WR.vects[w]]))
        # elif w.lower() in WR.vects:
        #     F.write("%s " % w.lower())
        #     F.write("%s\n" % " ".join([str(x) for x in WR.vects[w.lower()]]))
    F.close()
    pass


def batch_process():
    """
    Create all combinations of meta embeddings from the sources.
    """
    sources_info = [("wiki", "wiki-news-300d-1M-subword.vec.selected", 300), 
                    ("glove", "glove.840B.300d.txt.selected", 300),
                    ("crawl", "crawl-300d-2M-subword.vec.selected", 300),
                    ("news", "GoogleNews-vectors-negative300.bin.selected", 300)]
    
    #sources_info = [("wiki", "wiki.small", 300), 
    #                ("glove", "glove.small", 300),
    #                ("crawl", "crawl.small", 300),
    #                ("news", "w2v.small", 300)]

    comps = [100] # dimensionality of the meta embedding
    nns = 1000  # number of nearest neighbours to consider in LLE
    
    # Load all sources
    embeddings = []
    for (embed_name, embd_fname, dim) in sources_info:
        start_time = time.process_time()
        sys.stdout.write("Loading %s -- (%d dim) ..." % (embd_fname, dim))
        sys.stdout.flush()        
        WR = WordReps()
        WR.read_model("./data/" + embd_fname, dim)
        end_time = time.process_time()
        sys.stdout.write("\nDone. took %s seconds\n" % str(end_time - start_time))
        sys.stdout.flush()
        embeddings.append((embed_name, WR))

    common_words = set(embeddings[0][1].vocab)
    for i in range(1, len(embeddings)):
        common_words = common_words.intersection(set(embeddings[i][1].vocab))
    selected_words = get_selected_words("./data/bias-selected-words")
    words = []
    for word in selected_words:
        if word in common_words and word not in words:
            words.append(word)
    print("No. of common words =", len(common_words))
    print("Vocabulary size =", len(words))    

    # Create all combinations of meta embeddings
    for no_sources in [2,3,4]:
        for  sources in combinations(embeddings, no_sources):
            output_fname = "./work/LLE/%s" % "+".join([x[0] for x in sources])
            print(output_fname)
            ME = meta_embed([x[1] for x in sources], words, nns, comps, output_fname)
    pass


def command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("-nns", type=int, help="number of nearest neighbours")
    parser.add_argument("-comps", nargs="+", type=int, help="components for the projection")
    parser.add_argument("-i", nargs="+", help="source embedding files")
    parser.add_argument("-d", nargs="+", type=int, help="dimensionalities of the source embeddings [comma separated and ordered in the files specificed by option -i")
    parser.add_argument("-o", type=str, help="output file name. If multiple comps are specified we will append the comp as a suffix")
    args = parser.parse_args()
    perform_embedding(args.i, args.d, args.nns, args.comps, args.o)
    pass


def merge_word_lists():
    freq_words = get_selected_words("data/selected-words")
    bias_words = get_selected_words("data/bias_vocab.txt")
    print(len(freq_words), len(bias_words))
    words = set(freq_words).union(set(bias_words))
    print(len(words))
    with open("data/bias-selected-words", 'w') as F:
        for w in words:
            F.write("{0}\n".format(w))
    pass


def load_embed_vocab(fname):
    print(fname)
    embed_vocab = []
    with open(fname) as F:
        for line in F:
            embed_vocab.append(line.lower().split()[0])
    return embed_vocab

def check_vocabs():
    """
    Check whether embeds contain words from the vocab.
    """
    lle_vocab = load_embed_vocab("work/LLE/wiki+glove+crawl+news_n=1000_k=300")
    print("LLE =", len(lle_vocab))

    wiki_vocab = load_embed_vocab("data/wiki-news-300d-1M-subword.vec.selected")
    print("wiki =", len(wiki_vocab))

    glove_vocab = load_embed_vocab("data/glove.840B.300d.txt.selected")
    print("glove =", len(glove_vocab))

    crawl_vocab = load_embed_vocab("data/crawl-300d-2M-subword.vec.selected")
    print("crawl =", len(crawl_vocab))

    news_vocab = load_embed_vocab("data/GoogleNews-vectors-negative300.bin.selected")
    print("news =", len(news_vocab))

    vocab = set(wiki_vocab)
    vocab = vocab.intersection(set(glove_vocab))
    vocab = vocab.intersection(set(crawl_vocab))
    vocab = vocab.intersection(set(news_vocab))

    print("Common =", len(vocab))

    bias_vocab = []
    with open("data/bias_vocab.txt") as bias_file:
        for line in bias_file:
            bias_vocab.append(line.lower().strip())

    missing_bias_words = []
    for word in bias_vocab:
        if word not in vocab:
            missing_bias_words.append(word)
            print(word)
            
    
    print("Total no. of words missing from bias words =", len(missing_bias_words))



if __name__ == '__main__':
    #command_line()
    #batch_process()
    #merge_word_lists()
    check_vocabs()
    pass
