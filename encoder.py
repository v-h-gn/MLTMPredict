import numpy as np
import pandas as pd
import biovec as bv

class DNAEncoder:

    def encode(self, sequence="AAAAAAA") -> np.ndarray:
        return int(sequence.replace("A", 0).replace("T", 1).replace("G", 2).replace("C", 3), 4)

    def multi_encode(self, sequences) -> 'list[np.ndarray]':
        return [self.encode(seq) for seq in sequences]



class ProtVecEncoder(DNAEncoder):
    def __init__(self, model=None, fasta_fname=None, corpus=None, ngram_size=2, vec_dim=16, corpus_fname="corpus.txt", sg=1, window=25, min_count=1, workers=3):
        """
        Creates an encoder that uses ProtVec to encode DNA sequences into vectors. This is mainly a wrapper class for some utilities and better documentation.

        You must either provide a model or a fasta_fname or corpus. If you provide a model, the other parameters are ignored.
        
        model: model filename to load ProtVec from if you already have one
        fasta_fname: fasta file for training new ProtVec
        corpus: gensim corpus file for training new ProtVec
        ngram_size: length of n-gram for splitting sequences for training new ProtVec
        vec_dim: dimension of the vector representation for training new ProtVec
        corpus_fname: file path for generated corpus if trained from fasta for training new ProtVec
        sg: 1 for skip-gram, 0 for CBOW for training new ProtVec
        min_count: least appearance count in corpus. if the n-gram appear k times which is below min_count, the model does not remember the n-gram 
        workers: number of workers for training new ProtVec
        """
        if model is not None:
            self.protvec = bv.models.load_protvec(model)
        else:
            if fasta_fname is None and corpus is None:
                raise Exception("Either fasta_fname or corpus is needed!")
            self.protvec = bv.models.ProtVec(fasta_fname=fasta_fname, corpus=corpus, n=ngram_size, size=vec_dim, corpus_fname=corpus_fname, sg=sg, window=window, min_count=min_count, workers=workers)

    def encode(self, seq) -> np.ndarray:
        """
        Encodes a single sequence into a vector of shape (3, vec_dim), then concatenates into (3*vec_dim, )

        seq: sequence to encode
        """
        return np.concatenate(self.protvec.to_vecs(seq))
    
    def save(self, fname) -> None:
        self.protvec.save("encoders/" + fname)