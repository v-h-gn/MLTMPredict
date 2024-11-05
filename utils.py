import numpy as np
import pandas as pd
import Bio as bio
from Bio.SeqUtils import MeltingTemp as mt

def Tm_molar(seq, dna_conc1, dna_conc2, Na_conc, salt_corr) -> float:
    """
    Wrapper method for predicting the nearest neighbors melting temperature of a DNA sequence when concentrations are given in molar units.  
    seq: DNA sequence
    dna_conc1: concentration of the first DNA strand in molar units
    dna_conc2: concentration of the second DNA strand in molar units
    Na_conc: concentration of Na+ ions in molar units
    salt_corr: integer value which selects salt-correction method (see Bio.SeqUtils.MeltingTemp for details)
    """
    return mt.Tm_NN(seq, dnac1=dna_conc1*(10**9), dnac2=dna_conc2*(10**9), Na=Na_conc*(10**6), saltcorr=salt_corr)

def gen_seqs(length) -> 'list[str]':
    """
    Generates a list of all DNA sequences of a given length

    For lengths greater than 15, defaults to generating 4**10 random sequences of the given length using gen_rand_seqs.

    length: length of the DNA sequences

    """
    if length >= 15:
        return gen_rand_seqs(length, 4**10)
    else:
        return [np.base_repr(i, base=4).zfill(length).replace('0','A').replace('1','T').replace('2','G').replace('3','C') for i in range(4**length)]

def gen_rand_seqs(length, amount) -> 'list[str]':
    """
    Generates a list of k random DNA sequences of a given length
    length: length of the DNA sequences
    amount: number of sequences to generate
    """
    return [''.join(np.random.choice(['A','T','G','C'], length)) for i in range(amount)]

def gen_rand_seqs_df(length, amount) -> 'pd.DataFrame':
    """
    Generates a dataframe of k random DNA sequences of a given length
    length: length of the DNA sequences
    amount: number of sequences to generate
    """
    return pd.DataFrame(gen_rand_seqs(length, amount), columns=['seq'])

def gen_rand_seqs_df_Tm(length, amount, dna_conc1, dna_conc2, Na_conc, salt_corr) -> 'pd.DataFrame':
    """
    Generates a dataframe of k random DNA sequences of a given length with their nearest neighbors melting temperature
    length: length of the DNA sequences
    amount: number of sequences to generate
    dna_conc1: concentration of the first DNA strand in molar units
    dna_conc2: concentration of the second DNA strand in molar units
    Na_conc: concentration of Na+ ions in molar units
    salt_corr: integer value which selects salt-correction method (see Bio.SeqUtils.MeltingTemp for details)
    """
    df = gen_rand_seqs_df(length, amount)
    df['Tm'] = df['seq'].apply(lambda x: Tm_molar(x, dna_conc1, dna_conc2, Na_conc, salt_corr))
    return df

def num_to_dna(seq) -> str:
    """
    Converts a string of numbers to a string of DNA bases
    sequence: string of numbers
    """
    return seq.replace("0", "A").replace("1", "T").replace("2", "C").replace("3", "G")

