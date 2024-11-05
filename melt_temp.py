import encoder as enc
import utils as utils
import prediction as pred
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts



class MeltTempPipeline:

    def __init__(self, dataset=None, encoder=None, predictor=None):
        """
        Wrapper class for a melting temperature prediction pipeline. 
        
        encoder: encoder to use for prediction
        predictor: predictor to use for prediction
        dataset (pandas.DataFrame):, there must exist a column named 'Sequence' which contains strings of the DNA sequences, there must also be a column named 'Experimental Temperature' which contains the experimental melting temperatures of the sequences
        """
        
        if dataset is None:
            print("No dataset provided, using default dataset")
            dataset = pd.read_csv("data/default_dataset.csv")

        if encoder is None:
            print("No encoder provided, using default encoder")
            encoder = enc.ProtVecEncoder(model="encoders/default_encoder")

        if predictor is None:
            print("No predictor provided, using default predictor")
            predictor = pred.XGBTmPredictor(model="models/default_predictor")

        self.dataset = dataset
        self.encoder = encoder
        self.predictor = predictor

    def predict(self):
        """
        Predicts the melting temperatures of the sequences in the dataset
        """
        self.dataset['Predicted Temperature'] = self.predictor.predict(self.encoder.multi_encode(self.dataset['Sequence']))
        return self.dataset
    
def violin_plots(experimental, predictions, labels):
    """
    Plots violin plots of the melting temperature predictions of the three models

    experimental: list of experimental melting temperatures
    predictions: list of lists of melting temperature predictions
    labels: list of labels for the models in the same order as the predictions

    Modified from code provided by Dr. William H. Grover
    """
    diffs = []
    for i in range(len(predictions)):
        diffs.append(np.subtract(predictions[i] - experimental))

    plt.boxplot(diffs, labels=labels)
    plt.violinplot(diffs)
    plt.axhline(0)
    plt.gca().set_ylabel("$T_m$ difference from experimental [C]")
    plt.savefig("box.png")


    
