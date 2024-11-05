from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

class TmPredictor():

    def predict():
        raise NotImplementedError

class TmRegressor(TmPredictor):

    def __init__(self, model=None, params=None, train_x=None, train_y=None, test_x=None, test_y=None, cv_folds=RepeatedKFold(), scoring=['r2', 'neg_mean_squared_error'], random_state=42):
        """
        Wrapper class for a melting temperature predictor. Can be trained on a dataset and used to predict melting temperatures of DNA sequences.

        model: model to use for prediction (must implement Sci-Kit Learn API)
        params (dict): The hyperparameters for the XGBRegressor model.
        train_x (numpy.ndarray): The training input data.
        train_y (numpy.ndarray): The training output data.
        test_x (numpy.ndarray): The testing input data.
        test_y (numpy.ndarray): The testing output data.
        cv_folds (int | Iterator | BaseCrossValidator | BaseShuffleSplit): The cross-validation folds to use.
        scoring (list): The scoring metrics to use. (See Sci-Kit Learn API for valid metrics) [default: r2 and neg_mean_squared_error]
        """
        
        self.model = model
        self.params = params
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.trained = False
        self.random_state = random_state

    def predict(self, seqs) -> 'list[float]':
        """
        Predicts the melting temperature of a list of DNA sequences
        seqs: list of DNA sequences
        """
        if not self.trained:
            raise Exception("Model has not been trained!")
        return self.model.predict(seqs)
    
    def hyper_parameter_tune(self):
        """
        Performs hyperparameter tuning via cross validation on the model using the training data

        """
        if self.params is not None:
            param_tune = GridSearchCV(self.model, self.params, cv=self.cv_folds, scoring=self.scoring, refit='neg_mean_squared_error')
            param_tune.fit(self.train_x, self.train_y)
            self.model = param_tune.best_estimator_
            self.trained = True

    def train_and_score(self) -> 'tuple[TmPredictor, float, float]':
        """
        Train the model on the train set and score it on both sets
        """
        self.model.fit(self.train_x, self.train_y, eval_set=[(self.test_x, self.test_y)])

        train_score = self.model.score(self.train_x, self.train_y)
        test_score = self.model.score(self.test_x, self.test_y)

        self.trained = True

        return self, train_score, test_score

    
class XGBTmPredictor(TmPredictor):

    def __init__(self, model=None, params=None, train_x=None, train_y=None, test_x=None, test_y=None, cv_folds=RepeatedKFold(), scoring=['r2', 'neg_mean_squared_error'], random_state=42):
        """
        Initialize the model training class with the given parameters.

        model (XGBRegressor): The XGBRegressor model to use for training. If None, trains from scratch (See XGBoost API for details
        params (dict): The hyperparameters for the XGBRegressor model.
        train_x (numpy.ndarray): The training input data.
        train_y (numpy.ndarray): The training output data.
        test_x (numpy.ndarray): The testing input data.
        test_y (numpy.ndarray): The testing output data.
        cv_folds (int | Iterator | BaseCrossValidator | BaseShuffleSplit): The cross-validation folds to use.
        scoring (list): The scoring metrics to use. (See Sci-Kit Learn API for valid metrics)
        """
        if model is None:
            model = XGBRegressor(device="cuda", tree_method="exact", random_state=random_state)
            super().__init__(model, params, train_x, train_y, test_x, test_y, cv_folds, scoring)
            self.hyper_parameter_tune()
        else:
            super().__init__(model, params, train_x, train_y, test_x, test_y, cv_folds, scoring)
            self.trained = True

        
    def hyper_parameter_tune(self) -> None:
        """
        Performs hyperparameter tuning for the model using GridSearchCV.

        If self.params is None, default hyperparameter space is used for tuning:
        n_estimators: [100, 500, 1000]
        max_depth: [2, 4, 6, 8, 10]
        learning_rate: [0.01, 0.05, 0.1, 0.15, 0.2]
        min_child_weight: [1, 2, 4, 8]
        """
        if self.params is None:
            self.params['n_estimators'] = [100, 500, 1000]
            self.params['max_depth'] = [2, 4, 6, 8, 10]
            self.params['learning_rate'] = [0.01, 0.05, 0.1, 0.15, 0.2]
            self.params['min_child_weight'] = [1,2,4,8]
        
        param_tune = GridSearchCV(self.model, self.params, cv=self.cv_folds, scoring=self.scoring, refit='neg_mean_squared_error')

        param_tune.fit(self.train_x, self.train_y)
        self.model.set_params(**param_tune.best_params_)
    

    


