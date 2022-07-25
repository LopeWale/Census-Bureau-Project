from sklearn.metrics import fbeta_score, precision_score, recall_score
from pycaret.classification import *
from train_model import X_test, X_train, y_train, y_test, cat_features
from process_data import X, y, encoder, lb

def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    # Add code to load in the data.
    setup(data=X_train, target=y_train, log_experiment=True, experiment_name="Census_Application")

    # Train and  compare all models from the pycaret classifier modules
    import yaml

    best = compare_models()
    with open("./model/compare_models.yaml", "x") as file:
        yaml.dump(best, file)
        file.close()

    #save model
    # finalize the model
    final_best = finalize_model(best)
    # save model to disk
    save_model(final_best, './model/census_clf')

    #load model
    model = load_model('model/census_clf.pkl')
    return model


def inference(model, X_test):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X_test : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    # generate predictions
    from pycaret.classification import predict_model
    preds = predict_model(model, data=X_test)
    return preds


def compute_model_metrics(y_test, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y_test, preds, beta=1, zero_division=1)
    precision = precision_score(y_test, preds, zero_division=1)
    recall = recall_score(y_test, preds, zero_division=1)
    return precision, recall, fbeta
