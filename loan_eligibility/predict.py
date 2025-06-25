import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError

def predict_applicant(model, input_dict):
    input_array = np.array(list(input_dict.values())).reshape(1, -1)
    
    # Try predicting
    try:
        prediction = model.predict(input_array)[0]
        return prediction
    except NotFittedError:
        raise NotFittedError("Model is not trained. Cannot make predictions.")
