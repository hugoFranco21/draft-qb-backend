from sklearn.preprocessing import MinMaxScaler
import numpy as np
import json
import jsonschema
from jsonschema import validate
import numpy
import pickle
from sklearn.ensemble import RandomForestClassifier

min_max_scaler_1 = MinMaxScaler()
min_max_scaler_1.scale_ = [0.2, 0.07633588, 0.06060606, 0.24390244, 0.01510574]
min_max_scaler_1.min_ = [-0.2, -0.13740458,-3.26060606,-1.53658537, -1.73867069]
min_max_scaler_1.clip = False
coef_rookies = [1.71814446, -4.7037779,  25.32033643,  3.60841173, -6.91661286]
intercept_1 = 64.67014634129038

min_max_scaler_2 = MinMaxScaler()
min_max_scaler_2.scale_ = [0.01416431, 0.00301205, 0.31197354, 0.00373134, 0.36425892, 0.04166667]
min_max_scaler_2.min_ = [-0.73654391, -2.5873494,  -1.17673301, -3.29104478, -1.41926201, -0.08333333]
min_max_scaler_2.clip = False
coef_w_1 = [11.526699240171151, 5.720350561790972, 3.530300996067043, 3.100404856127282, -7.191579327156075, 0.9559021636746066]
coef_w_2 = [0.000663071675071019, -3.34425460659029e-06, -0.0007341352696179361, 4.880640027024546e-06, 6.54479671678466e-05, 3.0522765234519485e-05]
coef_w_3 = [0.0006476834459069197, -3.267055848933503e-06, -0.0007170978143550761, 4.7668964161111525e-06, 6.39296947635263e-05, 2.9814685214290547e-05]
coef_w_4 = [0.0006326523416857217, -3.1916168325412085e-06, -0.0007004557575324727, 4.655832751163434e-06, 6.244660514369526e-05, 2.912301665079273e-05]

forest = None

with open('forest.dat', 'rb') as f:
    forest = pickle.load(f)

script_schema_1 = {
    "type": "object",
    "properties": {
        "draft": {"type": "number"},
        "sack": {"type": "number"},
        "completion": {"type": "number"},
        "yards": {"type": "number"},
        "efficiency": {"type": "number"},
    },
}

script_schema_2 = {
    "type": "object",
    "properties": {
        "rating": {"type": "number"},
        "offplay": {"type": "number"},
        "offyard": {"type": "number"},
        "defplay": {"type": "number"},
        "defyard": {"type": "number"},
        "turnovers": {"type": "number"},
    },
}

script_schema_3 = {
    "type": "object",
    "properties": {
        "wins": {"type": "number"},
        "completion": {"type": "number"},
        "yards": {"type": "number"},
        "tds": {"type": "number"},
        "ints": {"type": "number"},
        "rating": {"type": "number"},
        "years": {"type": "number"},
        "age": {"type": "number"},
    },
}

def validate_input(array):
    for x in array:
        if(x < 0):
            return False
    return True

def h(params, sample):
    """
    This evaluates a generic linear function h(x) with current parameters.  h stands for hypothesis
	Args:
		params (lst) a list containing the corresponding parameter for each element x of the sample
		sample (lst) a list containing the values of a sample 

	Returns:
		Evaluation of h(x)
	"""
    acum = 0
    if any(isinstance(i, numpy.ndarray or list) for i in sample):
        sample = sample[0]
    for i in range(len(params)):
        acum = acum + params[i]*sample[i]  #evaluates h(x) = a+bx1+cx2+ ... nxn.. 
    return acum

def validateJson(jsonData, schema):
    try:
        validate(instance=jsonData, schema=schema)
    except jsonschema.exceptions.ValidationError as err:
        return False
    return True

def get_rookie_prediction(input):
    """This function receives the data from the requests, parses it as a dictionary and gets the prediction

    Args:
        input (json) The request body
    
    Returns:
        The prediction for the input value
    """
    try:
        data = json.loads(input)
    except ValueError as err:
        return -1
    if not (validateJson(data, script_schema_1)):
        return -1
    data = json.loads(input)
    an_array = np.array(list(data.values()))
    if not validate_input(an_array):
        return -1
    norm = min_max_scaler_1.transform([an_array])
    pred = h(coef_rookies, norm) + intercept_1
    return pred

def get_wins_prediction(input):
    """This function receives the data from the requests, parses it as a dictionary and gets the prediction

    Args:
        input (json) The request body
    
    Returns:
        The prediction for the input value
    """
    try:
        data = json.loads(input)
    except ValueError as err:
        return -1
    if not (validateJson(data, script_schema_2)):
        return -1
    an_array = np.array(list(data.values()))
    if not validate_input(an_array):
        return -1
    norm = min_max_scaler_2.transform([an_array])
    pred = h(coef_w_1, norm) + h(coef_w_2, norm) + h(coef_w_3, norm) + h(coef_w_4, norm)
    return pred

def get_should_draft(input):
    """This function receives the data from the requests, parses it as a dictionary and gets the prediction

    Args:
        input (json) The request body
    
    Returns:
        The prediction for the input value
    """
    try:
        data = json.loads(input)
    except ValueError as err:
        return None
    if not (validateJson(data, script_schema_3)):
        return None
    an_array = np.array(list(data.values()))
    if not validate_input(an_array):
        return None
    norm = forest.predict([an_array])
    pred = norm[0]
    return pred

