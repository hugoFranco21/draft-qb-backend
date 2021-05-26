from sklearn.preprocessing import MinMaxScaler
import numpy as np
import json

min_max_scaler_1 = MinMaxScaler()
min_max_scaler_1.scale_ = [0.2, 0.07633588, 0.06060606, 0.24390244, 0.01510574]
coef_rookies = [1.71814446, -4.7037779,  25.32033643,  3.60841173, -6.91661286]

min_max_scaler_2 = MinMaxScaler()
min_max_scaler_2.scale_ = [0.01416431, 0.00301205, 0.31197354, 0.00373134, 0.36425892, 0.04166667]

def h(params, sample):
	"""This evaluates a generic linear function h(x) with current parameters.  h stands for hypothesis

	Args:
		params (lst) a list containing the corresponding parameter for each element x of the sample
		sample (lst) a list containing the values of a sample 

	Returns:
		Evaluation of h(x)
	"""
	acum = 0
	for i in range(len(params)):
		acum = acum + params[i]*sample[i]  #evaluates h(x) = a+bx1+cx2+ ... nxn.. 
	return acum

def get_rookie_prediction(input):
    """This function receives the data from the requests, parses it as a dictionary and gets the prediction

    Args:
        input (json) The request body
    
    Returns:
        The prediction for the input value
    """
    data = list(input.values())
    an_array = np.array(data)
    an_array= an_array.reshape(-1,1)
    norm = min_max_scaler_1.transform(an_array)
    return h(coef_rookies, norm)

