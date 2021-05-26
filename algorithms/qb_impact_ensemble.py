import numpy  #numpy is used to make some operrations with arrays more easily
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

__errors__= [];  #global variable to store the errors/loss for visualisation

renamed = {
    'W': 'Wins',
	'Sk': 'Sacks',
    'Rate': 'QB Rating',
    'Ply':  'Offensive plays',
    'Y/P': 'Yards per play', 
    'DPly': 'Defensive plays',
    'DY/P': 'D Yards per play',
	'NQTO': 'Non QB Turnovers'
}

def prepare_data():
	df = pd.read_excel('../datasets/qb_impact.xlsx', header=1, usecols=[4,11, 13, 14, 15, 16, 18, 19, 20, 21, 22])
	df['NQTO'] = df['TO']-df['Int']
	df = df[['W','Sk', 'Rate', 'Ply', 'Y/P', 'DPly', 'DY/P', 'NQTO']]
	df.rename(columns = renamed, inplace = True)
	return df


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


def show_errors(params, samples,y):
	"""Appends the errors/loss that are generated by the estimated values of h and the real value y
	
	Args:
		params (lst) a list containing the corresponding parameter for each element x of the sample
		samples (lst) a 2 dimensional list containing the input samples 
		y (lst) a list containing the corresponding real result for each sample
	
	"""
	error_acum =0
#	print("transposed samples") 
#	print(samples)
	for i in range(len(samples)):
		hyp = h(params,samples[i])
		#print( "hyp  %f  y %f " % (hyp,  y[i]))   
		error=hyp-y[i]
		error_acum=+error**2 # this error is the original cost function, (the one used to make updates in GD is the derivated verssion of this formula)
	mean_error_param=error_acum/len(samples)
	__errors__.append(mean_error_param)

def GD(params, samples, y, alfa):
	"""
	Gradient Descent algorithm 
	Args:
		params (lst) a list containing the corresponding parameter for each element x of the sample
		samples (lst) a 2 dimensional list containing the input samples 
		y (lst) a list containing the corresponding real result for each sample
		alfa(float) the learning rate
	Returns:
		temp(lst) a list with the new values for the parameters after 1 run of the sample set
	"""
	temp = list(params)
	general_error=0
	for j in range(len(params)):
		acum =0; error_acum=0
		for i in range(len(samples)):
			error = h(params,samples[i]) - y[i]
			acum = acum + error*samples[i][j]  #Sumatory part of the Gradient Descent formula for linear Regression.
		temp[j] = params[j] - alfa*(1/len(samples))*acum  #Subtraction of original parameter value with learning rate included.
	return temp

def main():
	df = prepare_data()
	df_y = df['Wins']
	df_x = df[['QB Rating',
    'Offensive plays',
    'Yards per play',
	'Defensive plays',
    'D Yards per play',
	'Non QB Turnovers']]
	X = df_x.to_numpy()
	y = df_y.to_numpy()
	min_max_scaler = MinMaxScaler()
	data_minmax = min_max_scaler.fit_transform(X)
	params1 = [0,0,0,0,0,0]
	alfa = 0.1
	epochs = 0
	while True:  #  run gradient descent until local minima is reached
		oldparams = list(params1)
		print (params1)
		params1=GD(params1, data_minmax,y,alfa)	
		show_errors(params1, data_minmax, y)  #only used to show errors,it is not used in calculation
		print (params1)
		epochs = epochs + 1
		if(oldparams == params1 or epochs == 10000):   #  local minima is found when there is no further improvement
			print ("samples:")
			print(data_minmax)
			print ("final params:")
			print (params1)
			break
	y2 = []
	for index, x in enumerate(data_minmax):
		aux = y[index] - h(params1, x)
		y2.append(aux)
	epochs = 0
	params2 = [0,0,0,0,0,0]
	alfa = 0.01
	while True:  #  run gradient descent until local minima is reached
		oldparams = list(params2)
		print (params2)
		params2=GD(params2, data_minmax,y2,alfa)	
		show_errors(params2, data_minmax, y2)  #only used to show errors,it is not used in calculation
		print (params2)
		epochs = epochs + 1
		if(oldparams == params2 or epochs == 500):   #  local minima is found when there is no further improvement
			print ("samples:")
			print(data_minmax)
			print ("final params:")
			print (params2)
			break
	y3 = []
	for index, x in enumerate(data_minmax):
		aux = y2[index] - h(params2, x)
		y3.append(aux)
	epochs = 0
	params3 = [0,0,0,0,0,0]
	while True:  #  run gradient descent until local minima is reached
		oldparams = list(params3)
		print (params3)
		params3=GD(params3, data_minmax,y3,alfa)	
		show_errors(params3, data_minmax, y3)  #only used to show errors,it is not used in calculation
		print (params3)
		epochs = epochs + 1
		if(oldparams == params3 or epochs == 500):   #  local minima is found when there is no further improvement
			print ("samples:")
			print(data_minmax)
			print ("final params:")
			print (params3)
			break
	y4 = []
	for index, x in enumerate(data_minmax):
		aux = y3[index] - h(params3, x)
		y4.append(aux)
	epochs = 0
	params4 = [0,0,0,0,0,0]
	while True:  #  run gradient descent until local minima is reached
		oldparams = list(params4)
		print (params4)
		params4=GD(params4, data_minmax,y4,alfa)	
		show_errors(params4, data_minmax, y4)  #only used to show errors,it is not used in calculation
		print (params4)
		epochs = epochs + 1
		if(oldparams == params4 or epochs == 500):   #  local minima is found when there is no further improvement
			print ("samples:")
			print(data_minmax)
			print ("final params:")
			print (params3)
			break
	y_pred = []
	for index, x in enumerate(data_minmax):
		aux = h(params1, x) + h(params2, x) + h(params3, x) + h(params4,x)
		y_pred.append(aux)
	# The coefficients
	f = open("../output/script2_ensemble.txt", "w")
	f.write("Output of the script 2 (ensemble version)\n")
	f.write('\nscale ' + str(min_max_scaler.scale_))
	# The coefficients
	f.write('\nCoefficients 1: \n' + str(params1))
	# The coefficients
	f.write('\nCoefficients 2: \n' + str(params2))
	# The coefficients
	f.write('\nCoefficients 3: \n' + str(params3))

	f.write('\nCoefficients 4: \n' + str(params4))
# The mean squared error
	f.write('\nMean squared error: %.2f'
		% mean_squared_error(y, y_pred))
# The coefficient of determination: 1 is perfect prediction
	f.write('\nCoefficient of determination: %.2f'
		% r2_score(y, y_pred))

#use this to generate a graph of the errors/loss so we can see whats going on (diagnostics)

main()

plt.plot(__errors__)
plt.savefig('../assets/error2.png')