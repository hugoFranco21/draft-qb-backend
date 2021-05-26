import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

renamed = {
    'Heisman': 'Heisman',
    'Pct': 'Completion Percentage',
    'Y/A.1':  'Yards per Attempt',
    'Rate.1': 'Efficiency Rating', 
    'Rate': 'QB Rating',
    'Sk%': 'Sacked %'
}

columns = [
    'Draft',
    'Sacked %',
    'Completion Percentage',
    'Yards per Attempt',
    'Efficiency Rating'
]

df = pd.read_excel('../datasets/collegeToPros.xlsx', header=1, usecols=[0, 4, 6, 20, 28, 29, 30, 31])
df.rename(columns = renamed, inplace = True)
df['Sacked %'] = df['Sacked %']*100
print(df)
x_plot = df[['Rk']]

df_y = df['QB Rating']
df_x = df[['Draft',
    'Sacked %',
    'Completion Percentage',
    'Yards per Attempt',
    'Efficiency Rating']]

X = df_x
y = df_y
x = 0
n = -1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=531)
min_max_scaler = MinMaxScaler()
data_minmax = min_max_scaler.fit_transform(X_train)

f = open("../output/script1.txt", "w")
f.write("Output of the script 1\n")

regr = linear_model.LinearRegression()

sample_weight = y_train.apply(lambda h: 3 if h < 90 else 2)

regr.fit(data_minmax, y_train, sample_weight=sample_weight)

# Make predictions using the testing set
y_pred = regr.predict(min_max_scaler.transform(X_test))

y_plot = regr.predict(min_max_scaler.transform(df_x.to_numpy()))
plt.plot(x_plot.to_numpy(), df_y.to_numpy())
plt.plot(x_plot.to_numpy(), y_plot)
plt.savefig('../assets/comparison1.png')

f.write('\nscale ' + str(min_max_scaler.scale_))
f.write('\nmin ' + str(min_max_scaler.min_))
# The coefficients
f.write('\nCoefficients: \n' + str(regr.coef_))
f.write('\nIntercept: \n' + str(regr.intercept_))
# The mean squared error
f.write('\nMean squared error: %.2f\n'
    % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
f.write('\nCoefficient of determination: %.2f\n'
    % r2_score(y_test, y_pred))

f.close()