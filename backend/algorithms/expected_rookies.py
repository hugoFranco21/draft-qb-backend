import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

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

df = pd.read_excel('../datasets/collegeToPros.xlsx', header=1, usecols=[4, 6, 20, 28, 29, 30, 31])
df.rename(columns = renamed, inplace = True)
df['Sacked %'] = df['Sacked %']*100
print(df)

df_y = df['QB Rating']
df_x = df[['Draft',
    'Sacked %',
    'Completion Percentage',
    'Yards per Attempt',
    'Efficiency Rating']]

X = df_x
y = df_y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

regr = linear_model.LinearRegression(normalize=True)

sample_weight = y_train.apply(lambda h: 3 if h < 90 else 2)

regr.fit(X_train, y_train, sample_weight=sample_weight)

# Make predictions using the testing set
y_pred = regr.predict(X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print('Mean squared error: %.2f'
    % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
    % r2_score(y_test, y_pred))


