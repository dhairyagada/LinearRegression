import scipy.io as spio
from sklearn import linear_model
import matplotlib.pyplot as plt
from pylab import polyval
import numpy as np


mat = spio.loadmat('co2.mat', squeeze_me=True)
y=mat['co2']
X=mat['year']

X=X.reshape(-1,1)

lm = linear_model.LinearRegression(fit_intercept=True)
model = lm.fit(X,y)

x_preds=np.linspace(2015,2016,10,endpoint=False)
print('Years for prediction of data')
print('\n'.join(map(str, x_preds)))
x_preds=x_preds.reshape(-1,1)
predictions = lm.predict(x_preds)

print('Prediction Data: ')
print('\n'.join(map(str, predictions)))


print('Coefficient : ',model.coef_)
print('Intercept : ',model.intercept_)

yp= polyval([model.coef_,model.intercept_],X)

plt.scatter(X,y,color='black')
plt.plot(X,yp,color='red',linewidth=3)
plt.xlabel('Years')
plt.ylabel('CO2 Data')
plt.show()