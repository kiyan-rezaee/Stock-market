import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

data = pd.read_csv('data/AMZN_2006-01-01_to_2018-01-01.csv')
# print(data.head())

data = data[['High']]
# print(data.head())

x = data['High'].values #to numpy array
X_train, X_test = x[:2500], x[2500:]

XT, yT, Xt, yt = [], [], [], []
for i in range(len(X_train) - 90):
    d = i + 90
    XT.append(X_train[i:d,])
    yT.append(X_train[d])
for i in range(len(X_test) - 90):
    d = i + 90
    Xt.append(X_test[i:d,])
    yt.append(X_test[d])
XT = np.array(XT)
Xt = np.array(Xt)
yT = np.array(yT)
yt = np.array(yt)
XT = np.reshape(XT, (XT.shape[0], XT.shape[1], 1))
Xt = np.reshape(Xt, (Xt.shape[0], Xt.shape[1], 1))

model = Sequential()
model.add(SimpleRNN(units=64, activation='tanh'))
model.add(Dense(1))
model.compile(loss='mean_absolute_error', optimizer='adam')
mymodel = model.fit(XT, yT, epochs=100, batch_size=20)
model.evaluate(Xt, yt)

XTPredicted = model.predict(XT)
XtPredicted = model.predict(Xt)
XFinal = np.concatenate([XTPredicted, XtPredicted], axis=0)
plt.plot(x, color='green')
plt.plot(XFinal, color='yellow')
plt.show()