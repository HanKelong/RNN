import numpy as np
from matplotlib import pyplot as plt

def Tanh(x):
	return 2 / (1 + np.exp(-2*x)) - 1

def dTanh(x):
	return 1-Tanh(x)*Tanh(x)

x = np.linspace(-10,10,num=100)
y = Tanh(x)
dy= dTanh(x)

plt.subplot(1,2,1)
plt.title('Tanh')
plt.plot(x,y)
plt.subplot(1,2,2)
plt.title('grad of Tanh')
plt.plot(x,dy)
plt.show()
