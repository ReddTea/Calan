import scipy as sp
import matplotlib.pyplot as plt



def normal_pdf(x, mean=0.0, variance = 0.01):
    var = 2 * variance
    thing = (var * sp.pi) ** -0.5
    thing = 1.
    return thing * sp.exp( - (x - mean) ** 2 / var)

'''
x = sp.linspace(-5, 5, 200)

plt.figure()
y = normal_pdf(x)
plt.plot(x, y)
plt.show()
'''
