# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 22:47:27 2020

@author: HP
"""

import numpy as np
from numba import jit
'''
#@jit
def test():
    a = [1,2]
    for i in range(10):
        a.append([1,2])

test()



def solve_function(x):
    return x**3-3*x-2

def solve_derivatives(x):
    return 3*x**2-3

def Newton(x,eps):
    count=0
    while abs(solve_function(x))>eps:
        x=x-solve_function(x)/solve_derivatives(x)
        count=count+1
    return count,x


x=3
eps=0.0001
count,middle=Newton(x,eps)
print("迭代%d次得到的根是%f" %(count,middle))

@jit
def polyfit():
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, -2.3, 10)
    k = (10 * np.sum(x*y) - np.sum(y)*np.sum(x)) / (10 * np.sum(x**2) - (np.sum(x))**2)
    #z = np.polyfit(x, y, 1)
    print(k)
#print(k)
polyfit()

#@jit
def interplot():
    x = np.linspace(0, 4, 12)
    y = np.cos(x**2/3+4)
    
    xn = np.linspace(0, 4, 100)
    y0 = np.cos(xn**2/3+4)
    yn = np.interp(x, x.tolist(), y.tolist())
    
    return xn, y0, x, yn
xn, y0, x, yn = interplot()
plt.plot(x, yn, 'or', label='True values')
plt.plot(x, y, 'ok', label='Known points')
'''
def dR_db(b):
    p = -b**2
    q = -2*M*b**2
    delta = q**2/4 + p**3/27
    
    if delta > 0:
        return False, 0
    Q = -p/3
    R = q/2
    theta = np.arccos(R/Q/np.sqrt(Q))
    x = 2*np.sqrt(Q)*np.cos((theta + 0*np.pi)/3)
    
    dQ_db = 2*b/3
    dtheta_db = -1/np.sqrt(1-R**2 / Q**3) * (1/3**2.5 - 2/3**1.5) * M/b**2
    
    dx_db = 1/np.sqrt(Q) * dQ_db * np.cos(theta/3) - \
            2 * np.sqrt(Q) * np.sin(theta/3) * theta/3 * dtheta_db
    print(dx_db)
    return dx_db

#dR_db(1e10)
    


a = [1, 2, 3]
b = [4, 5, 6]

for i, j in zip(a,b):
    print(i, j)
