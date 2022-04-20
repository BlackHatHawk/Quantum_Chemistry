import Function as Fn
import numpy as np
from typing import List,Callable
from scipy.constants import pi


def One_D(X : np.ndarray , length: float= 10 , node : int=1) ->np.ndarray:
    a = length
    n = node
    return np.math.pow((2/a),0.5) * np.sin((n*pi*X)/a)

def square(X : np.ndarray) ->np.ndarray:

    return X**2

def cube(X : np.ndarray) -> np.ndarray:
    return X**3

def power(X: np.ndarray, pow: int=  1) -> np.ndarray:
    return X**pow

def Leguerre_poly(X,q,power=1000):
    def func(X,Z=q):
        return np.math.exp(-X)*X**Z
    return np.math.exp(X)* Fn.power_deriv(func,X,q,delta=1/power)

def Ass_Leguerre_poly(X,p,q):
    def func(X,Z=q):
        return Leguerre_poly(X,Z)
    return (-1)**p * Fn.power_deriv(func,X,p)

def Rodrigues(X,l):
    def func(X,con=l):
        return (X**2 -1)**con
    return (1/(2**l*np.math.factorial(l))) * Fn.power_deriv(func,X,l)

def Ass_Legendre(X,l,m):
    assert X<=1 and X>=-1,\
        'parameter value of Associated Legendre function is out of domain, pass either sin or cosine function'
    m = np.abs(m)
    def func(X,con = l):
        return Rodrigues(X,con)
    
    return (1-X**2)**(m/2) * Fn.power_deriv(func,X,m)

def angular_Wave_func(Theta, Psi, l : int, m: int):
    m = np.abs(m)
    part1 = (-1)**m
    part2 = (2*l +1)/(4*pi)
    part3 = np.math.factorial(l-m)/np.math.factorial(l+m)
    part4 = np.math.exp((0+1j)*m*Psi)
    param = np.math.cos(Theta)
    part5 = Ass_Legendre(param,l,m)

    return part1 * (part2*part3)**0.5 * part4 * part5

def Radial(n,l,r,a):
    #Not finished
    part1 = (2/(n*a))**3
    part2 = np.math.factorial(n-l-1)/(2*n*(np.math.factorial(n+l)**3))
    part3 = np.math.exp((-r)/(n*a))
    part4 = ((2*r)/(n*a))**l
    param = 2*r/(n*a)
    part5 = Ass_Leguerre_poly(param,(2*l+1),(n-l-1))

def Norm_Y(m,l):
    part1 = (-1)**m
    part2 = (2*l +1)/(4*pi)
    part3 = np.math.factorial(l-m)/np.math.factorial(l+m)
    return part1* (part2*part3)**0.5

#spherical harmonics

def Y_theta(theta: np.ndarray,l: int,m: int,Norm: float = 1):
    mod_m = np.abs(m)
    if m>=0:
        epsilon = (-1)**mod_m
    else:
        epsilon = 1

    return Norm * epsilon * np.sin(theta)**mod_m * np.cos(theta)**(l-mod_m)

def Y_phi(phi,m,Norm: float=1):

    return Norm  * np.exp(0+(phi*m)*1j)

#List of radial wavefunction of Hydrogen like nucleus, written by Anurag Maurya
# R_< n >_< l >

def R_1_0(r:np.ndarray, a: float = 100):

    norm = 2*(a**(-3/2))
    return norm * np.exp(-r/a)

def R_2_0(r:np.ndarray, a: float = 100):
    norm = a**(-3/2)/2**0.5
    return norm * (1-0.5*(r/a))* np.exp(-r/(2*a))

def R_2_1(r: np.ndarray, a: float=100):
    norm = a**(-3/2)/24**0.5
    return norm * (r/a) * np.exp(-r/(2*a))

def R_3_0(r:np.ndarray, a: float = 100):
    norm = 2*(a**(-3/2))/27**0.5
    return norm * (1 - (2/3)*(r/a) + (2/27)* (r/a)**2) * np.exp(-r/(3*a))

def R_3_1(r: np.ndarray, a: float= 100):
    norm = (8/27)*(1/6**0.5)*a**(-3/2)
    return norm * (1-(1/6)*(r/a))*(r/a)*np.exp(-r/(3*a))

def R_3_2(r: np.ndarray, a: float = 100):
    norm = (4/81)*(1/30**0.5)*a**(-3/2)
    return norm * (r/a)**2 * np.exp(-r/(3*a))

def R_4_0(r:np.ndarray, a: float = 100):
    norm = a**(-3/2)/4
    return norm * (1 - (3/4)*(r/a) + (1/8)* (r/a)**2 - (1/192)*(r/a)**3) * np.exp(-r/(4*a))

def R_4_1(r:np.ndarray, a: float = 100):
    norm = 1/16 * (5/3)**0.5 * a**(-3/2)
    return norm * (1 - (1/4)*(r/a) + (1/80)* (r/a)**2) * (r/a) * np.exp(-r/(4*a))

def R_4_2(r: np.ndarray, a: float= 100):
    norm = (1/64)*(1/5**0.5)*a**(-3/2)
    return norm * (1-(1/12)*(r/a)) * (r/a)**2 *np.exp(-r/(4*a))

def R_4_3(r: np.ndarray, a: float= 100):
    norm = (1/768)*(1/35**0.5)*a**(-3/2)
    return norm * (r/a)**3 * np.exp(-r/(4*a))