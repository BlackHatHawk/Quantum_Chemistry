import numpy as np
from Function import *
from scipy.constants import Planck, pi

Hcut = Planck/(2*pi)

#one_dimensional
def kinetic(name: str, mass: float , X: np.ndarray):
    temp = Operator(name)
    h = -(Planck/(2*pi))**2/(2*mass)
    temp.Left_to_right(multiply,h)
    temp.Left_to_right(double_deriv, X)

    return temp

def position(name: str , X: np.ndarray):
    temp = Operator(name)
    temp.Left_to_right(multiply,X)

    return temp

def momentum(name: str, X: np.ndarray):
    temp = Operator(name)
    h = -(0+1j)*Hcut
    temp.Left_to_right(multiply,h)
    temp.Left_to_right(deriv,X)

    return temp

def potential(name: str, X: np.ndarray):
    temp = Operator(name)
    temp.Left_to_right(multiply, X)
    return temp


#three_dimensional
def angular_momentum(name: str, vector: Tuple[np.ndarray,np.ndarray]):
    
    def custom_operation(func: Callable[[np.ndarray],np.ndarray],
                         vector: Tuple[np.ndarray,np.ndarray]):
        return vector[0]*deriv(func,vector[1]) - vector[1]*deriv(func, vector[0])
    
    temp = Operator(name)
    h = -(0+1j)*Hcut
    temp.Left_to_right(multiply, h)
    temp.Left_to_right(custom_operation, vector)

    return temp

def Laplacian(name: str, vector: np.ndarray):
    def custom_operation(func: Callable[[np.ndarray],np.ndarray],
                        vector: np.ndarray):
        
        funcX,inputX = partial_func_gen(func,vector,0) 
        funcY,inputY = partial_func_gen(func,vector,1)
        funcZ,inputZ = partial_func_gen(func,vector,2)

        part1 = power_deriv(funcX,inputX,2)
        part2 = power_deriv(funcY,inputY,2)
        part3 = power_deriv(funcZ,inputZ,2)
        return part1+part2+part3
    tm = Operator(name)
    tm.Left_to_right(custom_operation, vector)

    return tm

def Hermitian(name: str, Kinetic: Operator, Potential: Operator):
    pass
    
def Spherical_Laplacian(name: str, vector: np.ndarray):
    def _custom_operation(func: Callable[[np.ndarray],np.ndarray],
                        vector: Tuple[np.ndarray,np.ndarray,np.ndarray]):
        funcR, inputR = partial_func_gen(func,vector,0)
        funcTheta, inputTheta = partial_func_gen(func,vector,1)
        funcPsi, inputPsi = partial_func_gen(func,vector, 2)
        
        # d/dr(r^2 (d/dr))
        def custom_deriv1(func: Callable[[np.ndarray],np.ndarray], input_: np.ndarray, delta: float = 0.001):
            return ((input_+delta)**2 * func(input_ + 2* delta) - 4*delta* input_ * func(input_) + (input_ - delta)**2 *func(input_ - 2*delta))/(4*delta*delta)

        # d/dx(sinx (d/dx))
        def custom_deriv2(func: Callable[[np.ndarray], np.ndarray], input_: np.ndarray, delta: float = 0.001):
            return (np.sin(input_+ delta)*func(input_+2*delta) - 2*np.sin(input_)*np.cos(delta)*func(input_) + np.sin(input_ - delta)* func(input_ - 2*delta))/(4*delta*delta)

        one_by_r_sqr = 1/(vector[0]**2)
        one_by_sin_theta = 1/np.sin(vector[1])
        part1 = one_by_r_sqr * custom_deriv1(funcR, inputR)
        part2 = one_by_r_sqr * one_by_sin_theta * custom_deriv2(funcTheta,inputTheta)
        part3 = one_by_r_sqr * one_by_sin_theta**2 * double_deriv(funcPsi, inputPsi)

        return part1 + part2 + part3

    tmp = Operator(name)
    tmp.Left_to_right(_custom_operation,vector)

    return tmp


