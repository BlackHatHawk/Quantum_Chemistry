from math import pi
import numpy as np
from typing import Callable,Dict,Tuple,List


def integrate(func: Callable[[np.ndarray],np.ndarray],limits: Tuple[np.ndarray,np.ndarray], steps: float = 1000):
    assert steps >1
    X = np.linspace(limits[0],limits[1],steps)
    delta= X[1]-X[0]
    result = func(X[:-1]+delta/2)*delta
    return  np.sum(result)

def double_intergrate(func: Callable[[np.ndarray,np.ndarray],np.ndarray],Xlimits: Tuple[np.ndarray,np.ndarray],Ylimits: Tuple[np.ndarray,np.ndarray], steps: float = 1000):
    assert steps>1
    X, Y =  np.linspace(Xlimits[0],Xlimits[1],steps), np.linspace(Ylimits[0],Ylimits[1],steps)
    delta_X, delta_Y = X[1]-X[0], Y[1]-Y[0]
    C = np.meshgrid(X,Y)
    res = func(C[0],C[1])*(delta_X*delta_Y)
    return np.sum(res)

def double_intergrate_1(func: Callable[[np.ndarray,np.ndarray],np.ndarray],Xlimits: Tuple[np.ndarray,np.ndarray],Ylimits: Tuple[np.ndarray,np.ndarray], steps: float = 1000):
    assert steps>1
    X, Y =  np.linspace(Xlimits[0],Xlimits[1],steps), np.linspace(Ylimits[0],Ylimits[1],steps)
    delta_X, delta_Y = X[1]-X[0], Y[1]-Y[0]
    res = 0
    for i in range(steps):
        res += np.sum(func(X,Y[i])*delta_X*delta_Y)
    return res

def deriv(func: Callable[[np.ndarray],np.ndarray], input_: np.ndarray, delta:float = 0.001):

    return ( func(input_+delta)-func(input_-delta))/(2*delta)

def double_deriv(func: Callable[[np.ndarray],np.ndarray], input_: np.ndarray, delta: float = 0.001):
    return (func(input_ + 2* delta) - 2* func(input_) + func(input_ - 2*delta))/(4*delta*delta)

def power_deriv(func: Callable[[np.ndarray],np.ndarray], input_: np.ndarray, power:int=1, delta:float = 0.001):
    if power==1:
        return ( func(input_+delta)-func(input_-delta))/(2*delta)
    else:
        return (power_deriv(func,input_+delta,power-1) - power_deriv(func, input_-delta,power-1))/(2*delta)

#following function accept a function which depends on three variable and returns a function that takes only one
#variable and makes other 2 constant.
# func --> function as f(X,Y,Z)
#'input_' --> (:,2) array as 2 constant
# variable_index --> position of variable as in (X,Y,Z); 0,1,2 for X,Y,Z respectively 
def partial_func_gen(func: Callable[[np.ndarray],np.ndarray],input_:np.ndarray,variable_index: int) -> Tuple[Callable[[np.ndarray],np.ndarray], np.ndarray]:
    if variable_index == 0:
        def mod_func(X:np.ndarray,Y=input_[:,1],Z=input_[:,2]):
            return func(X,Y,Z)
    if variable_index == 1:
        def mod_func(Y: np.ndarray,X=input_[:,0], Z = input_[:,2]):
            return func(X,Y,Z)
    if variable_index ==2:
        def mod_func(Z: np.ndarray, X= input_[:,0], Y = input_[:,1]):
            return func(X,Y,Z)
    return mod_func, input_[:,variable_index]

def power_deriv_mem(func: Callable[[np.ndarray],np.ndarray], input_: np.ndarray, power:int=1, mem: Dict[str,np.ndarray]=None, delta:float = 0.001):
    if mem==None:
        mem = {}
    if power==1:
        return ( func(input_+delta)-func(input_-delta))/(2*delta)
    if str(input_)+str(power) in mem.keys():
        return mem[str(input_)+str(power)]
    else:
        mem[str(input_)+str(power)] = (power_deriv_mem(func,input_+delta,power-1,mem) - power_deriv_mem(func, input_-delta,power-1,mem))/(2*delta)
        return mem[str(input_)+str(power)]

def multiply(func: np.ndarray, input: np.ndarray)->np.ndarray:
    return input*func

def add(func: np.ndarray, input: np.ndarray) -> np.ndarray:
    return input + func

def subtr(func: np.ndarray, input: np.ndarray) -> np.ndarray:
    return input - func

def blank():
    return 1
    

class Operator:
    def __init__(self,name:str):
        self.name = name
        self.operation : List = []
        self.inputparam : List = []
    
    def Left_to_right(self,operation: Callable[[np.ndarray],np.ndarray] , inputparam):
        self.operation.append(operation)
        self.inputparam.append(inputparam)

    def compute(self, func:Callable[[np.ndarray],np.ndarray]) -> np.ndarray:

        count = len(self.operation)
        cache = self.operation[-1](func,self.inputparam[-1])
        for i in range(1,count):
            cache = self.operation[-i-1](cache, self.inputparam[-i-1])

        return cache

class ComplexOperator:
    def __init__(self,name: str,partcount: int):
        self.name = name
        self.partcount = partcount
        self.parts : List[Operator]

    def storeparts(self, operators: List[Operator]):
        for op in operators:
            self.partcount.append(op)
    
    def compute(self, Wavefunction: np.ndarray):
        partcompute : List[np.ndarray]
        for part in self.parts:
            partcompute.append(part.compute(Wavefunction))

        return partcompute
    
class EigenOperator():
    def __init__(self, name: str,cookies):
        self.cookies = cookies
        self.name = name

    def compute(self, opertor: Operator) -> np.ndarray:

        if opertor.name in self.cookies:
            func =  self.cookies[opertor]
        else:
            func = opertor.compute(np.linspace(0,100,1000))


        return  np.linalg.eigh(func)

def fourierTrick(func1: Callable[[np.ndarray],np.ndarray],
                 func2: Callable[[np.ndarray],np.ndarray],
                 limits:Tuple[float,float]):
    def group(X: np.ndarray):
        return func1(X) * func2(X)
    return integrate(group,limits)

def fourierTrick_spherical_radial(func: Callable[[np.ndarray],np.ndarray],
                                limits: Tuple[float,float]
                                ):
    def group(X: np.ndarray):
        return func(X) * func(X) * X * X
    return integrate(group,limits)

def fourierTrick_spherical_angular(functheta: Callable[[np.ndarray],np.ndarray],
                                    funcphi: Callable[[np.ndarray],np.ndarray],
                                    limitsTheta: Tuple[float, float],
                                    limitsPsi: Tuple[float,float]
                                    ):
    def group_1(theta: np.ndarray):
        return (functheta(theta).conjugate()*functheta(theta)).real * np.sin(theta)
    def group_2(phi: np.ndarray):
        return (funcphi(phi).conjugate()*funcphi(phi)).real
    return integrate(group_1,limitsTheta) * integrate(group_2,limitsPsi)

def Norm_sph_harmonics(functheta: Callable[[np.ndarray],np.ndarray],
                            funcphi:Callable[[np.ndarray],np.ndarray],
                            limitsTheta: Tuple[float, float],
                            limitsPsi: Tuple[float,float]):
    
    def group_1(theta: np.ndarray):
        return (functheta(theta).conjugate()*functheta(theta)).real * np.sin(theta)
    def group_2(phi: np.ndarray):
        return (funcphi(phi).conjugate()*funcphi(phi)).real

    X,Y =  integrate(group_1,limitsTheta), integrate(group_2,limitsPsi)
    return (1/X)**0.5, (1/Y)**0.5
