import numpy as np
from typing import Callable, Tuple
from Function import fourierTrick, fourierTrick_spherical_angular, integrate, fourierTrick_spherical_radial
def Area_Indexing(ProbCurve: Callable[[np.ndarray],np.ndarray],limits:Tuple[np.ndarray,np.ndarray],parts: int=1000):

    assert integrate(ProbCurve,limits) == 1 ,\
        'function does not represent probability distribution'

    delx = np.linspace(limits[0],limits[1],parts+1)
    area_delx = np.zeros(parts,2)
    for i in range(parts):
        if i != 0:
            area_delx[i,0] = area_delx[i-1,0]
        area_delx[i,1] = area_delx[i,0] + integrate(ProbCurve,(delx[i],delx[i+1]))

    return area_delx

def Area_Indexing_wave(wavefunction: Callable[[np.ndarray],np.ndarray],limits:Tuple[np.ndarray,np.ndarray],parts: int=1000):

    '''
    assert fourierTrick(wavefunction,wavefunction,limits) == 1 ,\
        'wavefunction is not normalised'
    '''
    delx = np.linspace(limits[0],limits[1],parts+1)
    area_delx = np.zeros((parts,2))
    for i in range(parts):
        if i != 0:
            area_delx[i,0] = area_delx[i-1,1]
        area_delx[i,1] = area_delx[i,0] + fourierTrick(wavefunction,wavefunction,(delx[i],delx[i+1]))

    return area_delx

def Area_Indexing_spherical_wave(wavefunction: Callable[[np.ndarray],np.ndarray],limits:Tuple[np.ndarray,np.ndarray],parts: int=1000):

    '''
    assert fourierTrick(wavefunction,wavefunction,limits) == 1 ,\
        'wavefunction is not normalised'
    '''
    delx = np.linspace(limits[0],limits[1],parts+1)
    area_delx = np.zeros((parts,2))
    for i in range(parts):
        if i != 0:
            area_delx[i,0] = area_delx[i-1,1]
        area_delx[i,1] = area_delx[i,0] + fourierTrick_spherical_radial(wavefunction,(delx[i],delx[i+1]))

    return area_delx

def prob_wt_cal(funcradial: Callable[[np.ndarray],np.ndarray],
                                functheta: Callable[[np.ndarray],np.ndarray],
                                funcphi: Callable[[np.ndarray],np.ndarray],
                                limits_r: Tuple[float,float],
                                limits_theta:Tuple[float,float],
                                limits_phi:Tuple[float,float],
                                parts: int=1000):

    del_r = np.linspace(limits_r[0],limits_r[1],parts+1)
    del_theta = np.linspace(limits_theta[0],limits_theta[1],parts+1)
    del_phi = np.linspace(limits_phi[0],limits_phi[1],parts+1)

    def group_1(theta: np.ndarray):
        return (functheta(theta).conjugate()*functheta(theta)).real * np.sin(theta)
    def group_2(phi: np.ndarray):
        return (funcphi(phi).conjugate()*funcphi(phi)).real

    wt_r = np.zeros(parts)
    wt_theta = np.zeros(parts)
    wt_phi = np.zeros(parts)

    for i in range(parts):
        wt_r[i] = fourierTrick_spherical_radial(funcradial,(del_r[i],del_r[i+1]))
        wt_theta[i] = integrate(group_1,(del_theta[i],del_theta[i+1]))
        wt_phi[i] = integrate(group_2,(del_phi[i],del_phi[i+1]))

    return wt_r,wt_theta,wt_phi
    


