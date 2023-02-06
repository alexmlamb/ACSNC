__all__ = ["chomp_cost_helper", "chomp_cost"]

__author__ 		= "Lekan Molu"
__copyright__ 	= "2022, Discrete Cosserat SoRO Analysis in Python"
__credits__  	= "Alex Lamb, Shaoru Chen, Anurag Koul."
__license__ 	= "MSR Licence"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__comments__    = "This code was written during the IROS/CDC 2022 deadlines."
__loc__         = "Philadelphia, PA"
__date__ 		= "February 06, 2023"
__status__ 		= "Completed"

import copy 
 
def chomp_cost_helper(latent_state):
    """
        Computes the cost for a latent state trajectory using 
        finite differencing. This is based on equation (16) in
        the CHOMP paper:
            https://www.ri.cmu.edu/pub_files/2013/5/CHOMP_IJRR.pdf

        Inputs: 
            latent_state: A 1-dim array of encoded latent states.
    """

    assert isinstance(latent_state, np.ndarray), "Input array has to be numpy type."

    latent_state_out_r          = np.zeros(latent_state.shape[0]+2) 
    latent_state_out_l          = np.zeros(latent_state.shape[0]+2) 
    latent_state_out_r[1:-1]    = copy.copy(latent_state)
    latent_state_out_l[:-2]     = copy.copy(latent_state)
    diff                        = latent_state_out_l - latent_state_out_r
    ans                         = diff[1:-1]
    ans[-1]                     = 0
    
    dxInv = 1.0/(latent_state[1]-latent_state[2])
    diff = dxInv*(ans)

    cost_this = 0.5*np.linalg.norm(diff ** 2, ord=2, axis=0)
    
    return cost_this 

def chomp_cost(latent_states):
    """
        Computes the respective costs for all latent states in a 
        latent state batch. Respective costs are computed in the 
        subroutine `chomp_cost_helper`.

        Inputs:
            latent_states: A list of latents (a numpy array) sampled from observations.
    """
    costs = 0
    for latent in latent_states:
        costs += chomp_cost_helper(latent)
    
    return costs     