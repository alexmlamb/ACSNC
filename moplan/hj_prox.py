__all__ = ["compute_hj_prox"]

import torch
import numpy as np

def compute_hj_prox(x, t, f, delta=1e-1, int_samples=1000, alpha=2.0,
                    recursion_depth=0, alpha_decay=0.631, tol=1.0e-6, 
                    tol_underflow=0.9, device='cpu', verbose=False,
                    return_samples=False):
    """ Estimate proximals from function value sampling via HJ-Prox Algorithm.

        Args:
            x (tensor): Input vector
            t (tensor): Time > 0
            f: Function to minimize

        Returns:
            tensor: Estimate of the proximal of f at x
    """
    valid_vector_shape = x.shape[1] == 1 and x.shape[0] >= 1
    assert valid_vector_shape

    recursion_depth +=1
    std_dev = np.sqrt(delta * t / alpha)
    dim     = x.shape[0]
    y       = std_dev * torch.randn(int_samples, dim, device=device) 
    y       = y + x.permute(1,0)
    z       = -f(y) * (alpha / delta)

    underflow         = torch.exp(z)  <= tol
    underflow_freq    = float(underflow.sum()) / underflow.shape[0]
    observe_underflow = underflow_freq > tol_underflow

    if observe_underflow: 
        alpha *= alpha_decay
        return compute_hj_prox(x, t, f, delta=delta, int_samples=int_samples,
                               alpha=alpha, recursion_depth=recursion_depth,
                               alpha_decay=alpha_decay, tol=tol, 
                               tol_underflow=tol_underflow, device=device,
                               verbose=verbose, return_samples=return_samples)         
    else:                
        soft_max = torch.nn.Softmax(dim=1)  
        HJ_prox  = soft_max(z.permute(1,0)).mm(y)   

        valid_prox_shape = HJ_prox.shape == x.shape
        assert valid_prox_shape

        prox_is_finite = (HJ_prox < np.inf).all()
        assert prox_is_finite 

        if verbose:
            envelope = - (delta / alpha) * torch.log(torch.mean(torch.exp(z)))
            return HJ_prox, recursion_depth, envelope
        elif return_samples:
            return HJ_prox, y, alpha
        else:
            return HJ_prox
