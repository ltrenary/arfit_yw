def arfit_yw(x,max_order=10):
    import numpy as np
    """
     PURPOSE:
       This function selects the order (p) and returns the AR(p)-parameters for a numpy array using the 
       Yule-Walker equations to estimate coefficients. The best-fit AR(p) model is determined with respect to a 
       revised Akaike Information Criterion (AICr). 
      
     REFERENCES: 
     Details of the AICr can be found in DelSole and Tippett (2021): https://www.sciencedirect.com/science/article/pii/S0167715221000262
 
     CALLING SEQUENCE:
     order,phi,aic_ar,res,varp,aicc,aic,bic = arfit_yw(simulated_data,max_order)

     INPUTS:
       data: A 1-D numpy array 
       max_order: The max_order of the AR(p) model to test for fit 
    OUTPUTS:
       order: The order 'p'  of the best fit AR(p) model
       phi: The AR coefficients
       AICr: The revised Akaike Information Criterion used to select the best-fit AR(p) model.
       residuals: Residuals from the best fit AR(p) model
       AICc: A corrected Akaike Information Criterion 
       AIC: The standard Akaike Information Criterion
       BIC: The Bayesian Information Criterion
     Written by L. Trenary (2022) and based on R code by T. Delsole
    """
    nsamp = x.size
    npic  = np.arange((max_order),nsamp,1)
    ntot = npic.size
    
    y = x[npic]
    y = y.reshape((npic.size,1))
    xmat = np.repeat(1, ntot, axis=0)
    for p in range(0,max_order,1):
        xmat = np.vstack([xmat, x[npic-(p+1)]])
    xmat = xmat.transpose()


    c_tau   = np.empty((max_order+1,1))
    c_tau[:] = np.nan
    x_mean = x.mean()
    for p in range(0,max_order+1,1):
        n = np.arange(p,nsamp,1)
        ss = ((x[n]-x_mean)* (x[n-p]-x_mean)).sum()
        c_tau[p] = ss/nsamp

    ctau_vec = c_tau[1:max_order+1]
    ctau_mat = np.empty((max_order,max_order))
    ctau_mat[:] = np.nan
    for j in range(0,max_order,1):
        for i in range(0,max_order,1):
            ctau_mat[i,j] = c_tau[abs(i-j)]

    phi_save = np.empty((max_order,max_order))
    aicr_ar = np.empty([max_order+1], dtype=float)
    aicc_ar = np.empty([max_order+1], dtype=float)
    aic_ar = np.empty([max_order+1], dtype=float)
    bic_ar = np.empty([max_order+1], dtype=float)
    phi_save[:] = np.nan
    for p in range(0,max_order+1,1): 
        m  = p + 1
        
        if p == 0:
                phi = 0
                sse = ((y - x_mean)**2).sum()
        else: 
                phi = np.linalg.solve(ctau_mat[0:p,0:p],ctau_vec[0:p])
                sse = ( (y - x_mean - np.matmul(xmat[:,1:p+1], phi))**2).sum()
                
                phi_save[0:p,p-1] = phi.reshape(phi.size)
        aicr_ar[p] = ntot * ( np.log(sse/ntot) + (ntot+1) * (ntot - 2)/(ntot-m-2)/(ntot-m-1))
        aicc_ar[p] = ntot * ( np.log(sse/ntot) + (ntot + m)/(ntot-m-2))
        aic_ar [p] = ntot * ( np.log(sse/ntot) + 2*m/ntot )
        bic_ar [p] = ntot * ( np.log(sse/ntot) + np.log(ntot) * m / ntot)

        
    m_min     = np.argmin(aicr_ar)
    p_min = m_min
    
    if m_min == 0:
        phi = 0
        residuals = y - x_mean
    else:
        
        phi = phi_save[0:p_min,p_min-1]
        
        if len(phi) == 1:
                residuals = (y - x_mean - xmat[:,1:p_min+1]*phi)
        else:     
                residuals = (y - x_mean - np.multiply(xmat[:,1:p_min+1], phi))
     
    var_pred  = (residuals**2).sum()/(ntot-(m_min+1))
    
    return(p_min,phi, aicr_ar, residuals,var_pred,aicc_ar, aic_ar,bic_ar)


