# -*- coding: utf-8 -*-
"""
Created on Fri May 24 08:37:33 2024

@authors: David Bailey and Marcos López de Prado
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2197616
@copier: Charles Rambo
"""

import numpy as np 


class CLA:
    
    def __init__(self, mean, covar, lB, uB):
     
        # Initialize the class
        if (mean == np.ones(mean.shape) * mean.mean()).all():
            
            mean[-1,0] += 1e-5
                     
        self.mean = np.asarray(mean)
        self.covar = np.asarray(covar)
        self.lB = np.asarray(lB)
        self.uB = np.asarray(uB)
        
        # Solution
        self.w = [] 
        
        # Lambdas
        self.lam = []
        
        # Gammas
        self.g = [] 
        
        # Free weights
        self.f = []


#---------------------------------------------------------------        
    def initialize_algo(self):
        
        # Initialize the algorithm
        
        #1) Form structured array
        a = np.zeros((self.mean.shape[0]), 
                     dtype = [('id', int), ('mu', float)])
        
        # Dump array into list
        b = [self.mean[i][0] for i in range(self.mean.shape[0])]
        
        # Fill structured array
        a[:] = list(zip(range(self.mean.shape[0]), b))
        
        #2) Sort structured array
        b = np.sort(a, order = 'mu')
        
        #3) First free weight
        i, w = b.shape[0], np.copy(self.lB)
        
        while sum(w) < 1.0:
            
            i -= 1
            
            w[b[i][0]] = self.uB[b[i][0]]
        
        w[b[i][0]] += 1 - sum(w)
        
        return [b[i][0]], w


#---------------------------------------------------------------        
    def get_b(self,f):
        
        return self.diff_lists(range(self.mean.shape[0]), f)
   
    
#---------------------------------------------------------------    
    def compute_bi(self, c, bi):
        
        if c > 0:
            
            bi = bi[1]
            
        if c < 0:
            
            bi = bi[0]
            
        return bi
 
    
#---------------------------------------------------------------
    def diff_lists(self,list1,list2):
        
        return list(set(list1) - set(list2))


#---------------------------------------------------------------
    def get_matrices(self, f):
        
        # Slice covarF, covarFB, covarB, meanF, meanB, wF, wB
        covarF = self.reduce_matrix(self.covar, f, f)
        
        meanF = self.reduce_matrix(self.mean, f, [0])
        
        b = self.get_b(f)
        
        covarFB = self.reduce_matrix(self.covar, f, b)
        
        wB = self.reduce_matrix(self.w[-1], b, [0])
        
        return covarF, covarFB, meanF, wB


#---------------------------------------------------------------
    def reduce_matrix(self, matrix, listX, listY):
        
        # Reduce a matrix to the provided list of rows and columns
        if len(listX) == 0 or len(listY) == 0: 
            
            return None
        
        # Subset to just rows we want
        matrix_ = matrix[listX, :]
         
        # Subset to just columns we want
        return matrix_[:, listY]
 
    
#---------------------------------------------------------------
    def compute_lambda(self, covarF_inv, covarFB, meanF, wB, i, bi):
        
     #1) C
     onesF = np.ones(meanF.shape)
     
     c1 = (onesF.T @ covarF_inv) @ onesF
     
     c2 = covarF_inv @ meanF
     
     c3 = (onesF.T @ covarF_inv) @ meanF
     
     c4 = covarF_inv @ onesF
     
     c = -c1 * c2[i] + c3 * c4[i]
     
     if c == 0:
         
         return np.nan, np.nan
     
     #2) bi
     if type(bi) == list:
         
         bi = self.compute_bi(c,bi)
     
     #3) Lambda
     if wB is None:
         
         # All free assets
         return float((c4[i] - c1 * bi)/c), bi
     
     else:
         
        onesB = np.ones(wB.shape)
        
        lam1 = onesB.T @ wB
        
        lam2 = covarF_inv @ covarFB
        
        lam3 = lam2 @ wB
        
        lam2 = onesF.T @ lam3
        
        return float(((1 - lam1 + lam2) * c4[i] - c1 * (bi + lam3[i]))/c), bi
 
    
#---------------------------------------------------------------
    def compute_w(self, covarF_inv, covarFB, meanF, wB):
        
         #1) Compute gamma
         onesF = np.ones(meanF.shape)
         
         g1 = (onesF.T @ covarF_inv) @ meanF
         
         g2 = (onesF.T @ covarF_inv) @ onesF
         
         if wB is None:
             
             g, w1 = float(-self.lam[-1] * g1/g2 + 1/g2), 0
             
         else:
             
             onesB = np.ones(wB.shape)
             
             g3 = onesB.T @ wB
             
             g4 = covarF_inv @ covarFB
             
             w1 = g4 @ wB
             
             g4 = onesF.T @ w1
             
             g = float(-self.lam[-1] * g1/g2 + (1 - g3 + g4)/g2)
             
         #2) compute weights
         w2 = covarF_inv @ onesF
             
         w3 = covarF_inv @ meanF
         
         # Get free weight
         wF = -w1 + g * w2 + self.lam[-1] * w3
         
         # Make sure sum is 1
         #wF *= (1 - np.sum(wB))/np.sum(wF)
         
         return wF, g

         
#---------------------------------------------------------------
    def get_min_var(self):
        
        # Get the minimum variance solution
        var = [w.T @ self.covar @ w for w in self.w]
        
        # Get the index corresponding to the lowest variance
        idx_min = np.argmin(var)
            
        return np.sqrt(var[idx_min]), self.w[idx_min]  
 
    
#---------------------------------------------------------------    
    def golden_section(self, obj, a, b, **kwargs):
         # Golden section method. Maximum if kargs['minimum'] == False is passed
         
         tol, sign, args = 1.0e-9, 1, None
         
         if 'minimum' in kwargs and kwargs['minimum'] == False:
             
             sign = -1
             
         if 'args' in kwargs:
             
             args = kwargs['args']
             
         numIter = int(-2.078087 * np.log(tol/np.abs(b-a)) + 1)
         
         r = 0.618033989
         c = 1.0 - r
         
         # Initialize
         x1 = r * a + c * b
         x2 = c * a + r * b
         
         f1 = sign * obj(x1, *args)
         f2 = sign * obj(x2, *args)
         
         # Loop
         for i in range(numIter):
             
             if f1 > f2:
                 
                 a = x1
                 x1 = x2
                 f1 = f2
                 x2 = c * a + r * b
                 f2 = sign * obj(x2, *args)
                 
             else:
                 
                 b = x2
                 x2 = x1
                 f2 = f1
                 x1 = r * a + c * b
                 f1 = sign * obj(x1, *args)
                 
         if f1 < f2:
             
             return x1, sign * f1
         
         else:
             
             return x2, sign * f2

        
#---------------------------------------------------------------           
    def get_max_SR(self):
         # Get the max Sharpe ratio portfolio
         
         #1) Compute the local max SR portfolio between any two neighboring turning points
         w_sr, sr = [], []
         
         if len(self.w) > 1:
             
             for i in range(len(self.w) - 1):
                 
                 w0 = np.copy(self.w[i])
                 
                 w1 = np.copy(self.w[i + 1])
                 
                 kwargs = {'minimum':False,'args':(w0, w1)}
                 
                 alpha, b = self.golden_section(self.eval_SR, 0, 1, **kwargs)
                 
                 w_sr.append(alpha * w0 + (1 - alpha) * w1)  
                 
                 sr.append(b)
                 
         else:
             
             w = self.w[0]
             
             w_sr.append(w)
             
             b = (w.T @ self.mean)[0,0]/np.sqrt((w.T @ self.covar @ w)[0,0])
             
             sr.append(b)
                        
         # Get the index corresponding to the maximum sharpe ratio
         idx_max = np.argmax(sr)
         
         return sr[idx_max], w_sr[idx_max]


#---------------------------------------------------------------
    def eval_SR(self, alpha, w0, w1):
        
        # Evaluate SR of the portfolio within the convex combination
        w = alpha * w0 + (1 - alpha) * w1
     
        b = (w.T @ self.mean)[0,0]
     
        c = np.sqrt(((w.T @ self.covar) @ w)[0,0])
     
        return b/c
   
    
#---------------------------------------------------------------    
    def efficient_frontier(self, points):
        
        # Get the efficient frontier
        mu, sigma, weights = [], [], []
        
        # Increase points if too small
        if points < 3 * len(self.w):
            
            points = 3 * len(self.w)
        
        # Remove the 1 to avoid duplicates
        a = np.linspace(0, 1, int(points/len(self.w)))[:-1] 
        
        for i in range(len(self.w) - 1):
            
            w0, w1 = self.w[i], self.w[i + 1]
            
            # Include the 1 in the last iteration
            if i == len(self.w) - 2:
                
                a = np.linspace(0, 1, int(points/len(self.w)))
             
            # Calculate weights
            w_vals = [alpha * w1 + (1 - alpha) * w0 for alpha in a]
            
            # Extend list of weights
            weights.extend(w_vals)
            
            # Extend list of means
            mu.extend([(w.T @ self.mean)[0,0] for w in w_vals]) 
            
            # Extend list of stds
            sigma.extend([np.sqrt((w.T @ self.covar @ w)[0,0]) for w in w_vals])
            
        return mu, sigma, weights
  
    
#---------------------------------------------------------------
    def purge_numerical_error(self, tol):
        
        # Track number removed
        removed = 0
        
        # Purge violations of inequality constraints (associated with ill-conditioned covar matrix)       
        for i in range(len(self.w)):
                  
            w = self.w[i - removed]
            
            if np.any(w - self.lB < -tol) or np.any(w - self.uB > tol):
                
                del self.w[i - removed]
                del self.lam[i - removed]
                del self.g[i - removed]
                del self.f[i - removed]
                
                removed += 1
            
            
#---------------------------------------------------------------
    def purge_excess(self): 
        # Remove violations of the convex hull
        
        # Track number removed
        removed = 0
        
        for i in range(len(self.w) - 1):
            
            w = self.w[i - removed]
            
            mu = (w.T @ self.mean)[0,0]
            
            mu_next = [(self.w[j].T @ self.mean)[0,0] for j in range(i - removed + 1, len(self.w))]
            
            if mu < np.max(mu_next):
                    
                del self.w[i - removed]
                del self.lam[i - removed]
                del self.g[i - removed]
                del self.f[i - removed]
                
                removed += 1
            
        
#---------------------------------------------------------------
    def solve(self):
        
         # Compute the turning points,free sets and weights
         f, w = self.initialize_algo()
         
         # Store solution
         self.w.append(np.copy(w))
         self.lam.append(np.nan)
         self.g.append(np.nan)
         self.f.append(f[:])
         
         while True:
             
             #1) case a): Bound one free weight
             lam_in = np.nan
             
             if len(f) > 1:
                 
                 covarF, covarFB, meanF, wB = self.get_matrices(f)
                 
                 covarF_inv = np.linalg.inv(covarF)
                 
                 for j, i in enumerate(f):
                     
                    lam, bi = self.compute_lambda(covarF_inv, covarFB, meanF, 
                                                  wB, j, [self.lB[i], self.uB[i]])
                     
                    if lam > lam_in:
                         
                        lam_in, i_in, bi_in = lam, i, bi
                    
             #2) case b): Free one bounded weight
             lam_out = np.nan
             
             if len(f) < self.mean.shape[0]:
                 
                 b = self.get_b(f)
                 
                 for i in b:
                     
                     covarF, covarFB, meanF, wB = self.get_matrices(f + [i])
                     
                     covarF_inv = np.linalg.inv(covarF)
                     
                     lam, bi = self.compute_lambda(covarF_inv, covarFB, meanF, wB,
                                                meanF.shape[0] - 1, self.w[-1][i])
                     
                     if (self.lam[-1] is np.nan or lam < self.lam[-1]) and (lam_out is np.nan or lam > lam_out):
                         
                         lam_out, i_out = lam, i
              
             
             if (lam_in is np.nan or lam_in < 0) and (lam_out is np.nan or lam_out < 0):
                 
                 # 3) Compute minimum variance solution
                 self.lam.append(0)
                 
                 covarF, covarFB, meanF, wB = self.get_matrices(f)
                 
                 covarF_inv = np.linalg.inv(covarF)
                 
                 meanF = np.zeros(meanF.shape)
                 
             else:
                
                # 4) Decide lambda
                if lam_in > lam_out:
                    
                    self.lam.append(lam_in)
                    
                    f.remove(i_in)
                    
                    # Set value at the correct boundary
                    w[i_in] = bi_in
                    
                else:
                    
                    self.lam.append(lam_out)
                    
                    f.append(i_out)
                    
                covarF, covarFB, meanF, wB = self.get_matrices(f)
                
                covarF_inv = np.linalg.inv(covarF)
                
             # 5) Compute solution vector
             wF, g = self.compute_w(covarF_inv, covarFB, meanF, wB)
                    
             for i in range(len(f)):
                 
                 w[f[i]] = wF[i]
             
             # Store solution
             self.w.append(np.copy(w))
             self.g.append(g)
             self.f.append(f[:])
             
             if self.lam[-1] == 0:
                 
                 break
            
         # 6) Purge turning points
         self.purge_numerical_error(1e-9)
         self.purge_excess()
                 
 