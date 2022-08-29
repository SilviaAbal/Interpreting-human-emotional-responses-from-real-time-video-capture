#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 13:28:12 2022

@author: sabal

This script implements a weak loss function that applies constraints on the cumulative probability 
that an object is the "active object" in all frames of a video clip. 

"""

# --------------------------------------------------------
# CCNN 
# Copyright (c) 2015 [See LICENSE file for details]
# Written by Deepak Pathak, Philipp Krahenbuhl
# --------------------------------------------------------
import pdb
import numpy as np
import sklearn.metrics as sklm
import sys
import torch
from torch import Tensor, einsum
from torch.nn import functional as F

class WeakLoss():

    def __init__(self,bounds,fg_slack=1000.0,bg_slack=1000.0):

        print(f"Initialized {self.__class__.__name__} with {bounds}")

        self.bg_lower,self.bg_upper = 0.3,0.7
        self.bg_slack = bg_slack 		# slack : 5.0
        self.fg_slack = fg_slack #3.0 		# no slack : 1e10
        self.fg_lower = 0.05
        self.cfg_upper = 0.15
        self.neg_upper = 0.01
        self.hardness = 1 				# no hardness : 1 and hardness : 1000
        self.comp_slack = False;
        self.semi_supervised = False
        self.normalization = False  # models/fcn_8s/solver_8s.prototxt needs the loss to be normalized and solver_32s doesn't
        self.itersInitialization = 0 #4620;
        self.counter = 0
        self.video_idx = [];
        self.rejection_prob=1.00;
        self.zero_constraints=[]
        self.linear_constraints=[]
        self.iter=0
        self.t_bg_slack = self.bg_slack
        self.t_fg_slack = self.fg_slack
        
		
        #Set the bounds
        boundsnp=bounds.clone().detach()
        self.neg_upper = boundsnp[0,1]
        self.fg_lower = boundsnp[1,0]
        self.fg_upper = boundsnp[1,1]
        
        
    def __call__(self, scores: Tensor, target: Tensor, weights: Tensor) -> Tensor:
        
        self.iter +=1
        V, D = scores.shape
        #Compute probabilities q => IMP: This is the variable to be backwarded
        q = F.softmax(scores,dim=1)
        p = torch.zeros_like(q,device=q.device,requires_grad=False)
        loss = 0
        
        # Run constrained optimization and obtain the variational p => No grads during this code!!
        with torch.no_grad():
            for v in range(V):
                p[v,...] = self.computeP(scores[v,...], q[v,...], target[v], weights[v])
                weights[v]=weights[v]/weights[v].sum()
                p[v,...]= (p[v,...].t()*weights[v]).t()
        
        loss = -torch.sum(p*torch.log(q+1e-10))

        return loss
    
    
    def computeP(self, scores: Tensor, q: Tensor, target: Tensor, weights: Tensor) -> Tensor:
        self.linear_constraints=[]
        self.zero_constraints=[]
        D = len(scores)
        L = np.zeros((D,),dtype=np.int)

        L[target.cpu().numpy().astype(np.int)] = 1 
        numPos=L[0:].sum()
        
        #Set weights 
        w=weights.clone().detach()
        if(w.dim()>1):
            w=w[0,...]
            
        if w.shape[0]!=q.shape[0]:
            nBBs=int(q.shape[0]/w.shape[0])
            w = w.repeat(nBBs)
            #Normalization w sums to the number of samples
            w = w.shape[0]*w/w.sum();
        
 	    
        #Set the background to positive           
        #Compute the slack variable if necessary
        
        self.counter += 1
        if self.comp_slack and self.counter>self.itersInitialization:
            self.compute_slack(q,L)
        else:
            self.t_fg_slack=self.fg_slack
            self.t_bg_slack=self.bg_slack
           
            
            #Background constraints (lower and upper bounds)
            v = torch.zeros(D,device=scores.device); v[0] = 1
                  
            #Object constraints                          
            for l in range(0,D):
                #NEGATIVE CONSTRAINTS
                if L[l]==0:
                    v = torch.zeros(D,device=scores.device); v[l] = 1 #Array all zeroes but the selected category
                    self.addLinearConstraint( -v,w,-self.neg_upper, self.t_bg_slack )
                if L[l]==1:
                    #POSITIVE CONSTRAINTS (Lower bound)
                    v = torch.zeros(D,device=scores.device); v[l] = 1 #Array all zeroes but the selected category
                    self.addLinearConstraint(  v, w,self.fg_lower/numPos, self.t_fg_slack ) #We add a lower bound

        p = self.compute(scores)
    
        return p
         

    def compute (self, f : Tensor) -> Tensor:
        p=self.computeLog(f)
        p=F.softmax(self.hardness*p, dim=0)
        return p;
         
    def computeLog(self, f : Tensor) -> Tensor:
        #print("Starting optimization")
        M=len(f);
        N=len(f);
        K=len(self.linear_constraints)

        #print('M %f N %f K %f'%(M,N,K));
        A=torch.zeros(K,M,device=f.device,requires_grad=False)
        W=torch.zeros(1,K,device=f.device,requires_grad=False)
        b=torch.zeros(K,device=f.device,requires_grad=False)
        slack=torch.zeros(K,device=f.device,requires_grad=False)
        
       
        for i in range(K):
            A[i,:] = self.linear_constraints[i]['a'] # Normalize by spatial_dim (no change theoretically, for implementation stability)
            b[i] = self.linear_constraints[i]['b']
            slack[i] = self.linear_constraints[i]['slack'] * N; #Scale regularizer of slack according to spatial_dim
        W[0,:] = self.linear_constraints[0]['w']/self.linear_constraints[0]['w'].sum() 
        lambdap=self.optimize(f,A,W,b,slack)
        
        Ma=lambdap*torch.ones_like(b);

        p = f + Ma@A;
        
        return p
    
    def computeLogFast(self, f : Tensor) -> Tensor:
        #print("Starting optimization")
        N,M=f.shape;
        K=len(self.linear_constraints)
        pM=M
        pf=f
        if self.zero_constraints.any():
            #We count the number of categories that appear (pM number of positive categories)
            pM = (self.zero_constraints==0).sum()
            #If we have one or less than one categories (all pixels belong to the same category)
            if pM <= 1:
                idx=np.argwhere(self.zero_constraints)
                #the original prob for the positive categories
                p=-1e10*torch.ones_like(f,device=f.device,requires_grad=False)
                p[:,idx]=0
                return p
            
            #Build the projection matrix
            P = torch.zeros(M,pM,device=f.device,requires_grad=False)
            k=0
            for i in range(M):
                # If the category i is present
                if self.zero_constraints[i]==0:
                    P[i,k] = 1; #This is a shift matrix to generate a new one with just the positive categories
                    k+=1
            
        
            #Project onto the matrix : Means remove the variables with zero equality constraints
            pf = f @ P #This is the new matrix without the negative categories
            for i in range(K):
               self.linear_constraints[i]['a']=P.t() @ self.linear_constraints[i]['a']
               
        
        #print('M %f N %f K %f'%(M,N,K));
        A=torch.zeros(K,pM,device=f.device,requires_grad=False)
        W=torch.zeros(N,K,device=f.device,requires_grad=False)
        b=torch.zeros(K,device=f.device,requires_grad=False)
        slack=torch.zeros(K,device=f.device,requires_grad=False)
        
       
        for i in range(K):
            A[i,:] = self.linear_constraints[i]['a'] # Normalize by spatial_dim (no change theoretically, for implementation stability)
            W[:,i] = self.linear_constraints[i]['w']/self.linear_constraints[i]['w'].sum() # Normalizamos w, que es equivalente a haber normalizado A dividiendo entre el número de datos 		
            b[i] = self.linear_constraints[i]['b']
            slack[i] = self.linear_constraints[i]['slack'] * N; #Scale regularizer of slack according to spatial_dim
        
        lambdap=self.optimize(pf,A,W,b,slack)
        
        Ma=lambdap*torch.ones_like(W);
        p = pf + Ma@A;
        
        
        #Construct the result adding the negatives
        if self.zero_constraints.any():
            p = p @ P.t();
            for i in range(M):
                if self.zero_constraints[i]:
                    p[:,i]=-1e10
		
        
        return p
    
    #Function that implements the optimization
    def optimize(self, f: Tensor, A: Tensor,W: Tensor, b: Tensor,slack: Tensor) -> Tensor:
        N_ITER = 3000
        beta = 0.5
        alpha = 10.0; #Learning decreasing factor
        # THlikelihood = 1e-10
        lambdap = torch.zeros_like(b,requires_grad=False,device=f.device)
        grad = torch.zeros_like(A,requires_grad=False,device=f.device)
        
        k=1
        # bestk=1;
        prev_likelihood,grad = self.compute_likelihood_grad(lambdap,f,A,W,b,slack,grad);
        #print("iter %d likelihood %f alpha %f lambda %f"%(k,prev_likelihood,alpha,lambdap.max()))   
        
        while k<N_ITER and alpha>1e-8:
            k=k+1
            #fx es lo que devuelve, ns se le pasaba por parámetro 
            new_lambdap=self.projection(lambdap+alpha*grad,slack)
            likelihood,new_grad = self.compute_likelihood_grad(new_lambdap,f,A,W,b,slack,grad)
            if( likelihood > prev_likelihood ):
                lambdap = new_lambdap;
                #print(new_lambdap)
                grad = new_grad

                    
                prev_likelihood = likelihood
                alpha = alpha*1.1
                
            else:
                alpha = alpha*beta;

        return lambdap;
    
    def projection(self, x: Tensor, slack: Tensor) -> Tensor:
        x=torch.max(torch.zeros_like(x,device=x.device),x);
        x=torch.min(x,slack);
        return x

    
    def logSumExp(self, m: Tensor ) -> Tensor :
        maxVal = m.max();
        return maxVal + torch.log(torch.exp(m-maxVal).sum(dim=1))

    def compute_likelihood_grad (self, lambdap: Tensor,f : Tensor,A : Tensor, W : Tensor ,b : Tensor ,slack: Tensor, g: Tensor)-> Tensor:
        Ma=lambdap*torch.ones_like(W);
        Ma = self.hardness*(f + Ma@A);
        p = F.softmax( Ma,dim=1 );
        g =  b-((W.t() @ p )*A).sum(dim=1)

        likelihood= lambdap @ b - (1.0/self.hardness)*W[...,0]@self.logSumExp(Ma)
        return (likelihood,g)
            				
    
    def backward(self, top, propagate_down, bottom):
        if (not self.semi_supervised):
            bottom[0].diff[...] = top[0].diff[0,0,0,0]*self.diff
        else:
            bottom[5].diff[...] = top[0].diff[0,0,0,0]*self.diff
            
    def addZeroConstraint(self, a : Tensor):
    
        if not self.zero_constraints:
            self.zero_constraints=torch.tensor(a>0);
        else:
            self.zero_constraints = self.zero_constraints or (a > 0);

    def addLinearConstraint(self, a : Tensor, w : Tensor, b : Tensor, slack : Tensor):
        lc= {'a'  :a, 'b': b, 'w': w, 'slack': slack}
        self.linear_constraints.append(lc)
        
    def compute_slack(self, q, L):
        acc_probs=np.mean(q,axis=0)
        #include also the BG
        L[0]=0;
        max_prob=np.max(acc_probs[~L])-np.max(acc_probs[L])
        if (max_prob>self.rejection_prob):
            self.t_fg_slack=self.fg_slack*np.maximum(1-max_prob/0.20,0)
            self.t_bg_slack=self.bg_slack;#(1-max_prob)
            sys.stdout.write('%d '%(self.video_idx[0]+1))
            print('%f %.2f %.2f'%(max_prob,self.t_fg_slack,self.t_bg_slack))
        else:
            self.t_fg_slack=self.fg_slack
            self.t_bg_slack=self.bg_slack
            L[0]=1;
