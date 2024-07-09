import numpy as np 
import scipy.stats as stats


class Gibbs_Sampler(object):
    def __init__(self, data, n_components,tau, n_iter, burn_in, init_G, init_norm_G):
        self.n_components = n_components
        self.G_storage = np.repeat(np.expand_dims(init_G[:,0:self.n_components],2),n_iter,axis=2)
        self.norm_G_storage = np.repeat(np.expand_dims(init_norm_G[0:self.n_components,0:self.n_components],2),n_iter,axis=2)
        self.data = data
        self.G_interim= init_G
        self.G = init_G[:,0:self.n_components]
        self.norm_G_interim = init_norm_G
        self.norm_G = init_norm_G[0:self.n_components,0:self.n_components]
        self.n = data.shape[1]
        self.n_iter = n_iter
        self.burn_in = burn_in
        self.sort = []
        self.tau = tau
    def norm_G_Update(self):
        scale = 1/(.5 + .5*np.sum(np.power(np.divide(np.dot(self.data, self.G_interim),np.diag(np.dot(self.G_interim.T,self.G_interim))),2),axis=0))
        shape = self.n_components/2 +self.n/2
        rvs= stats.gamma.rvs(shape, scale)
        self.sort = np.argsort(rvs)[::-1]
        self.norm_G_interim = np.diag(rvs[self.sort])
        self.norm_G =self.norm_G_interim[0:self.n_components,0:self.n_components]
            
    def G_Update(self):
        xTx = np.dot(self.data[:,self.sort].T,self.data[:,self.sort])
        xTx = xTx[0:self.n_components,0:self.n_components]
        accept= False
        count = 0
        while accept==False:
            count+=1
            cutoff= np.log(np.random.uniform(size=1))

            self.hold_G = stats.matrix_normal.rvs(self.G, np.linalg.inv(xTx), np.linalg.inv(self.norm_G_interim[0:self.n_components,0:self.n_components]))
            mean_G = self.hold_G
            for i in range(1, self.n_components):
                projection =np.dot(mean_G[:,i],np.squeeze(mean_G[:,0:i]))
                norm = np.array(np.dot(np.squeeze(mean_G[:,0:i].T),np.squeeze(mean_G[:,0:i])))         
                if projection.shape != () and projection.shape != (0,):
                    projection = np.diag(projection)
                if norm.shape != () and norm.shape != (0,):
                    norm= np.diag(np.diag(1/np.array(norm)))
                    projectionnorm = np.matmul(projection,norm)
                else:
                    projectionnorm = projection/norm
                test1 = mean_G[:,0:i]
                test2 = projection
                test3 = norm
                if i >= 2:
                    mean_G[:,i] = mean_G[:,i] - np.sum(np.matmul(test1,projectionnorm),1)
                elif i==1:
                    mean_G[:,i] = np.squeeze(mean_G[:,i]) - np.squeeze(test1)*projectionnorm
            norm = np.diag(np.array(np.dot(np.squeeze(mean_G[:,:].T),np.squeeze(mean_G[:,:]))))
            for i in range(self.n_components):
                mean_G[:,i] = mean_G[:,i]/np.linalg.norm(mean_G[:,i])
            upper=(np.matrix.trace(np.matmul(np.matmul(np.matmul(xTx,mean_G),self.norm_G[0:self.n_components,0:self.n_components]),mean_G.T))-np.matrix.trace(np.matmul(np.matmul(np.matmul(xTx,self.G,),self.norm_G[0:self.n_components,0:self.n_components]),self.G.T)))
            if cutoff <  upper:
                accept = True
        self.G = mean_G[:,0:self.n_components]
    def _sample_parameters(self):
        for iter in range(self.n_iter+self.burn_in):
            self.norm_G_Update()
            self.G_Update()
            if iter%100==0:
                print(iter)
            if iter >= self.burn_in:
                self.norm_G_storage[:,:,iter-self.burn_in] = self.norm_G
                self.G_storage[:,:,iter-self.burn_in] = self.G