import numpy as np 
import scipy.stats as stats


class Gibbs_Sampler(object):
    def __init__(self, data, n_components,tau, n_iter, burn_in, init_G, init_norm_G):
        self.n_components = n_components
        self.G_storage = np.repeat(np.expand_dims(init_G[:,0:self.n_components],2),n_iter,axis=2)
        self.norm_G_storage = np.repeat(np.expand_dims(init_norm_G[0:self.n_components,0:self.n_components],2),n_iter,axis=2)
        self.data = data
        self.G_interim = init_G
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
    def sample_parameters(self):
        for iter in range(self.n_iter+self.burn_in):
            self.norm_G_Update()
            self.G_Update()
            if iter%100==0:
                print(iter)
            if iter >= self.burn_in:
                self.norm_G_storage[:,:,iter-self.burn_in] = self.norm_G
                self.G_storage[:,:,iter-self.burn_in] = self.G# -*- coding: utf-8 -*-

cfrom math import isinf
import numpy as np

def DictParts(m, n):
    D = []
    Last = np.array([[0], [m], [m]])
    end = 0
    for i in range(n):
        NewLast = np.empty((3, 0), dtype=int)
        ncol = Last.shape[1]
        for j in range(ncol):
            record = Last[:, j]
            lack = record[1]
            l = min(lack, record[2])
            if l > 0 :
                D.append((record[0]+1, end+1))
                x = np.empty((3, l), dtype=int)
                for k in range(l):
                    x[:, k] = np.array([end+k+1, lack-k-1, k+1])
                NewLast = np.hstack((NewLast, x))
                end += l
        Last = NewLast
    return (dict(D), end)
   
def _N(dico, kappa):
    kappa = kappa[kappa > 0]
    l = len(kappa)
    if l == 0:
        return 1
    return dico[_N(dico, kappa[:(l-1)])] + kappa[-1] 

def _T(alpha, a, b, kappa):
    i = len(kappa)
    if i == 0 or kappa[0] == 0:
        return 1
    c = kappa[-1] - 1 - (i - 1) / alpha
    d = alpha * kappa[-1] - i
    s = np.asarray(range(1, kappa[-1]), dtype=int)
    e = d - alpha * s + np.array(
        list(map(lambda j: np.count_nonzero(kappa >= j), s))
    )
    g = e + 1
    ss = range(i-1)
    f = alpha * kappa[ss] - (np.asarray(ss) + 1) - d
    h = alpha + f
    l = h * f
    prod1 = np.prod(a + c)
    prod2 = np.prod(b + c)
    prod3 = np.prod((g - alpha) * e / (g * (e + alpha)))
    prod4 = np.prod((l - f) / (l + h))
    out = prod1 / prod2 * prod3 * prod4
    return 0 if isinf(out) or np.isnan(out) else out


def betaratio(kappa, mu, k, alpha):
    muk = mu[k-1]
    t = k - alpha * muk
    prod1 = prod2 = prod3 = 1
    if k > 0:
        u = np.array(
            list(map(lambda i: t + 1 - i + alpha * kappa[i-1], range(1,k+1)))
        )
        prod1 = np.prod(u / (u + alpha - 1))
    if k > 1:
        v = np.array(
            list(map(lambda i: t - i + alpha * mu[i-1], range(1, k)))
        )
        prod2 = np.prod((v + alpha) / v)
    if muk > 1:
        muPrime = dualPartition(mu)
        w = np.array(
            list(map(lambda i: muPrime[i-1] - t - alpha * i, range(1, muk)))
        )
        prod3 = np.prod((w + alpha) / w)
    return alpha * prod1 * prod2 * prod3

def dualPartition(kappa):
    out = []
    if len(kappa) > 0 and kappa[0] > 0:
        for i in range(1, kappa[0]+1):
            out.append(np.count_nonzero(kappa >= i))
    return np.asarray(out, dtype=int)


def hypergeomI(m, alpha, a, b, n, x):
    def summation(i, z, j, kappa):
        def go(kappai, zz, s):
            if i == 0 and kappai > j or i > 0 and kappai > min(kappa[i-1], j):
                return s
            kappap = np.vstack((kappa, [kappai]))
            t = _T(alpha, a, b, kappap[kappap > 0])
            zp = zz * x * (n - i + alpha * (kappai -1)) * t
            sp = s
            if j > kappai and i <= n:
                sp += summation(i+1, zp, j - kappai, kappap)
            spp = sp + zp
            return go(kappai + 1, zp, spp)
        return go(1, z, 0)
    return 1 + summation(0, 1, m, np.empty((0,1), dtype=int))

def is_square_matrix(x):
    x = np.asarray(x)
    return x.ndim == 2 and x.shape[0] == x.shape[1]

def hypergeomPQ(m, a, b, x, alpha=2):
    """Hypergeometric function of a matrix argument.
    
    :param m: truncation weight of the summation, a positive integer
    :param a: the "upper" parameters, a numeric or complex vector, possibly empty (or `None`)
    :param b: the "lower" parameters, a numeric or complex vector, possibly empty (or `None`)
    :param x: a numeric or complex vector, the eigenvalues of the matrix
    :param alpha: the alpha parameter, a positive number
    
    """
    if a is None or len(a) == 0:
        a = np.array([])
    else:
        a = np.asarray(a)
    if b is None or len(b) == 0:
        b = np.array([])
    else:
        b = np.asarray(b)
    x = np.asarray(x)
    n = len(x)
    if all(x == x[0]):
        return hypergeomI(m, alpha, a, b, n, x[0])
    def jack(k, beta, c, t, mu, jarray, kappa, nkappa):
        lmu = len(mu)
        for i in range(max(1, k), (np.count_nonzero(mu)+1)):
            u = mu[i-1]
            if lmu == i or u > mu[i]:
                gamma = beta * betaratio(kappa, mu, i, alpha)
                mup = mu.copy()
                mup[i-1] = u - 1
                mup = mup[mup > 0]
                if len(mup) >= i and u > 1:
                    jack(i, gamma, c + 1, t, mup, jarray, kappa, nkappa)
                else:
                    if nkappa > 1:
                        if len(mup) > 0:
                            jarray[nkappa-1, t-1] += (
                                gamma * jarray[_N(dico, mup)-2, t-2] 
                                * x[t-1]**(c+1)
                            )
                        else:
                            jarray[nkappa-1, t-1] += gamma * x[t-1]**(c+1)
        if k == 0:
            if nkappa > 1:
                jarray[nkappa-1, t-1] += jarray[nkappa-1, t-2]
        else:
            jarray[nkappa-1, t-1] += (
                beta * x[t-1]**c * jarray[_N(dico, mu)-2, t-2]
            )
    def summation(i, z, j, kappa, jarray):
        def go(kappai, zp, s):
            if (
                    i == n or i == 0 and kappai > j 
                    or i > 0 and kappai > min(kappa[-1], j)
                ):
                return s
            kappap = np.concatenate((kappa, [kappai]))
            nkappa = _N(dico, kappap) - 1
            zpp = zp * _T(alpha, a, b, kappap)
            if nkappa > 1 and (len(kappap) == 1 or kappap[1] == 0):
                 jarray[nkappa-1, 0] = (
                     x[0] * (1 + alpha * (kappap[0] - 1)) * jarray[nkappa-2, 0]
                 )
            for t in range(2, n+1):
                jack(0, 1.0, 0, t, kappap, jarray, kappap, nkappa)
            sp = s + zpp * jarray[nkappa-1, n-1]
            if j > kappai and i <= n:
                spp = summation(i+1, zpp, j-kappai, kappap, jarray)
                return go(kappai+1, zpp, sp + spp)
            return go(kappai+1, zpp, sp)
        return go(1, z, 0)
    (dico, Pmn) = DictParts(m, n)
    T = type(x[0])
    J = np.zeros((Pmn, n), dtype=T)
    J[0, :] = np.cumsum(x)
    return 1 + summation(0, T(1), m, np.empty(0, dtype=int), J)