def kaiser_bessel(x,J):# parameters were removed for code to run
    
    J=6
    alpha = 2.34*J
    kb_m = 0
    
    kernel_string = "ans(k,J) = kaiser_bessel(k,J,"+str(alpha)+","+str(kb_m)+")"
    return kernel_string, alpha, kb_m
    

def reale(x):
    import numpy as np
    com = "error"
    tol = 1*10**-13
    onlywarn = 0
    frac = np.max(np.abs(np.imag(x[:])))/np.max(np.abs(x[:]))
    y = np.real(x)
    return y

def kaiser_bessel_ft(u,J,alpha,kb_m,d):
    import numpy as np
    import math
    from mpmath import besselj, besseli 
    import cmath

    z = map(cmath.sqrt,( (2*math.pi*(J/2)*u)**2 - alpha**2 ))
    #print(u[0])
    
    
    nu = d/2.0 + kb_m
    
    a = np.shape(z)[0]
    bslj = np.zeros(a,np.complex128)
    for i in range(0,a):
        bslj[i] = np.array(np.complex(besselj(nu,z[i])))

    z_sqrt = map(cmath.sqrt, z)
    
    

    a = (2*math.pi)**(d/2.0)*(J/2)**d*alpha**kb_m
    b = a/float(besseli(kb_m,alpha))*bslj
    y = b/(z_sqrt)
        
    y = reale(y)
    
    return y


def block_outer_sum(x1,x2):
    import numpy as np
    
    [J1, M] = np.shape(x1)
    [J2, M] = np.shape(x2)
    xx1= np.reshape(x1, (J1,1,M))
    xx1 = np.tile(xx1,(1,J2,1))#xx1[:,np.ones(J2,1),:]
    xx2 = np.reshape(x2, (1,J2,M))
    xx2 = np.tile(xx2,(J1,1,1)) #xx2[np.ones(J1,1),:,:]
    y = xx1+xx2
    return y

def block_outer_prod(x1,x2):
    import numpy as np
    [J1, M] = np.shape(x1)
    [J2, M] = np.shape(x2)
    xx1 = np.reshape(x1,(J1,1,M))
    xx1 = np.tile(xx1,(1,J2,1))#xx1[:, np.ones(J2,1),:]
    xx2 = np.reshape(x2,(1,J2,M))
    xx2 = np.tile(xx2,(J1,1,1))#xx2[np.ones(J1,1),:,:]
    y = np.multiply(xx1,xx2)

    return y


def nufft_offset(om,J,K):
    import numpy as np
    import math

    k0 = np.floor(om/(2*math.pi/K)) - J/2
    return k0

def outer_sum(xx,yy):
    import numpy as np
    import numpy.matlib
    xx = xx[:]
    yy = yy[:]
    nx = len(xx)
    ny = len(yy)
    xx = np.matlib.repmat(xx,ny,1)
    xx = np.transpose(xx)
    yy = np.matlib.repmat(yy,nx,1)
    ss = xx + yy
    return ss

def inlineeval(INLINE_INPUTS_ , INLINE_INPUTEXPR_ , INLINE_EXPR_):
    import numpy as np
    from mpmath import besseli
    INLINE_OUT_ = []
    print(INLINE_INPUTS_)
    k = INLINE_INPUTS_[0]
    print(k[1])
    J = INLINE_INPUTS_[1]
    kb_m = 0
    alpha = 14.04
    kb_m_bi = abs(kb_m)
    ii = np.abs(k) < J/2
    f = np.sqrt(1-(k[ii]/(J/2))**2)
    f = np.expand_dims(f,axis=1)
    
    denom = besseli(kb_m_bi, alpha)
    kb = np.zeros(np.shape(f))
    
    for j in range(0,len(kb)):
        kb[j] = np.float(besseli(kb_m_bi, alpha*f[j][0]))
    
    kb_1 = np.zeros(np.shape(k)).flatten()
   
    ii = ii.flatten()
    a = ((f**kb_m)*kb)/np.float(denom)
    d = 0
    for u in range(0,len(ii)):
        if ii[u] != 0:
            kb_1[u] = a[d]
            d = d+1


    kb = np.reshape(kb_1, np.shape(k))

    return kb




def feval(*args):
    INLINE_OBJ_ = args[0]
    INLINE_INPUTS_ = args[1:3]
    
    a = "k = INLINE_INPUTS_[0], J = INLINE_INPUTS_[1]"
    INLINE_OUT_ = inlineeval(INLINE_INPUTS_ , a, INLINE_OBJ_)#write function
    b = INLINE_OUT_
    return b

def nufft_coef(om,J,K,kernel):
    import numpy as np
    import math

    M = len(om)
    gam = 2*math.pi/K
    dk = om/gam - nufft_offset(om,J,K)
    print(dk[0])
    arg = outer_sum(-np.transpose(np.arange(1,J+1,1)), np.transpose(dk))
    coef= feval(kernel, arg, J)#write function
    return coef, arg

def nufft_init(om,Nd, Jd,Kd, n_shift, kernel):
    import numpy as np
    import math
    import numpy.matlib
    from scipy.sparse import csr_matrix

    st = {}
    st["n_shift"]=n_shift
    st["ktype"] = kernel
    st["alpha"] = {}
    st["beta"] = {}
    is_kaiser_scale = 1
    st["kernel"] = {}
    st["kb_afl"] = {}
    st["kb_m"] = {}
    for id in range(0,2):
        [st["kernel"][id], st["kb_afl"][id], st["kb_m"][id]] = kaiser_bessel("inline", Jd[id])#write the function
    
    st["tol"] = 0
    st["Jd"] = Jd
    st["Nd"] = Nd
    M = np.shape(om)[0]
    st["M"] = M
    st["om"] = om
    st["Kd"] = Kd[0]
    st["sn"] = 1
    for id in range(0,2):
        if is_kaiser_scale:
            nc = np.arange(0,Nd[id],1.0) - ((Nd[id] -1)/2.0000)
            print(nc)
            
            tmp = 1/kaiser_bessel_ft(nc/Kd[0][id], Jd[id], st["kb_afl"][id], st["kb_m"][id],1)#write the function
        print(tmp[0])
        print(np.shape(tmp))
        print(0)
        if id == 0: 
            st["sn"] = st["sn"]*np.transpose(tmp)
            
        elif id == 1:
            st["sn"] = np.expand_dims(st["sn"],axis=1)
            tmp = np.expand_dims(tmp,axis=1)
            st["sn"] = st["sn"]*np.transpose(tmp)
            

    
    if (len(Nd))> 1:
        st["sn"] = np.reshape(st["sn"],Nd)
    ud = {}
    kd = {}
    for id in range(0,2):
        N = Nd[id]
        J = Jd[id]
        K = Kd[0][id]
        
        [c, arg] = nufft_coef(om[:,id], J, K,st["kernel"][id])#write the function

        gam = 2*math.pi/K
        phase_scale = 1j*gam*(N-1)/2
        phase = np.exp(phase_scale*arg)
    
        
        ud[id] = np.multiply(phase,c)
        

        koff = nufft_offset(om[:,id], J,K)#write the function
        koff = np.expand_dims(koff, axis=1)
        
        kd[id] = np.mod(outer_sum([np.transpose(np.arange(1,J+1))], np.transpose(koff)),K)+1
        
        if id > 0:
            kd[id] = (kd[id]-1)*np.prod(Kd[0][0:(id)])
        
            

    kk= kd[0]
    uu = ud[0]
    
    for id in range(2,3):
        Jprod = np.prod(Jd[0:id])
        kk = block_outer_sum(kk, kd[id-1])
        kk = np.reshape(np.swapaxes(kk,0,1), (Jprod, M))
        uu = block_outer_prod(uu, ud[id-1])
        uu = np.reshape(np.swapaxes(uu, 0, 1), (Jprod, M))
    
    
    
    
     
    n_shift = np.expand_dims(n_shift, axis=1)
    phase = np.transpose(np.exp(1j*(np.dot(om,n_shift[:]))))
    phase = np.matlib.repmat(phase, np.prod(Jd), 1)
    uu = np.multiply(np.conj(uu), phase)
    mm = np.arange(0,M,1)

    mm = np.matlib.repmat(mm, np.prod(Jd), 1)
    
    mm_1 = np.reshape(np.swapaxes(mm,0,1), (np.shape(mm)[0]*np.shape(mm)[1],1))
    mm_1 = np.asarray(mm_1, dtype=int)
    mm_1 = np.squeeze(mm_1)
    kk_1 = np.reshape(np.swapaxes(kk,0,1), (np.shape(kk)[0]*np.shape(kk)[1],1))
    kk_1 = np.asarray(kk_1, dtype=int)
    kk_1 = np.squeeze(kk_1)-1
    uu_1 = np.reshape(np.swapaxes(uu,0,1), (np.shape(uu)[0]*np.shape(uu)[1],1))
    uu_1 = np.squeeze(uu_1)

    st["p"] = csr_matrix((uu_1,(mm_1,kk_1)), shape=(M,int(np.prod(Kd))))
    return st

def MCNUFFT(k,w,b1):
    #Multicoil NUFFT operator
    #based on the NUFFT toolbox from Jeff Fessler
    #Input
    #k:k-space trajectory
    #w:density compensation
    #b1:coil sensitivity maps
    #
    import numpy as np
    import math
    Nd=np.shape(b1[:,:,1])
    Jd = [6,6]
    Kd= np.array(np.floor([np.multiply(Nd,1.5)]))
    
    n_shift=np.divide(Nd,2)

    res = {}
    res["st"] = {}
    
    for tt in range(0,np.shape(k)[2]):

        kk = k[:,:,tt]
        kk_real = np.real(np.reshape(np.swapaxes(kk,0,1),(np.shape(kk)[0]*np.shape(kk)[1],1)))
    
        kk_imag = np.imag(np.reshape(np.swapaxes(kk,0,1),(np.shape(kk)[0]*np.shape(kk)[1],1)))
    
        kk_concat = np.concatenate((kk_real,kk_imag),axis=1)
    
        om = kk_concat*2*math.pi
        

        res["st"][tt] = nufft_init(om,Nd,Jd,Kd,n_shift, "kaiser") #how to implement it in python
    
    res["adjoint"] = 0
    res["imSize"] = np.shape(b1[:,:,1])
    res["imSize2"] = [np.shape(k)[0],np.shape(k)[0]]
    res["dataSize"] = np.shape(k)
    res["w"] = np.sqrt(w)
    res["b1"] = b1

    return res

def col(x):
    import numpy as np
    x = np.reshape(np.swapaxes(x,0,1),(np.shape(x)[0]*np.shape(x)[1],1))
    return x

def nufft_adj(X,st):
    # extract attributes from structure
    import numpy as np

    Nd = st["Nd"]
    Kd = st["Kd"]
    print(Kd)
    dims = np.shape(X)
    Lprod = 1
    Xk_all = np.transpose(st["p"])*X
    
    

    x = np.zeros((int(np.prod(Kd)), Lprod))
    
    for ll in range(0,Lprod):
        Xk = np.reshape(Xk_all[:,ll],(Kd.astype(int)))
        x = col(np.prod(Kd)*np.fft.ifftn(Xk)) # write the col() function


    x = np.reshape(x, (Kd.astype(int)))
    import matplotlib 
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    #plt.imshow(np.abs(x),cmap="gray")
    #plt.savefig("Full_image.png")
    
    x = x[0:Nd[0],0:Nd[1]]

    x = np.reshape(x, (int(np.prod(Nd)), Lprod))
    # Fix st["sn"]
    snc = np.conj(col(st["sn"]))
    print(snc[1]) 
    snc = np.matlib.repmat(snc, 1,Lprod)
    
    
    x = x*snc

    x = np.reshape(x, (Nd[0], Nd[1]))
    #plt.imshow(np.abs(x),cmap="gray")
    #plt.savefig("Full_image.png")

    
    return x

def nufft(x, st):
    import numpy as np
    Nd = st["Nd"]
    Kd = st["Kd"]
    dims = np.shape(x)
    dd = len(Nd)
    if len(np.shape(x)) == dd:
        x = x*st["sn"]
        Xk = col(np.fft.fftn(x,Kd.astype(int)))
        
    X = st["p"]*Xk
    return X

def mtimes(a,bb):
    if a["adjoint"]==0:
        #Multicoil non-Cartesian k-space to Cartesian image domain nufft for each coil and time point
        res = np.zeros((a["imSize"][0],a["imSize"][1],np.shape(bb)[2]), dtype = np.complex128)
        
        ress = np.zeros((a["imSize"][0],a["imSize"][1], np.shape(bb)[3]),dtype = np.complex128)
        
        for tt in range(0, np.shape(bb)[3]):
            for ch in range(0, np.shape(bb)[2]):
                    b = bb[:,:,ch,tt]*a["w"][:,:,tt]
                    b = np.reshape(np.swapaxes(b,0,1), (np.shape(b)[0]*np.shape(b)[1],1))
                    
                    
                    res[:,:,ch] = np.reshape(nufft_adj(b,a["st"][tt])/np.sqrt(np.prod(a["imSize2"])), (a["imSize"][0], a["imSize"][1]))
                    if ch == 2:
                        import matplotlib
                        matplotlib.use("Agg")
                        import matplotlib.pyplot as plt
                        plt.imshow(np.abs(res[:,:,ch]),cmap="gray")
                        plt.savefig("NUFFT_recon_test.png")

            ress[:,:,tt] = np.sum(res*np.conj(a["b1"]), axis=2)/np.sum(np.abs((np.squeeze(a["b1"])))**2,axis=2)
        
        ress = ress*np.shape(a["w"])[0]*math.pi/2/np.shape(a["w"])[1]
        
    else:
        print(np.shape(bb))
        res = np.zeros((a["imSize"][0],a["imSize"][1]), dtype = np.complex128)
        ress = np.zeros((a["dataSize"][0],a["dataSize"][1],np.shape(a["b1"])[2],np.shape(bb)[2]), dtype = np.complex128)
        print(np.shape(ress))
                
        for tt in range(0,np.shape(bb)[2]):
            for ch in range(0,np.shape(a["b1"])[2]):
                res = bb[:,:,tt]*a["b1"][:,:,ch]
                ress[:,:,ch,tt] = np.reshape(nufft(res,a["st"][tt])/np.sqrt(np.prod(a["imSize2"])), (a["dataSize"][0],a["dataSize"][1]))*a["w"][:,:,tt]
    return ress

def mtimes_2(a,b):
    #res = b[:,:,[np.arange(1,31,1),31]] - b
    print(np.shape(np.arange(1,31,1)))


def objective(x,dx,t,param):
    import numpy as np

    #L2-norm part
    w=param["E"]*(x+t*dx)-param["y"]
    L2Obj = np.transpose(w.flatten())*w.flatten()
    
    # TV part along time
    if param["TVWeight_dim"]:
        w=param["TV_dim1"]*(x+t*dx)
        TVObj_dim1=np.sum((np.multiply(w.flatten(),np.conj(w.flatten()))+param["l1Smooth"])**(1/2))
    else:
        TVObj_dim1 = 0
    

    #TV part along respiration
    if param["TVWeight_dim2"]:
        w = param["TV_dim2"]*(x+t*dx)
        TVObj_dim2 = np.sum((np.multiply(w[:],np.conj(w[:]))+param["l1Smooth"])**(1/2))
    else:
        TVObj_dim2 = 0
    

    res = L2Obj + param["TVWeight_dim1"]*TVObj_dim1+param["TVWeight_dim2"]*TVObj_dim2
    return res
def grad(x,param):

    #L2-norm part
    param["E"]["adjoint"] = -1
    MC_kspace = mtimes(param["E"], x)

    print(MC_kspace[100,29,8,8])
    param["E"]["adjoint"] = 0
    L2Grad = mtimes(param["E"],MC_kspace)

    #TV part along time
    if param["TVWeight_dim1"] != 0:
        w = mtimes_2(param, x)
        TVGrand_dim1 = np.transpose(param["TV_dim1"])*(np.multiply(w,(np.multiply(w,np.conj(w))+param["l1Smooth"])**(-0.5)))
    else:
        TVGrad_dim1=0
    

    #TV part along respiration
    if param["TVWeight_dim2"]:
        w = param["TV_dim2"]*x
        TVGrad_dim2 = np.transpose(param["TV_dim2"])*(np.multiply(w,(np.multiply(w,np.conj(w))+param["l1Smooth"])**(-0.5)))
    else:
        TVGrad_dim2=0


    g=L2Grad+param["TVWeight_dim1"]*TVGrad_dim1+param["TVWeight_dim2"]*TVGrad_dim2
    return g
def CSL1NICg_XDGRASP(x0,param):
    """
    Non-linear Conjugate Gradient Algorithm adapted from the code provided by
    Miki Lustig in http://www.eecs.berkeley.edu/~mlustig/Software.html

    Input:
        (1) xo, starting point (gridding images)
        (2) param, reconstruction parameters
    Output:
        (1) x, reconstructed images

    """
    import numpy as np

    #staring point
    x=x0
    #line search parameters
    maxlsiter = 6
    gradToll = 1*10**-8
    param["l1Smooth"] = 1*10**-15
    alpha = 0.01
    beta = 0.6
    t0 = 1
    k = 0

    g0 = grad(x,param)
    dx= -q0

    while (1):

        #backtracking line-search
        f0 = objective(x,dx,0, param)
        t = t0
        f1 = objective(x,dx,t,param)
        lsiter = 0
        while (f1 > f0 - alpha*t*np.abs(np.transpose(q0[:])*dx[:]))**2 and (lsiter<maxlsiter):
            lsiter = lsiter + 1
            t = t*beta
            f1 = objective(x,dx,t,param)
        

        # control the number of line searches by adapting the initial step search
        if lsiter > 2:
            t0 = t0*beta
        
        if lsiter<1:
            t0=t0/beta
        

        x = (x+t*dx)
        k = k + 1

        #stopping criteia (to be improved)
        if(k> param.nite) or (np.linalg.norm(dx[:]) < gradToll):
            break
        
        #conjugate gradient calculation
        g1 = grad(x,param)
        bk = (np.transpose(g1[:])*g1[:])/(np.transpose(g0[:])*g0[:]+eps)
        g0 = g1
        dx = -g1 + bk*dx

    end
    return x

import scipy.io as sio
import numpy as np
import math
from mpmath import *

mat_cont = sio.loadmat("/home/bermuded/XDGRASP_Python/data_DCE.mat")

#kc is not used in this demo as respiratory motion signal is not needed.

#kdata is the k-space data. It is only one slice selected from the 3D stack
#dataset after a FFT along the kz. In order to save recon time and memory,
#the full stack-of-stars data is not included.

#b1 is the coil sensitivity maps of the selected slice
#k is the radial k-space trajectory and w is the corresponding density 
#compensation
[nx,ntviews,nc] = np.shape(mat_cont["kdata"])
#ntviews: number of acquired spokes
#nc: number of coil elements
#nx:readout point of each spoke (2x oversampling included)

#Data sorting
nline = 34 #number of spokes in each contrast-enhanced phase
nt=math.floor(ntviews/nline)#number of contrast-enhanced phases

kdata = np.multiply(mat_cont["kdata"],np.repeat(np.sqrt(mat_cont["w"])[:,:,np.newaxis],20,axis=2))

k_data_u =[]
k_u = []
w_u = []
Res_Signal_u = []

for ii in range(1,int(nt)+1):
   if ii == 1:
       k_data_u = np.expand_dims(kdata[:,(ii-1)*nline:ii*nline,:],axis=3)
       k_u = np.expand_dims(mat_cont["k"][:,(ii-1)*nline:ii*nline],axis=2)
       w_u = np.expand_dims(mat_cont["w"][:,(ii-1)*nline:ii*nline],axis=2)
        
   else:
        k_data_u= np.append(k_data_u,np.expand_dims(kdata[:,(ii-1)*nline:ii*nline,:],axis=3),axis=3)
        k_u = np.append(k_u, np.expand_dims(mat_cont["k"][:,(ii-1)*nline:ii*nline],axis=2),axis=2)
        w_u = np.append(w_u, np.expand_dims(mat_cont["w"][:,(ii-1)*nline:ii*nline],axis=2),axis=2)
        


#Recon

nufft_dic = MCNUFFT(k_u,np.double(w_u),mat_cont["b1"])
y = k_data_u
recon_cs= mtimes(nufft_dic,y)

data_gridding = recon_cs/np.max(np.abs(recon_cs[:]))
tvweight_dim1 = np.max(np.abs(recon_cs[:]))*0.03

param = {"E":nufft_dic, "y": y, "TV_dim1": 0, "TVWeight_dim1": tvweight_dim1, "TVWeight_dim2": 0, "nite":5, "display":1}

for n in range(1,3):
    recon_cs = CSL1NICg_XDGRASP(recon_cs,param)

data_grasp=recon_cs/np.max(np.abs(recon_cs[:]))


