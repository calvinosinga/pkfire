import pyfftw
import sys
import time
import numpy as np
# COMPUTE POWER SPECTRUM ####################################

def pk(delta, BoxSize, axis=2, MAS = 'CIC', threads = 1, verbose = True):

    start = time.time()
    # cdef int kxx, kyy, kzz, kx, ky, kz,dims, middle, k_index, MAS_index
    # cdef int kmax_par, kmax_per, kmax, k_par, k_per, index_2D, i
    # cdef double k, delta2, prefact, mu, mu2, real, imag, kmaxper, phase
    # cdef double MAS_corr[3]
    # ####### change this for double precision ######
    # cdef float MAS_factor
    # cdef np.complex64_t[:,:,::1] delta_k
    # ###############################################
    # cdef np.float64_t[::1] k1D, kpar, kper, k3D, Pk1D, Pk2D, Pkphase
    # cdef np.float64_t[::1] Nmodes1D, Nmodes2D, Nmodes3D
    # cdef np.float64_t[:,::1] Pk3D 

    # find dimensions of delta: we assume is a (dims,dims,dims) array
    # determine the different frequencies and the MAS_index
    if verbose:  print('\nComputing power spectrum of the field...')
    dims = len(delta);  middle = dims//2
    kF,kN,kmax_par,kmax_per,kmax = frequencies(BoxSize,dims)
    MAS_index = MAS_function(MAS)
    MAS_corr = np.zeros(3)
    ## compute FFT of the field (change this for double precision) ##
    delta_k = FFT3Dr_f(delta,threads)
    #################################

    # define arrays containing k1D, Pk1D and Nmodes1D. We need kmax_par+1
    # bins since modes go from 0 to kmax_par
    k1D      = np.zeros(kmax_par+1, dtype=np.float64)
    Pk1D     = np.zeros(kmax_par+1, dtype=np.float64)
    Nmodes1D = np.zeros(kmax_par+1, dtype=np.float64)

    # define arrays containing Pk2D and Nmodes2D
    Pk2D     = np.zeros((kmax_par+1)*(kmax_per+1), dtype=np.float64)
    Nmodes2D = np.zeros((kmax_par+1)*(kmax_per+1), dtype=np.float64)

    # define arrays containing k3D, Pk3D and Nmodes3D. We need kmax+1
    # bins since the mode (middle,middle, middle) has an index = kmax
    k3D      = np.zeros(kmax+1,     dtype=np.float64)
    Pk3D     = np.zeros((kmax+1,3), dtype=np.float64)
    Pkphase  = np.zeros(kmax+1,     dtype=np.float64)
    Nmodes3D = np.zeros(kmax+1,     dtype=np.float64)


    # do a loop over the independent modes.
    # compute k,k_par,k_per, mu for each mode. k's are in kF units
    start2 = time.time();  prefact = np.pi/dims
    for kxx in range(dims):
        kx = (kxx-dims if (kxx>middle) else kxx)
        MAS_corr[0] = MAS_correction(prefact*kx,MAS_index)
    
        for kyy in range(dims):
            ky = (kyy-dims if (kyy>middle) else kyy)
            MAS_corr[1] = MAS_correction(prefact*ky,MAS_index)

            for kzz in range(middle+1): #kzz=[0,1,..,middle] --> kz>0
                kz = (kzz-dims if (kzz>middle) else kzz)
                MAS_corr[2] = MAS_correction(prefact*kz,MAS_index)  

                # kz=0 and kz=middle planes are special
                if kz==0 or (kz==middle and dims%2==0):
                    if kx<0: continue
                    elif kx==0 or (kx==middle and dims%2==0):
                        if ky<0.0: continue


                # compute |k| of the mode and its integer part
                k       = np.sqrt(kx*kx + ky*ky + kz*kz)
                k_index = int(k)

                # compute the value of k_par and k_perp
                if axis==0:   
                    k_par, k_per = kx, int(np.sqrt(ky*ky + kz*kz))
                elif axis==1: 
                    k_par, k_per = ky, int(np.sqrt(kx*kx + kz*kz))
                else:         
                    k_par, k_per = kz, int(np.sqrt(kx*kx + ky*ky))

                # find the value of mu
                if k==0:  mu = 0.0
                else:     mu = k_par/k
                mu2 = mu*mu

                # take the absolute value of k_par
                if k_par<0:  k_par = -k_par

                # correct modes amplitude for MAS
                MAS_factor = MAS_corr[0]*MAS_corr[1]*MAS_corr[2]
                delta_k[kxx,kyy,kzz] = delta_k[kxx,kyy,kzz]*MAS_factor

                # compute |delta_k|^2 of the mode
                real = delta_k[kxx,kyy,kzz].real
                imag = delta_k[kxx,kyy,kzz].imag
                delta2 = real*real + imag*imag
                phase  = np.arctan2(real, np.sqrt(delta2)) 

                # Pk1D: only consider modes with |k|<kF
                if k<=middle:
                    k1D[k_par]      += k_par
                    Pk1D[k_par]     += delta2
                    Nmodes1D[k_par] += 1.0

                # Pk2D: P(k_per,k_par)
                # index_2D goes from 0 to (kmax_par+1)*(kmax_per+1)-1
                index_2D = (kmax_par+1)*k_per + k_par
                Pk2D[index_2D]     += delta2
                Nmodes2D[index_2D] += 1.0

                # Pk3D.
                k3D[k_index]      += k
                Pk3D[k_index,0]   += delta2
                Pk3D[k_index,1]   += (delta2*(3.0*mu2-1.0)/2.0)
                Pk3D[k_index,2]   += (delta2*(35.0*mu2*mu2 - 30.0*mu2 + 3.0)/8.0)
                Pkphase[k_index]  += (phase*phase)
                Nmodes3D[k_index] += 1.0
    if verbose:  print('Time to complete loop = %.2f'%(time.time()-start2))

    # Pk1D. Discard DC mode bin and give units
    # the perpendicular modes sample an area equal to pi*kmax_per^2
    # we assume that each mode has an area equal to pi*kmax_per^2/Nmodes
    k1D  = k1D[1:];  Nmodes1D = Nmodes1D[1:];  Pk1D = Pk1D[1:]
    for i in range(len(k1D)):
        Pk1D[i] = Pk1D[i]*(BoxSize/dims**2)**3 #give units
        k1D[i]  = (k1D[i]/Nmodes1D[i])*kF      #give units
        kmaxper = np.sqrt(kN**2 - k1D[i]**2)
        Pk1D[i] = Pk1D[i]*(np.pi*kmaxper**2/Nmodes1D[i])/(2.0*np.pi)**2
    
    k1D = np.asarray(k1D);  Pk1D = np.asarray(Pk1D)
    Nmodes1D = np.asarray(Nmodes1D)
    results1D = [k1D, Pk1D, Nmodes1D]
    # Pk2D. Keep DC mode bin, give units to Pk2D and find kpar & kper
    kpar = np.zeros((kmax_par+1)*(kmax_per+1), dtype=np.float64)
    kper = np.zeros((kmax_par+1)*(kmax_per+1), dtype=np.float64)
    for k_par in range(kmax_par+1):
        for k_per in range(kmax_per+1):
            index_2D = (kmax_par+1)*k_per + k_par
            kpar[index_2D] = 0.5*(k_par + k_par+1)*kF
            kper[index_2D] = 0.5*(k_per + k_per+1)*kF
    for i in range(len(kpar)):
        Pk2D[i] = Pk2D[i]*(BoxSize/dims**2)**3/Nmodes2D[i]
    kpar = np.asarray(kpar);  kper = np.asarray(kper)
    Pk2D = np.asarray(Pk2D);  Nmodes2D = np.asarray(Nmodes2D)
    results2D = [kpar, kper, Pk2D, Nmodes2D]
    # Pk3D. Check modes, discard DC mode bin and give units
    # we need to multiply the multipoles by (2*ell + 1)
    check_number_modes(Nmodes3D,dims)
    k3D  = k3D[1:];  Nmodes3D = Nmodes3D[1:];  Pk3D = Pk3D[1:,:]
    Pkphase = Pkphase[1:]
    for i in range(len(k3D)):
        k3D[i]     = (k3D[i]/Nmodes3D[i])*kF
        Pk3D[i,0]  = (Pk3D[i,0]/Nmodes3D[i])*(BoxSize/dims**2)**3
        Pk3D[i,1]  = (Pk3D[i,1]*5.0/Nmodes3D[i])*(BoxSize/dims**2)**3
        Pk3D[i,2]  = (Pk3D[i,2]*9.0/Nmodes3D[i])*(BoxSize/dims**2)**3
        Pkphase[i] = (Pkphase[i]/Nmodes3D[i])*(BoxSize/dims**2)**3
    k3D = np.asarray(k3D);  Nmodes3D = np.asarray(Nmodes3D)
    Pk = np.asarray(Pk3D);  Pkphase = Pkphase
    results3D = [k3D, Pk, Pkphase, Nmodes3D]
    if verbose:  print('Time taken = %.2f seconds'%(time.time()-start))
    return results1D, results2D, results3D

def pk2D(delta,BoxSize,MAS='CIC',threads=1,verbose=True):

    start = time.time()
    # cdef int kxx, kyy, kx, ky, grid, middle, k_index, MAS_index
    # cdef int kmax_par, kmax_per, kmax, i
    # cdef double k, delta2, prefact, real, imag
    # cdef double MAS_corr[2]
    # ####### change this for double precision ######
    # cdef float MAS_factor
    # cdef np.complex64_t[:,::1] delta_k
    # ###############################################
    # cdef np.float64_t[::1] k2D, Nmodes, Pk2D

    # find dimensions of delta: we assume is a (grid,grid) array
    # determine the different frequencies and the MAS_index
    if verbose:  print('\nComputing power spectrum of the field...')
    grid = len(delta);  middle = grid//2
    kF,kN,kmax_par,kmax_per,kmax = frequencies_2D(BoxSize,grid)
    MAS_index = MAS_function(MAS)
    MAS_corr = np.zeros(2)
    ## compute FFT of the field (change this for double precision) ##
    delta_k = FFT2Dr_f(delta,threads)
    #################################

    # define arrays containing k3D, Pk3D and Nmodes3D. We need kmax+1
    # bins since the mode (middle,middle, middle) has an index = kmax
    k2D    = np.zeros(kmax+1, dtype=np.float64)
    Pk2D   = np.zeros(kmax+1, dtype=np.float64)
    Nmodes = np.zeros(kmax+1, dtype=np.float64)


    # do a loop over the independent modes.
    # compute k,k_par,k_per, mu for each mode. k's are in kF units
    start2 = time.time();  prefact = np.pi/grid
    for kxx in range(grid):
        kx = (kxx-grid if (kxx>middle) else kxx)
        MAS_corr[0] = MAS_correction(prefact*kx,MAS_index)

        for kyy in range(middle+1): #kyy=[0,1,..,middle] --> ky>0
            ky = (kyy-grid if (kyy>middle) else kyy)
            MAS_corr[1] = MAS_correction(prefact*ky,MAS_index)

            # ky=0 & ky=middle are special (modes with (kx<0, ky=0) are not
            # independent of (kx>0, ky=0): delta(-k)=delta*(+k))
            if ky==0 or (ky==middle and grid%2==0):
                if kx<0:  continue

            # compute |k| of the mode and its integer part
            k       = np.sqrt(kx*kx + ky*ky)
            k_index = int(k)

            # correct modes amplitude for MAS
            MAS_factor = MAS_corr[0]*MAS_corr[1]
            delta_k[kxx,kyy] = delta_k[kxx,kyy]*MAS_factor

            # compute |delta_k|^2 of the mode
            real = delta_k[kxx,kyy].real
            imag = delta_k[kxx,kyy].imag
            delta2 = real*real + imag*imag

            # Pk
            k2D[k_index]    += k
            Pk2D[k_index]   += delta2
            Nmodes[k_index] += 1.0
    if verbose:  print('Time to complete loop = %.2f'%(time.time()-start2))

    # Pk2D. Check modes, discard DC mode bin and give units
    check_number_modes_2D(Nmodes,grid)
    k2D  = k2D[1:];  Nmodes = Nmodes[1:];  Pk2D = Pk2D[1:]
    for i in range(len(k2D)):
        k2D[i]  = (k2D[i]/Nmodes[i])*kF
        Pk2D[i] = (Pk2D[i]/Nmodes[i])*(BoxSize/grid**2)**2
    k  = np.asarray(k2D);  Nmodes = np.asarray(Nmodes)
    Pk = np.asarray(Pk2D)

    if verbose:  print('Time taken = %.2f seconds'%(time.time()-start))
    return [k, Pk, Nmodes]
# HELPER METHODS FOR COMPUTING POWER SPECTRUM, TAKEN FROM PYLIANS ###
def frequencies(BoxSize,dims):
    kF = 2.0*np.pi/BoxSize;  middle = dims//2;  kN = middle*kF
    kmax_par = middle
    kmax_per = int(np.sqrt(middle**2 + middle**2))
    kmax     = int(np.sqrt(middle**2 + middle**2 + middle**2))
    return kF,kN,kmax_par,kmax_per,kmax

def frequencies_2D(BoxSize,dims):
    kF = 2.0*np.pi/BoxSize;  middle = dims//2;  kN = middle*kF
    kmax_par = middle
    kmax_per = middle
    kmax     = int(np.sqrt(middle**2 + middle**2))
    return kF,kN,kmax_par,kmax_per,kmax

def MAS_function(MAS):
    MAS_index = 0;  #MAS_corr = np.ones(3,dtype=np.float64)
    if MAS=='NGP':  MAS_index = 1
    if MAS=='CIC':  MAS_index = 2
    if MAS=='TSC':  MAS_index = 3
    if MAS=='PCS':  MAS_index = 4
    return MAS_index#,MAS_corr

# This function performs the 3D FFT of a field in single precision
def FFT3Dr_f(a, threads = 1):

    # align arrays
    dims  = len(a)
    a_in  = pyfftw.empty_aligned((dims,dims,dims),    dtype='float32')
    a_out = pyfftw.empty_aligned((dims,dims,dims//2+1),dtype='complex64')

    # plan FFTW
    fftw_plan = pyfftw.FFTW(a_in, a_out, axes=(0,1,2),
                            flags=('FFTW_ESTIMATE',),
                            direction='FFTW_FORWARD', threads=threads)
                            
    # put input array into delta_r and perform FFTW
    a_in [:] = a;  fftw_plan(a_in,a_out);  return a_out

def FFT2Dr_f(a, threads = 1):

    # align arrays
    grid  = len(a)
    a_in  = pyfftw.empty_aligned((grid,grid),    dtype='float32')
    a_out = pyfftw.empty_aligned((grid,grid//2+1),dtype='complex64')

    # plan FFTW
    fftw_plan = pyfftw.FFTW(a_in, a_out, axes=(0,1),
                            flags=('FFTW_ESTIMATE',),
                            direction='FFTW_FORWARD', threads=threads)
                            
    # put input array into delta_r and perform FFTW
    a_in [:] = a
    fftw_plan(a_in,a_out) 
    return a_out

def MAS_correction(x, MAS_idx):
    return (1.0 if (x==0.0) else np.power(x/np.sin(x), MAS_idx))

# This function checks that all independent modes have been counted
def check_number_modes(Nmodes,dims):
    # (0,0,0) own antivector, while (n,n,n) has (-n,-n,-n) for dims odd
    if dims%2==1:  own_modes = 1 
    # (0,0,0),(0,0,n),(0,n,0),(n,0,0),(n,n,0),(n,0,n),(0,n,n),(n,n,n)
    else:          own_modes = 8 
    repeated_modes = (dims**3 - own_modes)//2  
    indep_modes    = repeated_modes + own_modes

    if int(np.sum(Nmodes))!=indep_modes:
        print('WARNING: Not all modes counted')
        print('Counted  %d independent modes'%(int(np.sum(Nmodes))))
        print('Expected %d independent modes'%indep_modes)
        sys.exit()
    return   

# This function checks that all independent modes have been counted
def check_number_modes_2D(Nmodes,dims):
    # (0,0) own antivector, while (n,n) has (-n,-n) for dims odd
    if dims%2==1:  own_modes = 1 
    # (0,0),(0,n),(0,n),(n,0),(n,n)
    else:          own_modes = 4
    repeated_modes = (dims**2 - own_modes)//2  
    indep_modes    = repeated_modes + own_modes

    if int(np.sum(Nmodes))!=indep_modes:
        print('WARNING: Not all modes counted')
        print('Counted  %d independent modes'%(int(np.sum(Nmodes))))
        print('Expected %d independent modes'%indep_modes)
        sys.exit()
    return