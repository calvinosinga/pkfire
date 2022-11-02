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

def xi(delta, BoxSize, MAS='CIC', axis=2, threads=1, verbose = True):

    start = time.time()
    # cdef int kxx, kyy, kzz, kx, ky, kz,dims, middle, k_index, MAS_index
    # cdef int kmax, i, k_par, k_per
    # cdef double k, prefact, mu, mu2
    # cdef double MAS_corr[3]
    # ####### change this for double precision ######
    # cdef float real, imag
    # cdef float MAS_factor
    # cdef np.complex64_t[:,:,::1] delta_k
    # cdef float[:,:,::1] delta_xi
    # ###############################################
    # cdef double[::1] r3D, Nmodes3D
    # cdef double[:,::1] xi3D

    # find dimensions of delta: we assume is a (dims,dims,dims) array
    # determine the different frequencies and the MAS_index
    if verbose: print('\nComputing correlation function of the field...')
    dims = delta.shape[0];  middle = dims//2
    kF,kN,kmax_par,kmax_per,kmax = frequencies(BoxSize,dims)
    MAS_index = MAS_function(MAS)
    MAS_corr = np.zeros(3)
    ## compute FFT of the field (change this for double precision) ##
    delta_k = FFT3Dr_f(delta,threads)
    #################################

    # for each mode correct for MAS and compute |delta(k)^2|
    prefact = np.pi/dims
    for kxx in range(dims):
        kx = (kxx-dims if (kxx>middle) else kxx)
        MAS_corr[0] = MAS_correction(prefact*kx,MAS_index)

        for kyy in range(dims):
            ky = (kyy-dims if (kyy>middle) else kyy)
            MAS_corr[1] = MAS_correction(prefact*ky,MAS_index)

            for kzz in range(middle+1):
                kz = (kzz-dims if (kzz>middle) else kzz)
                MAS_corr[2] = MAS_correction(prefact*kz,MAS_index)  

                # correct modes amplitude for MAS
                MAS_factor = MAS_corr[0]*MAS_corr[1]*MAS_corr[2]
                delta_k[kxx,kyy,kzz] = delta_k[kxx,kyy,kzz]*MAS_factor
                
                # compute |delta(k)^2|
                real = delta_k[kxx,kyy,kzz].real
                imag = delta_k[kxx,kyy,kzz].imag
                # delta_k[kxx,kyy,kzz].real = real*real + imag*imag
                # delta_k[kxx,kyy,kzz].imag = 0.0
                delta_k[kxx, kyy, kzz] = (real*real + imag*imag) + 0*1j

    ## compute IFFT of the field (change this for double precision) ##
    delta_xi = IFFT3Dr_f(delta_k,threads);  del delta_k
    #################################

    # define arrays containing r3D, xi3D and Nmodes3D. We need kmax+1
    # bins since the mode (middle,middle, middle) has an index = kmax
    r3D      = np.zeros(kmax+1,     dtype=np.float64)
    xi3D     = np.zeros((kmax+1,3), dtype=np.float64)
    Nmodes3D = np.zeros(kmax+1,     dtype=np.float64)

    # do a loop over the independent modes.
    # compute k,k_par,k_per, mu for each mode. k's are in kF units
    start2 = time.time()
    for kxx in range(dims):
        kx = (kxx-dims if (kxx>middle) else kxx)
    
        for kyy in range(dims):
            ky = (kyy-dims if (kyy>middle) else kyy)

            for kzz in range(dims): #kzz=[0,1,..,middle] --> kz>0
                kz = (kzz-dims if (kzz>middle) else kzz)

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
                if k_par<0: k_par = -k_par

                # Xi3D
                r3D[k_index]      += k
                xi3D[k_index,0]   +=  delta_xi[kxx,kyy,kzz]
                xi3D[k_index,1]   += (delta_xi[kxx,kyy,kzz]*(3.0*mu2-1.0)/2.0)
                xi3D[k_index,2]   += (delta_xi[kxx,kyy,kzz]*(35.0*mu2*mu2 - 30.0*mu2 + 3.0)/8.0)
                Nmodes3D[k_index] += 1.0


    if verbose: print('Time to complete loop = %.2f'%(time.time()-start2))

    # Xi3D. Discard DC mode bin and give units
    r3D  = r3D[1:];  Nmodes3D = Nmodes3D[1:];  xi3D = xi3D[1:,:]
    for i in range(r3D.shape[0]):
        r3D[i]    = (r3D[i]/Nmodes3D[i])*(BoxSize*1.0/dims)
        xi3D[i,0] = (xi3D[i,0]/Nmodes3D[i])*(1.0/dims**3)
        xi3D[i,1] = (xi3D[i,1]*5.0/Nmodes3D[i])*(1.0/dims**3)
        xi3D[i,2] = (xi3D[i,2]*9.0/Nmodes3D[i])*(1.0/dims**3)
    r3D = np.asarray(r3D);  Nmodes3D = np.asarray(Nmodes3D)
    xi = np.asarray(xi3D)

    if verbose: print('Time taken = %.2f seconds'%(time.time()-start))
    return [r3D, xi, Nmodes3D]

def xpk(delta,BoxSize,axis=2,MAS=['CIC', 'CIC'],threads=1, verbose = True):

    start = time.time()

    if verbose:
        print('\nComputing power spectra of the fields...')

    # find the number and dimensions of the density fields
    # we assume the density fields are (dims,dims,dims) arrays
    dims = len(delta[0]);  middle = dims//2;  fields = len(delta)
    Xfields = fields*(fields-1)//2  #number of independent cross-P(k)

    # check that the dimensions of all fields are the same
    for i in range(1,fields):
        if not len(delta[i])==dims:
            print('Fields have different grid sizes!!!'); sys.exit()

    # find the different relevant frequencies
    kF,kN,kmax_par,kmax_per,kmax = frequencies(BoxSize,dims)

    # find the independent MAS and the arrays relating both.
    # if MAS = ['CIC','PCS','CIC','CIC'] ==> unique_MAS = ['CIC','PCS']
    # num_unique_MAS = 2 : unique_MAS_id = [0,1,0,0]
    unique_MAS     = np.array(list(set(MAS))) #array with independent MAS
    num_unique_MAS = len(unique_MAS)          #number of independent MAS
    unique_MAS_id  = np.empty(fields,dtype=np.int32) 
    for i in range(fields):
        unique_MAS_id[i] = np.where(MAS[i]==unique_MAS)[0][0]

    # define and fill the MAS_corr and MAS_index arrays
    MAS_corr  = np.ones((num_unique_MAS,3), dtype=np.float64)
    MAS_index = np.zeros(num_unique_MAS,    dtype=np.int32)
    for i in range(num_unique_MAS):
        MAS_index[i] = MAS_function(unique_MAS[i])

    # define the real_part and imag_part arrays
    real_part = np.zeros(fields,dtype=np.float64)
    imag_part = np.zeros(fields,dtype=np.float64)

    ## compute FFT of the field (change this for double precision) ##
    # to try to have the elements of the different fields as close as 
    # possible we stack along the z-direction (major-row)
    delta_k = np.empty((dims,dims,(middle+1)*fields),dtype=np.complex64)
    for i in range(fields):
        begin = i*(middle+1);  end = (i+1)*(middle+1)
        delta_k[:,:,begin:end] = FFT3Dr_f(delta[i],threads)
    if verbose:
        print('Time FFTS = %.2f'%(time.time()-start))
    #################################

    # define arrays having k1D, Pk1D, PkX1D & Nmodes1D. We need kmax_par+1
    # bins since modes go from 0 to kmax_par. Is better if we define the
    # arrays as (kmax_par+1,fields) rather than (fields,kmax_par+1) since
    # in memory arrays numpy arrays are row-major
    k1D      = np.zeros(kmax_par+1,           dtype=np.float64)
    Pk1D     = np.zeros((kmax_par+1,fields),  dtype=np.float64)
    PkX1D    = np.zeros((kmax_par+1,Xfields), dtype=np.float64)
    Nmodes1D = np.zeros(kmax_par+1,           dtype=np.float64)

    # define arrays containing Pk2D and Nmodes2D. We define the arrays
    # in this way to have them as close as possible in row-major
    Pk2D     = np.zeros(((kmax_par+1)*(kmax_per+1),fields), 
                        dtype=np.float64)
    PkX2D    = np.zeros(((kmax_par+1)*(kmax_per+1),Xfields),
                        dtype=np.float64)
    Nmodes2D = np.zeros((kmax_par+1)*(kmax_per+1), 
                        dtype=np.float64)

    # define arrays containing k3D, Pk3D,PkX3D & Nmodes3D. We need kmax+1
    # bins since the mode (middle,middle, middle) has an index = kmax.
    # We define the arrays in this way to benefit of row-major
    k3D      = np.zeros(kmax+1,             dtype=np.float64)
    Pk3D     = np.zeros((kmax+1,3,fields),  dtype=np.float64)
    PkX3D    = np.zeros((kmax+1,3,Xfields), dtype=np.float64)
    Nmodes3D = np.zeros(kmax+1,             dtype=np.float64)

    # do a loop over the independent modes.
    # compute k,k_par,k_per, mu for each mode. k's are in kF units
    start2 = time.time();  prefact = np.pi/dims
    for kxx in range(dims):
        kx = (kxx-dims if (kxx>middle) else kxx)
        for i in range(num_unique_MAS):
            MAS_corr[i,0] = MAS_correction(prefact*kx,MAS_index[i])

        for kyy in range(dims):
            ky = (kyy-dims if (kyy>middle) else kyy)
            for i in range(num_unique_MAS):
                MAS_corr[i,1] = MAS_correction(prefact*ky,MAS_index[i])

            for kzz in range(middle+1): #kzz=[0,1,..,middle] --> kz>0
                kz = (kzz-dims if (kzz>middle) else kzz)
                for i in range(num_unique_MAS):
                    MAS_corr[i,2] = MAS_correction(prefact*kz,MAS_index[i])

                # kz=0 and kz=middle planes are special
                if kz==0 or (kz==middle and dims%2==0):
                    if kx<0: continue
                    elif kx==0 or (kx==middle and dims%2==0):
                        if ky<0.0: continue

                ###### k, k_index, k_par,k_per, mu ######
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
                val1 = (3.0*mu2-1.0)/2.0
                val2 = (35.0*mu2*mu2 - 30.0*mu2 + 3.0)/8.0

                # take the absolute value of k_par
                if k_par<0:  k_par = -k_par
                #########################################

                ####### fill the general arrays #########
                # Pk1D(k)
                if k<=middle:
                    k1D[k_par]      += k_par
                    Nmodes1D[k_par] += 1.0
                
                # Pk2D: index_2D goes from 0 to (kmax_par+1)*(kmax_per+1)-1
                index_2D = (kmax_par+1)*k_per + k_par
                Nmodes2D[index_2D] += 1.0

                # Pk3D
                k3D[k_index]      += k
                Nmodes3D[k_index] += 1.0
                #########################################

                #### correct modes amplitude for MAS ####
                for i in range(fields):
                    index = unique_MAS_id[i]
                    MAS_factor = MAS_corr[index,0]*\
                                    MAS_corr[index,1]*\
                                    MAS_corr[index,2]
                    index_z = i*(middle+1) + kzz
                    delta_k[kxx,kyy,index_z] = delta_k[kxx,kyy,index_z]*\
                                                MAS_factor
                    real_part[i] = delta_k[kxx,kyy,index_z].real
                    imag_part[i] = delta_k[kxx,kyy,index_z].imag

                    ########## compute auto-P(k) ########
                    delta2 = real_part[i]*real_part[i] +\
                                imag_part[i]*imag_part[i]

                    # Pk1D: only consider modes with |k|<kF
                    if k<=middle:
                        Pk1D[k_par,i] += delta2

                    # Pk2D: P(k_per,k_par)
                    Pk2D[index_2D,i] += delta2

                    # Pk3D
                    Pk3D[k_index,0,i] += (delta2)
                    Pk3D[k_index,1,i] += (delta2*val1)
                    Pk3D[k_index,2,i] += (delta2*val2)
                #########################################

                ####### compute XPk for each pair #######
                index_X  = 0
                for i in range(fields):
                    for j in range(i+1,fields):
                        delta2_X = real_part[i]*real_part[j] +\
                                    imag_part[i]*imag_part[j]            

                        # Pk1D: only consider modes with |k|<kF
                        if k<=middle:
                            PkX1D[k_par,index_X] += delta2_X

                        # Pk2D: P(k_per,k_par)
                        PkX2D[index_2D,index_X] += delta2_X
                        
                        # Pk3D
                        PkX3D[k_index,0,index_X] += delta2_X
                        PkX3D[k_index,1,index_X] += (delta2_X*val1)
                        PkX3D[k_index,2,index_X] += (delta2_X*val2)

                        index_X += 1
                #########################################

    if verbose:
        print('Time loop = %.2f'%(time.time()-start2))
    fact = (BoxSize/dims**2)**3

    # Pk1D. Discard DC mode bin and give units
    # the perpendicular modes sample an area equal to pi*kmax_per^2
    # we assume that each mode has an area equal to pi*kmax_per^2/Nmodes
    k1D  = k1D[1:];  Nmodes1D = Nmodes1D[1:]
    Pk1D = Pk1D[1:,:];  PkX1D = PkX1D[1:,:]
    for i in range(len(k1D)):
        k1D[i]  = (k1D[i]/Nmodes1D[i])*kF  #give units
        kmaxper = np.sqrt(kN**2 - k1D[i]**2)

        for j in range(fields):
            Pk1D[i,j] = Pk1D[i,j]*fact #give units
            Pk1D[i,j] = Pk1D[i,j]*(np.pi*kmaxper**2/Nmodes1D[i])/(2.0*np.pi)**2
        for j in range(Xfields):
            PkX1D[i,j] = PkX1D[i,j]*fact #give units
            PkX1D[i,j] = PkX1D[i,j]*(np.pi*kmaxper**2/Nmodes1D[i])/(2.0*np.pi)**2
    k1D = np.asarray(k1D);    Nmodes1D = np.asarray(Nmodes1D)  
    Pk1D = np.asarray(Pk1D);  PkX1D = np.asarray(PkX1D)
    results1D = [k1D, Pk1D, PkX1D, Nmodes1D]
    # Pk2D. Keep DC mode bin, give units to Pk2D and find kpar & kper
    kpar = np.zeros((kmax_par+1)*(kmax_per+1), dtype=np.float64)
    kper = np.zeros((kmax_par+1)*(kmax_per+1), dtype=np.float64)
    for k_par in range(kmax_par+1):
        for k_per in range(kmax_per+1):
            index_2D = (kmax_par+1)*k_per + k_par
            kpar[index_2D] = 0.5*(k_par + k_par+1)*kF
            kper[index_2D] = 0.5*(k_per + k_per+1)*kF
    for i in range(len(kpar)):
        for j in range(fields):
            Pk2D[i,j] = Pk2D[i,j]*fact/Nmodes2D[i]
        for j in range(Xfields):
            PkX2D[i,j] = PkX2D[i,j]*fact/Nmodes2D[i]
    kpar = np.asarray(kpar);  kper = np.asarray(kper)
    Nmodes2D = np.asarray(Nmodes2D)
    Pk2D = np.asarray(Pk2D);  PkX2D = np.asarray(PkX2D)
    results2D = [kpar, kper, Pk2D, PkX2D, Nmodes2D]
    # Pk3D. Check modes, discard DC mode bin and give units
    # we need to multiply the multipoles by (2*ell + 1)
    check_number_modes(Nmodes3D,dims)
    k3D  = k3D[1:];  Nmodes3D = Nmodes3D[1:];  
    Pk3D = Pk3D[1:,:,:];  PkX3D = PkX3D[1:,:,:]
    for i in range(len(k3D)):
        k3D[i] = (k3D[i]/Nmodes3D[i])*kF

        for j in range(fields):
            Pk3D[i,0,j] = (Pk3D[i,0,j]/Nmodes3D[i])*fact
            Pk3D[i,1,j] = (Pk3D[i,1,j]*5.0/Nmodes3D[i])*fact
            Pk3D[i,2,j] = (Pk3D[i,2,j]*9.0/Nmodes3D[i])*fact

        for j in range(Xfields):
            PkX3D[i,0,j] = (PkX3D[i,0,j]/Nmodes3D[i])*fact
            PkX3D[i,1,j] = (PkX3D[i,1,j]*5.0/Nmodes3D[i])*fact
            PkX3D[i,2,j] = (PkX3D[i,2,j]*9.0/Nmodes3D[i])*fact

    k3D = np.asarray(k3D);  Nmodes3D = np.asarray(Nmodes3D)
    Pk = np.asarray(Pk3D);  XPk = np.asarray(PkX3D)
    results3D = [k3D, XPk, Nmodes3D, Pk]
    if verbose:
        print('Time taken = %.2f seconds'%(time.time()-start))
    return results1D, results2D, results3D

def xxi(delta1, delta2, BoxSize, MAS=['CIC','CIC'],
                axis=2, threads=1, verbose=True):

    start = time.time()

    MAS_corr = np.zeros((2,3), dtype=np.float64)

    # find dimensions of delta: we assume is a (grid,grid,grid) array
    # determine the different frequencies and the MAS_index
    if verbose:
        print('\nComputing correlation function of the field...')
    grid = delta1.shape[0];  middle = grid//2
    if not grid==delta2.shape[0]:  raise Exception('grid sizes differ!!!')
    kF,kN,kmax_par,kmax_per,kmax = frequencies(BoxSize,grid)
    MAS_index1 = MAS_function(MAS[0])
    MAS_index2 = MAS_function(MAS[1])

    ## compute FFT of the fields (change this for double precision) ##
    delta1_k = FFT3Dr_f(delta1,threads)
    delta2_k = FFT3Dr_f(delta2,threads)
    #################################

    # for each mode correct for MAS and compute |delta(k)^2|
    prefact = np.pi/grid
    for kxx in range(grid):
        kx = (kxx-grid if (kxx>middle) else kxx)
        MAS_corr[0,0] = MAS_correction(prefact*kx,MAS_index1)
        MAS_corr[1,0] = MAS_correction(prefact*kx,MAS_index2)

        for kyy in range(grid):
            ky = (kyy-grid if (kyy>middle) else kyy)
            MAS_corr[0,1] = MAS_correction(prefact*ky,MAS_index1)
            MAS_corr[1,1] = MAS_correction(prefact*ky,MAS_index2)

            for kzz in range(middle+1):
                kz = (kzz-grid if (kzz>middle) else kzz)
                MAS_corr[0,2] = MAS_correction(prefact*kz,MAS_index1)  
                MAS_corr[1,2] = MAS_correction(prefact*kz,MAS_index2)  

                # correct modes amplitude for MAS
                MAS_factor = MAS_corr[0,0]*MAS_corr[0,1]*MAS_corr[0,2]
                delta1_k[kxx,kyy,kzz] = delta1_k[kxx,kyy,kzz]*MAS_factor
                MAS_factor = MAS_corr[1,0]*MAS_corr[1,1]*MAS_corr[1,2]
                delta2_k[kxx,kyy,kzz] = delta2_k[kxx,kyy,kzz]*MAS_factor
                
                # compute |delta(k)^2|
                real1 = delta1_k[kxx,kyy,kzz].real
                imag1 = delta1_k[kxx,kyy,kzz].imag
                real2 = delta2_k[kxx,kyy,kzz].real
                imag2 = delta2_k[kxx,kyy,kzz].imag
                # delta1_k[kxx,kyy,kzz].real = real1*real2 + imag1*imag2
                # delta1_k[kxx,kyy,kzz].imag = 0.0
                delta1_k[kxx, kyy, kzz] = (real1*real2 + imag1*imag2) + 0*1j

    ## compute IFFT of the field (change this for double precision) ##
    delta_xi = IFFT3Dr_f(delta1_k,threads);  del delta1_k, delta2_k
    #################################


    # define arrays containing r3D, xi3D and Nmodes3D. We need kmax+1
    # bins since the mode (middle,middle, middle) has an index = kmax
    r3D      = np.zeros(kmax+1,     dtype=np.float64)
    xi3D     = np.zeros((kmax+1,3), dtype=np.float64)
    Nmodes3D = np.zeros(kmax+1,     dtype=np.float64)

    # do a loop over the independent modes.
    # compute k,k_par,k_per, mu for each mode. k's are in kF units
    start2 = time.time()
    for kxx in range(grid):
        kx = (kxx-grid if (kxx>middle) else kxx)
    
        for kyy in range(grid):
            ky = (kyy-grid if (kyy>middle) else kyy)

            for kzz in range(grid): #kzz=[0,1,..,middle] --> kz>0
                kz = (kzz-grid if (kzz>middle) else kzz)

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
                if k_par<0: k_par = -k_par

                # Xi3D
                r3D[k_index]      += k
                xi3D[k_index,0]   +=  delta_xi[kxx,kyy,kzz]
                xi3D[k_index,1]   += (delta_xi[kxx,kyy,kzz]*(3.0*mu2-1.0)/2.0)
                xi3D[k_index,2]   += (delta_xi[kxx,kyy,kzz]*(35.0*mu2*mu2 - 30.0*mu2 + 3.0)/8.0)
                Nmodes3D[k_index] += 1.0


    if verbose:
        print('Time to complete loop = %.2f'%(time.time()-start2))

    # Xi3D. Discard DC mode bin and give units
    r3D  = r3D[1:];  Nmodes3D = Nmodes3D[1:];  xi3D = xi3D[1:,:]
    for i in range(r3D.shape[0]):
        r3D[i]    = (r3D[i]/Nmodes3D[i])*(BoxSize*1.0/grid)
        xi3D[i,0] = (xi3D[i,0]/Nmodes3D[i])*(1.0/grid**3)
        xi3D[i,1] = (xi3D[i,1]*5.0/Nmodes3D[i])*(1.0/grid**3)
        xi3D[i,2] = (xi3D[i,2]*9.0/Nmodes3D[i])*(1.0/grid**3)
    r3D = np.asarray(r3D);  Nmodes3D = np.asarray(Nmodes3D)
    xi = np.asarray(xi3D)
    results = [r3D, xi, Nmodes3D]
    if verbose:
        print('Time taken = %.2f seconds'%(time.time()-start))
    return results

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

# This function performs the 3D FFT of a field in single precision
def IFFT3Dr_f(a, threads):

    # align arrays
    dims  = len(a)
    a_in  = pyfftw.empty_aligned((dims,dims,dims//2+1),dtype='complex64')
    a_out = pyfftw.empty_aligned((dims,dims,dims),    dtype='float32')

    # plan FFTW
    fftw_plan = pyfftw.FFTW(a_in, a_out, axes=(0,1,2),
                            flags=('FFTW_ESTIMATE',),
                            direction='FFTW_BACKWARD', threads=threads)
                            
    # put input array into delta_r and perform FFTW
    a_in [:] = a;  fftw_plan(a_in,a_out);  return a_out