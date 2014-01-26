from __future__ import division
import numpy as np
from scipy.stats import nanmean, nanmedian
from scipy.io import loadmat, savemat
import os
import sys

ITER = 0


def icoshift(xT,  xP,  inter='whole',  n='f',  options=[1,  1,  0,  0,  0],  Scal=None):
    '''
    interval Correlation Optimized shifting
    [xCS, ints, ind, target] = icoshift(xT, xP, inter[, n[, options[, Scal]]])
    Splits a spectral database into "inter" intervals and coshift each vector
    left-right to get the maximum correlation toward a reference or toward an
    average spectrum in that interval. Missing parts on the edges after
    shifting are filled with "closest" value or with "NaNs".
    INPUT
    xT (1 * mT)    : target vector.
                     Use 'average' if you want to use the average spectrum as a reference
                     Use 'median' if you want to use the median spectrum as a reference
                     Use 'max' if you want to use for each segment the corresponding actual spectrum having max features as a reference
                     Use 'average2' for using the average of the average multiplied for a requested number (default=3) as a reference

    xP (nP * mP)   : Matrix of sample vectors to be aligned as a sample-set
                     towards common target xT
    inter          : definition of alignment mode
                     'whole'         : it works on the whole spectra (no intervals).
                     nint            : (numeric) number of many intervals.
                     'ndata'         : (string) length of regular intervals
                                       (remainders attached to the last).
                     [I1s I1e, I2s...]: interval definition. ('I(n)s' interval
                                       n start,  'I(n)e' interval n end).
                     (refs:refe)     : shift the whole spectra according to a
                                       reference signal(s) in the region
                                       refs:refe (in sampling points)
                     'refs-refe'     : `shift the whole spectra according to a
                                       reference signal(s) in the region
                                       refs-refe (in Scal units)
    n (1 * 1)      : (optional)
                     n = integer n.: maximum shift correction in data
                                     points/Scal units (cf. options[4])
                                     in x/rows. It must be >0
                     n = 'b' (best): the algorithm search for the best n
                                     for each interval (it can be time consuming!)
                     n = 'f' (fast): fast search for the best n for each interval (default)
                     A warning is displayed for each interval if "n" appears too small
    options (1 * 5): (optional)
                     (0) triggers plots & warnings:
                         0 : no on-screen output
                         1 : only warnings (default)
                         2 : warnings and plots
                     (1) selects filling mode
                         0 : using not a number
                         1 : using previous point (default)
                     (2) turns on Co-shift preprocessing
                         0 : no Co-shift preprocessing (default)
                         1 : Executes a Co-shift step before carrying out iCOshift
                     (3) max allowed shift for the Co-shift preprocessing (default = equal to n if not specified)
                         it has to be given in Scal units if option(5)=1
                     (4) 0 : intervals are given in No. of datapoints  (deafult)
                         1 : intervals are given in ppm --> use Scal for inter and n
    Scal           : vector of scalars used as axis for plot (optional)
    OUTPUT
    xCS  (nP * mT): shift corrected vector or matrix
    ints (nI * 4) : defined intervals (Int. No.,  starting point,  ending point,  size)
    ind  (nP * nI): matrix of indexes reporting how many points each spectrum
                    has been shifted for each interval (+ left,  - right)
    target (1 x mP): actual target used for the final alignment
    Authors:
    Francesco Savorani - Department of Food Science
                         Quality & Technology - Spectroscopy and Chemometrics group
                         Faculty of Sciences
                         University of Copenhagen - Denmark
    email: frsa@life.ku.dk - www.models.life.ku.dk
    Giorgio Tomasi -     Department of Basic Science and Environment
                         Soil and Environmental Chemistry group
                         Faculty of Life Sciences
                         University of Copenhagen - Denmark
    email: giorgio.tomasi@ec.europa.eu - www.igm.life.ku.dk
    Python conversion by:
    Martin Fitzpatrick -  Rheumatology Research Group
                          Centre for Translational Inflammation Research
                          School of Immunity and Infection
                          University of Birmingham - United Kingdom


    170508 (FrSa) first working code
    211008 (FrSa) improvements and bugs correction
    111108 (Frsa) Splitting into regular intervals (number of intervals or wideness in datapoints) implemented
    141108 (GT)   FFT alignment function implemented
    171108 (FrSa) options implemented
    241108 (FrSa) Automatic search for the best or the fastest n for each interval implemented
    261108 (FrSa) Plots improved
    021208 (FrSa) 'whole' case added & robustness improved
    050309 (GT)   Implentation of interpolation modes (NaN); Cosmetics; Graphics
    240309 (GT)   Fixed bug in handling missing values
    060709 (FrSa) 'max' target and output 'target' added. Some speed,  plot and robustness improvements
    241109 (GT)   Interval and band definition in units (added options[4])
    021209 (GT)   Minor debugging for the handling of options[4]
    151209 (FrSa) Cosmetics and minor debugging for the handling of options[4]
    151110 (FrSa) Option 'Max' works now also for alignment towards a reference signal
    310311 (FrSa) Bugfix for the 'whole' case when mP < 101
    030712 (FrSa) Introducing the 'average2' xT (target) for a better automatic target definition. Plots updated to include also this case
    281023 (MF)   Initial implementation of Python version of icoshift algorithm. PLOTS NOT INCLUDED
    '''

    # RETURNS [xCS, ints, ind, target]
    if Scal == None:
        Scal = np.array(range(0, xP.shape[1]))  # 1:size(xP, 2)]

    # ERRORS CHECK
    # Constant
    # To avoid out of memory errors when 'whole',  the job is divided in
    # blocks of 32MB
    BLOCKSIZE = 2 ** 25

    print xP.shape

    if len(Scal) != max(Scal.shape):
        error('Scal must be a vector')
    if max(Scal.shape) != xP.shape[1]:
        error('X and Scal are not of compatible length %d vs. %d' %
              (max(Scal.shape), xP.shape[1]))

    dScal = np.diff(Scal)

    incScal = Scal[0] - Scal[1]
    if incScal == 0 or not all(np.sign(dScal) == - np.sign(incScal)):
        error('Scal must be strictly monotonic')

    flag_ScalDir = incScal < 0
    flag_DiScal = any(abs(dScal) > 2 * np.min(abs(dScal)))
    max_flag = 0
    avg2_flag = 0
    if xT == 'average':
        xT = np.array([nanmean(xP, axis=0), ])
        note = 'average'
    else:
        if xT == 'median':
            xT = np.array([nanmedian(xP, axis=0), ])
            note = 'median'
        else:
            if xT == 'average2':
                xT = np.array([nanmean(xP, axis=0), ])
                avg2_flag = 1
                if options[0] != 0:
                    avg_power = raw_input(
                        'which multiplier for the 2nd average? [default is 3] ')
                else:
                    avg_power = 3
                if not is_number(avg_power):
                    avg_power = 3
                note = 'average2'
            else:
                if xT == 'max':
                    xT = np.zeros((1, xP.shape[1],))
                    max_flag = 1
                    note = 'max'

    nT, mT = xT.shape
    nP, mP = xP.shape
    if (mT != mP):
        error('Target "xT" and sample "xP" must have the same number of columns')
    if is_number(inter):
        if inter > mP:
            error(
                'ERROR: number of intervals "inter" must be smaller than number of variables in xP')

    options = np.array(options)
    options_co = np.array([1, 1, 0, 0, 0]).reshape(1, -1)

    if max(options.shape) < max(options_co.shape):
        options[(end + 1 - 1):max(options_co.shape)
                ] = options_co[(max(options.shape) + 1 - 1):]

    if options[0] < 0 or options[0] > 2:
        error('options(1) must be 0, 1 or 2')

    if options[4]:
        prec = abs(np.min(unique(dScal)))
        if flag_DiScal:
            warning('iCOshift:discontinuous_Scal',
                    'Scal vector is not continuous, the defined intervals might not reflect actual ranges')

    flag_coshift = not inter == 'whole' and options[2]

    if flag_coshift:
        if options[3] == 0:
            n_co = n
        else:
            n_co = options[3]
            if nargin >= 6 and options[4]:
                n_co = dscal2dpts(n_co, Scal, prec)
        if max_flag:
            xT = nanmean(xP, axis=0)
        xPo = xP
        xP, nil, wint = icoshift(
            xT, xP, 'whole', n_co, np.array([0, 1, 0]).reshape(1, -1))
        title_suff = ' (after coshift)'
        if note == 'average':
            xT = nanmean(xP)
        if xT == 'median':
            xT = nanmedian(xP)
        if note == 'average2':
            xT = nanmean(xP)
    else:
        title_suff = ''

    flag_allsegs = False
    whole = False
    flag2 = False

    if isinstance(inter, basestring):
        if inter == 'whole':
            inter = np.array([0, mP - 1]).reshape(1, -1)
            whole = True
        else:
            if not any(inter == '-'):
                interv = str2double(inter)
                if nargin < 6 or not options[4]:
                    interv = round_(interv)
                else:
                    interv = dscal2dpts(interv, Scal, prec)
                inter = defints(xP, interv, options[0])
                flag_allsegs = True
            else:
                interv = regexp(
                    inter, '(-{0,1}\\d*\\.{0,1}\\d*)-(-{0,1}\\d*\\.{0,1}\\d*)', 'tokens')
                interv = sort(
                    scal2pts(str2double(cat(0, interv[:])), Scal, prec))
                if interv.size != 2:
                    error('Invalid range for reference signal')
                inter = range(interv[0], (interv[1] + 1))
                flag2 = True
    else:
        if is_number(inter):
            if inter.size == 1:
                if np.fix(inter) == inter:
                    flag_allsegs = True
                    remain = mod(mP, inter)
                    N = np.fix(mP / inter)
                    startint = np.array(
                        [(range(1, ((remain - 1) * (N + 1) + 1 + 1), (N + 1))).T,
                         (range((remain - 1) * (N + 1) + 1 + 1 + N, (mP + 1), N)).T]).reshape(1, -1)
                    endint = np.array([startint[1:inter], mP]).reshape(1, -1)
                    inter = np.array([startint, endint]).reshape(1, -1).T
                    inter = inter[:].T
                else:
                    error('The number of intervals must be an integer')
            else:
                flag2 = np.array_equal(np.fix(inter), inter) and max(inter.shape) > 1 and np.array_equal(
                    np.array([1, np.max(inter) - np.min(inter) + 1]).reshape(1, -1), inter.shape) and np.array_equal(unique(np.diff(inter, 1, 2)), 1)
                if not flag2 and options[4]:
                    inter = scal2pts(inter, Scal, prec)
                    if any(inter[0:2:] > inter[1:2:]) and not flag_ScalDir:
                        inter = flipud(reshape(inter, 2, max(inter.shape) / 2))
                        inter = inter[:].T

    nint, mint = inter.shape
    # mint=mint-1 # Indexing?

    wmsg = np.array([])
    scfl = np.array_equal(np.fix(Scal), Scal) and not options[4]
    if isinstance(inter, basestring) and (len(n) > 1 or n not in ['b', 'f']):
        error('"n" must be a scalar b or f')

    else:
        if is_number(n):
            if any(n <= 0):
                error('ERROR: shift(s) "n" must be larger than zero')
            if n.size > 1:
                wmsg = sprintf(
                    '"n" must be a scalar/character; first element (i.e. %i) used', round_(n))
            if scfl and n != np.fix(n):
                wmsg = sprintf(
                    '"n" must be an integer if Scal is ignorde; first element (i.e. %i) used', round_(n))
            else:
                if options[4]:
                    n = dscal2dpts(n, Scal, prec)
            if not flag2 and any(np.diff((reshape(inter, 2, mint / 2)), 1, 1) < n):
                error(
                    'ERROR: shift "n" must be not larger than the size of the smallest interval')
            n = round_(n[0])
            if not (0 in wmsg.shape):
                warning('iCoShift:Input', wmsg)
                input('press a key to continue...')

    flag = np.isnan(cat(0, xT, xP))
    frag = False
    Ref = lambda c: np.reshape(c, (2, max(c.shape) / 2)).T
    vec = lambda A: A.flatten()

    mi, pmi = np.min(inter, axis=0)
    ma, pma = np.max(inter, axis=0)

    # There are missing values in the dataset; so remove them before starting
    # if they line up between datasets
    if vec(flag).any():
        if np.array_equal(flag[np.ones((nP, 1), dtype=int), :], flag[1:,:]):
            Select = np.any
        else:
            Select = np.all

        if flag2:
            intern_ = RemoveNaN(
                np.array([0, pma - pmi]).reshape(1, -1), cat(0, xT[:, inter], xP[:, inter]), Select)
            if intern_.shape[0] != 1:
                error(
                    'Reference region contains a pattern of missing that cannot be handled consistently')
            else:
                if not np.array_equal(intern_, np.array([1, inter[-2] - inter[0] + 1]).reshape(1, -1)):
                    warning('iCoShift:miss_refreg',
                            'The missing values at the boundaries of the reference region will be ignored')
            intern_ = range(inter[0] + intern_[0], (inter[0] + intern_[1] + 1))
        else:
            intern_, flag_nan = RemoveNaN(
                Ref(inter), cat(0, xT, xP), Select, flags=True)
            intern_ = vec(intern_.T).T
        if (0 in intern_.shape):
            error('iCoShift cannot handle this pattern of missing values.')

        if max(intern_.shape) != max(inter.shape) and not flag2:

            if whole:
                if max(intern_.shape) > 2:
                    if options[0] == 2:
                        xPBU = xP
                        xTBU = xT

                    # reshape(c,2,length(c)/2)';
                    XSeg, InOr = ExtractSegments(cat(0, xT, xP), Ref(intern_))
                    InOrf = InOr.flatten()
                    inter = np.array([InOrf[0], InOrf[-1] - 1]).reshape(1, -1)
                    InOr = cat(1, Ref(intern_), InOr)
                    xP = XSeg[1:, :]
                    xT = XSeg[0, :].reshape(1, -1)
                    frag = True

            else:
                warning(
                    'iCoShift:Missing_values', '\nTo handle the pattern of missing values, %i segments are created/removed',
                    abs(max(intern_.shape) - max(inter.shape)) / 2)
                inter = intern_
                nint, mint = inter.shape
    xCS = xP
    mi = np.min(inter)
    pmi = np.min(inter)
    ma = np.max(inter)
    pma = np.max(inter)

    flag = max(inter.shape) > 1 and np.array_equal(
        np.array([1, pma - pmi + 1]).reshape(1, -1), inter.shape) and np.array_equal(unique(np.diff(inter, 1, 2)), 1)

    if flag:
        if options[0]:
            if n == 'b':
                print(
                    'Automatic searching for the best "n" for the reference window "refW" enabled \That can take a longer time \\n')
            else:
                if n == 'f':
                    print(
                        'Fast automatic searching for the best "n" for the reference window "refW" enabled \\n')
        if max_flag:
            np.max(np.sum(xP))
            xT[mi:ma] = xP[bmax, mi:ma]
        ind = NaN(nP, 1)
        missind = not all(np.isnan(xP), 2)
        xCS[missind, :], ind[missind] = coshifta(xT, xP[missind,:], inter, n, np.array([options[0:3], BLOCKSIZE]).reshape(1, -1))
        ints = np.array([1, mi, ma]).reshape(1, -1)

    else:
        if mint > 1:
            if mint % 2:
                error('Wrong definition of intervals ("inter")')
            if ma > mP:
                error('Intervals ("inter") exceed samples matrix dimension')

            allint = np.zeros((3))  # .reshape(1,-1)
            allint[:] = np.arange(0, int(mint - 1 / 2))

            b = inter[:, range(0, mint, 1)]
            if len(b) > 0:
                allint[1:] = b

            c = inter[:, range(1, mint, 1)]
            if len(c) > 0:
                allint[2:] = c

            allint = allint.reshape(1, -1)

        sallint = sortrows(allint, 1)
        # sinter = reshape(sallint(:,2:3)',1,numel(sallint(:,2:3)));
        sinter = np.reshape(sallint[:, 1:3].T, (1, sallint[:, 1:3].size))
        sinter = np.sort(allint, axis=0);

        intdif = np.diff(sinter)

        if any(intdif[1:2:max(intdif.shape)] < 0):
            uiwait(
                msgbox('The user-defined intervals are overlapping: is that intentional?', 'Warning', 'warn'))

        #del sallint
        del sinter
        del intdif

        ints = allint
        # ints(:,4)=(ints(:,3)-ints(:,2))+1;
        ints = np.append(ints, ints[:, 2] - ints[:, 1])
        ind = np.zeros((nP, allint.shape[0]))

        if options[0]:
            if n == 'b':
                print(
                    'Automatic searching for the best "n" for each interval enabled \That can take a longer time...')

            else:
                if n == 'f':
                    print(
                        'Fast automatic searching for the best "n" for each interval enabled')
        for i in range(0, allint.shape[0]):
            if options[0] != 0:
                if whole:
                    print('Co-shifting the whole %s samples...' % nP)
                else:
                    print('Co-shifting interval no. %s of %s...' %
                          (i, allint.shape[0]))

            # xP(:,allint(i,2):allint(i,3));
            # FIXME? 0:2, or 1:2?
            intervalnow = xP[:, allint[i, 1]:allint[i, 2] + 1]

            if max_flag:

                np.max(np.sum(intervalnow, 2))
                target = intervalnow[bmax, :]
                xT[allint[i, 1]:allint[i, 2]] = target
            else:
                target = xT[:, allint[i, 1]:allint[i, 2] + 1]

            missind = ~np.all(np.isnan(intervalnow), axis=1)
            
            if not np.all(np.isnan(target)) and np.any(missind):

                cosh_interval, loc_ind, nul = coshifta(target, intervalnow[missind, :], np.array([0]), n, np.append(options[0:3], np.array([BLOCKSIZE]) ))

                xCS[missind, allint[i, 1]:allint[i, 2] + 1] = cosh_interval

                ind[missind, i] = loc_ind.flatten()

            else:
                xCS[:, allint[i, 1]:allint[i, 1] + 1] = intervalnow

        if avg2_flag:
            for i in range(0, allint.shape[0]):
                if options[0] != 0:
                    if whole:
                        print('Co-shifting again the whole %g samples... ', nP)
                    else:
                        print('Co-shifting again interval no. %g of %g... ',
                              i, allint.shape[0])

                intervalnow = xP[:, allint[i, 1]:allint[i, 2]]
                target1 = np.mean(xCS[:, allint[i, 1]:allint[i, 2]])
                min_interv = np.min(target1)
                target = (target1 - min_interv).dot(avg_power)
                missind = not all(np.isnan(intervalnow), 2)
                if not all(np.isnan(target)) and np.sum(missind) != 0:
                    cosh_interval, loc_ind = coshifta(target, intervalnow[missind, :], 0, n, np.append(options[0:3], np.array([BLOCKSIZE]) ))
                    xCS[missind, allint[i, 1]:allint[i, 2]] = cosh_interval
                    xT[allint[i, 1]:allint[i, 2]] = target
                    ind[missind, i] = loc_ind.T
                else:
                    xCS[:, allint[i, 1]:allint[i, 2]] = intervalnow
                home

    if frag:

        Xn = NaN(nP, mP)
        Xn.shape
        for i_sam in range(0, nP):
            for i_seg in range(0, InOr.shape[0]):
                Xn[i_sam, InOr[i_seg, 0]:InOr[i_seg, 1]
                    + 1] = xCS[i_sam, InOr[i_seg, 2]:InOr[i_seg, 3] + 1]
                if loc_ind[i_sam] < 0:
                    if flag_nan[i_seg, 0, i_sam]:
                        Xn[i_sam, InOr[i_seg, 0]:InOr[i_seg, 0]
                            - loc_ind[i_sam, 0] + 1] = np.nan
                else:
                    if loc_ind[i_sam] > 0:
                        if flag_nan[i_seg, 1, i_sam]:
                            Xn[i_sam, (InOr[i_seg, 1] - loc_ind[i_sam, 0] + 1)                               :InOr[i_seg, 1]+1] = np.nan

        if options[0] == 2:
            xP = xPBU
            xT = xTBU
        xCS = Xn
    target = xT

    if flag_coshift:
        ind = ind + wint * ones(1, ind.shape[1])

    return xCS, ints, ind, target


def coshifta(xT, xP, refW=np.array([0]), n=np.array([1, 2, 3]), options=np.array([])):

    if np.all(refW >= 0):
        rw = max(refW.shape)
    else:
        rw = 1

    options_def = np.array([1, 1, 1, (2 ** 25)])
    if len(options_def) > len(options):
        o = len(options)
        options = np.append(options, np.zeros(len(options_def) - o))
        options[o:] = options_def[o:]

    if options[1] == 1:
        Filling = - np.inf
    else:
        if options[1] == 0:
            Filling = np.nan
        else:
            error('options(2) must be 0 or 1')

    if xT == 'average':
        xT = nanmean(xP, axis=0)

    nT, mT = xT.shape
    nP, mP = xP.shape

    if len(refW.shape) > 1:
        nR, mR = refW.shape
    else:
        nR, mR = refW.shape[0], 0

    print 'mT,mP', mT, mP

    if (mT != mP):
        error(
            'Target "xT" and sample "xP" must be of compatible size (same vectors, same matrices or row + matrix of rows)')
    if np.any(n <= 0):
        error('Shift(s) "n" must be larger than zero')
    if (nR != 1):
        error('Reference windows "refW" must be either a single vector or 0')
    if rw > 1 and (np.min(refW) < 1) or (np.max(refW) > mT):
        error('Reference window "refW" must be a subset of xP')
    if (nT != 1):
        error('Target "xT" must be a single row spectrum/chromatogram')
        
    auto = 0
    if n == 'b':
        auto = 1
        if rw != 1:
            n = np.fix(0.05 * mR)
            n = 10 if n < 10 else n
            src_step = np.fix(mR * 0.05)
        else:
            n = np.fix(0.05 * mP)
            n = 10 if n < 10 else n
            src_step = np.fix(mP * 0.05)
        try_last = 0
    else:
        if n == 'f':
            auto = 1
            if rw != 1:
                n = mR - 1
                src_step = np.round(mR / 2) - 1
            else:
                n = mP - 1
                src_step = np.round(mP / 2) - 1
            try_last = 1
    if (nT != 1):
        error('ERROR: Target "xT" must be a single row spectrum/chromatogram')
    xW = NaN(nP, mP)
    ind = np.zeros((1, nP))
    BLOCKSIZE = options[3]
    nBlocks = int(np.ceil(sys.getsizeof(xP) / BLOCKSIZE))
    SamxBlock = np.array([int(nP / nBlocks)])

    SamxBlock = SamxBlock.T  # np.reshape(SamxBlock, (SamxBlock.shape[0],1) )

    indBlocks = SamxBlock[np.ones((nBlocks), dtype=bool)]
    indBlocks[0:nP % SamxBlock] = SamxBlock + 1
    indBlocks = np.array([0, np.cumsum(indBlocks, 0)]).reshape(1, -1)

    if auto == 1:
        while auto == 1:

            if Filling == -np.inf:
            
                print xP
                print np.tile(xP[:, :1], (1., n)
                print cat(1, np.tile(xP[:, :1], (1., n) )
            
                xtemp = cat(1, np.tile(xP[:, :1], (1., n)),
                            xP, np.tile(xP[:, -1:, ], (1., n)))

                

                # xtemp=np.array([xP[:,np.ones(1,n)],xP,xP[:,(mP[0,np.ones(1,n)]-1)]])
                # #.reshape(1,-1)

            else:
                if np.isnan(Filling):
                    # FIXME
                    xtemp = np.array(
                        [NaN(nP, n), xP, NaN(nP, n)]).reshape(1, -1)

            if rw == 1:
                refW = range(0, mP)

            ind = NaN(nP, 1)
            R = False  # np.empty((1,1))

            for i_block in range(0, nBlocks):
                block_indices = range(
                    indBlocks[0, i_block], indBlocks[0, i_block + 1])
                #dummy,ind[(block_indices-1)],Ri = CC_FFTShift(xT[0,(refW-1)],xP[(block_indices-1),(refW-1)],np.array([- n,n,2,1,Filling]).reshape(1,-1))
                # CC_FFTShift(xT(1,refW),xP(block_indices,refW),[-n n 2 1
                # Filling]); %#ok<ASGLU>
                a, b, c = CC_FFTShift( xT[0, [refW]], xP[block_indices, :][:, refW], np.array([-n, n, 2, 1, Filling]) )
                dummy, ind[block_indices], Ri = CC_FFTShift( xT[0, [refW]], xP[block_indices, :][:, refW], np.array([-n, n, 2, 1, Filling]) )

                if not R:
                    R = np.empty((0, Ri.shape[1]))
                R = cat(0, R, Ri).T

            temp_index = range(- n, n)

            for i_sam in range(0, nP):
                index = np.flatnonzero(temp_index == ind[i_sam])

                xW[i_sam, :] = xtemp[i_sam, index:index + mP]

            if (np.max(abs(ind)) == n) and try_last != 1:
                if n + src_step >= refW.shape[1]:
                    try_last = 1
                    continue
                n = n + src_step
                continue

            else:
                if (np.max(abs(ind)) < n) and n + src_step < len(refW) and try_last != 1:
                    n = n + src_step
                    try_last = 1
                    continue
                else:
                    auto = 0
                    if options[0] != 0:
                        print(
                            'Best shift allowed for this interval = %g \\n', n)
    else:
        if Filling == - np.inf:
            xtemp = np.array(
                [xP[:, np.ones((1, n))], xP, xP[:, mP[0, np.ones((1, n))]]]).reshape(1, -1)

        else:
            if np.isnan(Filling):
                xtemp = np.array([NaN(nP, n), xP, NaN(nP, n)]).reshape(1, -1)

        if rw == 1:
            refW = range(1, (mP + 1))

        ind = NaN(nP, 1)
        R = np.array([])

        for i_block in range(0, nBlocks):
            block_indices = range(indBlocks[i_block], indBlocks[i_block + 1])
            dummy, ind[block_indices], Ri = CC_FFTShift(xT[0, [refW]], xP[block_indices, :][:, refW], np.array([-n, n, 2, 1, Filling]))
            R = cat(0, R, Ri)
        temp_index = range(-n, n)

        for i_sam in range(0, nP):
            index = np.flatnonzero(temp_index == ind[i_sam])
            xW[i_sam, :] = xtemp[i_sam, index:index + mP]

        if (np.max(abs(ind)) == n) and options[0] != 0:
            disp(
                'Warning: Scrolling window size "n" may not be enough wide because extreme limit has been reached')

    return xW, ind, R


def defints(xP, interv, opt):
    nP, mP = xP.shape
    sizechk = mP / interv - round_(mP / interv)
    plus = (mP / interv - round_(mP / interv)) * interv
    mbx = 'Warning: the last interval will not fulfill the selected intervals size "inter" = ' + \
        num2str(interv)
    if plus >= 0:
        mbx2 = 'Size of the last interval = ' + num2str(plus)
    else:
        mbx2 = 'Size of the last interval = ' + num2str(interv + plus)
    mbx3 = np.array([mbx, mbx2]).reshape(1, -1)
    if opt[0] == 2 and (sizechk != 0):
        uiwait(msgbox(mbx3, 'Warning', 'warn'))
    else:
        if opt[0] != 0 and (sizechk != 0):
            print(
                ' Warning: the last interval will not fulfill the selected intervals size "inter"=%g. \ Size of the last interval = %g ',
                interv, plus)
    t = cat(1, range(0, (mP + 1), interv), mP)
    if t[-2] == t[-2]:
        t[-2] = np.array([])
    t = cat(0, t[0: - 1] + 1, t[1:])
    inter = t[:].T
    return inter


def CC_FFTShift(T, X=False, options=np.array([])):
    dimX = X.shape
    dimT = T.shape

    # .reshape(1,-1)
    optionsDefault = np.array(
        [-np.fix(dimT[-1] * 0.5), np.fix(dimT[-1] * 0.5), len(T.shape) - 1, 1, np.nan])

    if len(optionsDefault) > len(options):
        o = len(options)
        options = np.append(options, np.zeros(len(optionsDefault) - o))
        options[o:] = optionsDefault[o:]

    options[np.isnan(options)] = optionsDefault[np.isnan(options)]

    if options[0] > options[1]:
        error('Lower bound for shift is larger than upper bound')

    TimeDim = int(options[2] - 1)

    # if not(isequal(dimX([2:TimeDim - 1,TimeDim + 1:end]),dimX([2:TimeDim -
    # 1,TimeDim + 1:end])))
    if dimX[TimeDim] != dimT[TimeDim]:
        error('Target and signals do not have compatible dimensions')

    # [TimeDim,2:TimeDim - 1,TimeDim + 1:ndims(X),1];
    ord_ = np.array(
        [TimeDim] +
        range(1, TimeDim) +
        range(TimeDim, len(X.shape) - 1) +
        [0]
    ).T

    X_fft = np.transpose(X, ord_)  # permute
    X_fft = np.reshape(X_fft, (dimX[TimeDim], np.prod(dimX[ord_[1:]])))

    b = np.array(
        [
            range(1, (np.prod(dimX[ord_[1:]]) + 1)),
            range(1, (np.prod(dimX[ord_[1:]]) + 1)),
            np.sqrt(np.nansum(X_fft ** 2, axis=0))
        ]).T

    # FIXME? Sparse/dense switchg
    X_fft = X_fft / b[:, -1]

    T = np.transpose(T, ord_)
    T = np.reshape(T, (dimT[TimeDim], np.prod(dimT[ord_[1:]])))
    T = Normalise(T)

    nP, mP = X_fft.shape
    nT = T.shape[0]

    flag_miss = np.any(np.isnan(X_fft[:])) or np.any(np.isnan(T[:]))

    if flag_miss:
        if len(X.shape) > 2:
            error('Multidimensional handling of missing not implemented, yet')
        MissOff = NaN(1, mP)
        for i_signal in range(0, mP):
            # RemoveNaN([1,nP],X_fft(:,i_signal)',@all);
            Limits = RemoveNaN(
                np.array([0, nP - 1]).reshape(1, -1), X_fft[:, i_signal].T, np.all)
            if not np.array_equal(Limits.shape, np.array([1, 2]).reshape(1, -1)):
                error(
                    'Missing values can be handled only if leading or trailing')
            if any(cat(1, Limits[0], mP - Limits[1]) > np.max(abs(options[0:2]))):
                error('Missing values band larger than largest admitted shift')
            MissOff[i_signal] = Limits[0]

            if MissOff[i_signal] > 1:
                X_fft[0:Limits[1] - Limits[0] + 1,
                      i_signal] = X_fft[Limits[0]:Limits[1], i_signal]
            if Limits[1] < nP:
                X_fft[(Limits[1] - Limits[0] + 1):nP, i_signal] = 0
        Limits = RemoveNaN(np.array([0, nT - 1]), T.T, np.all)
        T[0:Limits[1] - Limits[0] + 1, :] = T[Limits[0]:Limits[1],:]
        T[Limits[1] - Limits[0] + 1:nP, :] = 0
        MissOff = MissOff[0:mP] - Limits[0]

    X_fft = cat(0, X_fft, np.zeros(
        (np.max(np.abs(options[0:2])), np.prod(dimX[ord_[1:]], axis=0))
    ))

    T = cat(0, T, np.zeros(
            (np.max(np.abs(options[0:2])), np.prod(dimT[ord_[1:]], axis=0))
            ))

    len_fft = max(X_fft.shape[0], T.shape[0])
    Shift = np.arange(options[0], options[1] + 1)

    if options[0] < 0 and options[1] > 0:
        # ind = [len_fft + Options(1) + 1:len_fft,1:Options(2) + 1];
        ind = range(int(len_fft + options[0]), int(len_fft) ) + \
            range(0,  int(options[1] + 1))

    elif options[0] < 0 and options[1] < 0:
        # ind = len_fft + Options(1) + 1:len_fft + Options(2) + 1;
        ind = range(len_fft + options[0], (len_fft + options[1] + 1))

    elif options[0] < 0 and options[1] == 0:
        # ind = [len_fft + Options(1) + 1:len_fft + Options(2),1];
        ind = range(int(len_fft + options[0]),
                    int(len_fft + options[1] + 1)) + [1]

    else:
        # ind = Options(1) + 1:Options(2) + 1;
        ind = range(int(options[0]), int(options[1] + 1))

    X_fft = np.fft.fft(X_fft, len_fft, axis=0)

    T_fft = np.fft.fft(T, len_fft, axis=0)
    T_fft = np.conj(T_fft)

    # np.reshape(T_fft[:,:,np.ones( (dimX[0],1), dtype=bool )],X_fft.shape)
    T_fft = np.tile(T_fft, (1, dimX[0]))
    dT = X_fft * T_fft
    cc = np.fft.ifft(dT, len_fft, axis=0)

    # reshape(cc(ind,:),[Options(2) - Options(1) + 1,prod(dimX(ord(2:end - 1))),dimX(1)]);
    # np.array([options[1] - options[0] + 1, np.prod(dimX[ord_[1: - 1]]),
    # dimX[0]]))
    if len(ord_[1:-1]) == 0:
        k = 1
    else:
        k = np.prod(dimX[ord_[1:-1]])
    #cc=np.tile(cc[ind,:], ( options[1]-options[0]+1, k, dimX[0])  )
    cc = np.reshape(cc[ind, :], ( options[1]-options[0]+1, k, dimX[0])  )

    if options[3] == 0:
        cc = np.squeeze(np.mean(cc, axis=1))
    else:
        if options[3] == 1:
            # cc=np.squeeze(, axis=1).T
            cc = np.squeeze(np.prod(cc, axis=1))
        else:
            error('Invalid options for correlation of multivariate signals')

    pos = cc.argmax(axis=0)
    Values = cat(1, np.reshape(Shift, (len(Shift), 1)), cc)
    Shift = Shift[pos]

    if flag_miss:
        Shift = Shift + MissOff

    Xwarp = NaN(*[dimX[0]] + list(dimT[1:]))
    ind = np.tile(np.nan, (len(X.shape), 18))  # .astype(int)
    indw = ind

    TimeDim = np.array([TimeDim])

    for i_X in range(0, dimX[0]):
        ind_c = i_X
        indw_c = i_X

        if Shift[i_X] >= 0:

            ind = np.arange(Shift[i_X], dimX[TimeDim]).reshape(1, -1)
            indw = np.arange(0, dimX[TimeDim] - Shift[i_X]).reshape(1, -1)

            if options[4] == - np.inf:
                # ind{TimeDim}  =
                # cat(2,ind{TimeDim},dimX(TimeDim(ones(1,abs(Shift(i_X))))));
                o = np.zeros(abs(Shift[i_X])).astype(int)
                if len(o) > 0:

                    ind = cat(1,
                              ind,
                              np.array(dimX[TimeDim[o]] - 1).reshape(1, -1)
                              )

                    # indw{TimeDim} = cat(2,indw{TimeDim},dimX(TimeDim) -
                    # Shift(i_X) + 1:dimX(TimeDim));
                    indw = cat(1,
                               indw,
                               np.arange(dimX[TimeDim] - Shift[i_X],
                                         dimX[TimeDim]).reshape(1, -1)
                               )
        else:
            if Shift[i_X] < 0:

                # ind{TimeDim}  = 1:(dimX(TimeDim) + Shift(i_X));
                # indw{TimeDim} = -Shift(i_X) + 1:dimX(TimeDim);
                ind = np.arange(0, dimX[TimeDim] + Shift[i_X]).reshape(1, -1)
                indw = np.arange(-Shift[i_X], dimX[TimeDim]).reshape(1, -1)

                if options[4] == - np.inf:

                    # ind{TimeDim}  = cat(2,ones(1,-Shift(i_X)),ind{TimeDim});
                    ind = cat(1, np.zeros((1, -Shift[i_X])), ind)

                    # indw{TimeDim} = cat(2,1:-Shift(i_X),indw{TimeDim});
                    indw = cat(
                        1, np.arange(0, -Shift[i_X]).reshape(1, -1), indw);

        Xwarp[ind_c, indw.astype(int)] = X[ind_c, ind.astype(int)]

    # Reshape for returning
    Shift = np.reshape(Shift, (len(Shift), 1))

    return Xwarp, Shift, Values


def Delt(varargin):
    delete(varargin[0])
    return


def RemoveNaN(B, Signal, Select=np.any, flags=False):
    '''
    Rearrange segments so that they do not include NaN's
    
    [Bn] = RemoveNaN(B,  Signal,  Select)
    [An, flag]
    INPUT
    B     : (p * 2) Boundary matrix (i.e. [Seg_start(1) Seg_end(1); Seg_start(2) Seg_end(2);...]
    Signal: (n * 2) Matrix of signals (with signals on rows)
    Select: (1 * 1) function handle to selecting operator
                     e.g. np.any (default) eliminate a column from signal matrix
                                         if one or more elements are missing
                          np.all           eliminate a column from signal matrix
                                         if all elements are missing
    
    OUTPUT
    Bn  : (q * 2)     new Boundary matrix in which NaN's are removed
    flag: (q * 2 * n) flag matrix if there are NaN before (column 1) or after (column 2)
                       the corresponding segment in the n signals.
    
    Author: Giorgio Tomasi
            Giorgio.Tomasi@gmail.com
    
    Created      : 25 February,  2009
    Last modified: 23 March,  2009; 18:02
    Python implementation: Martin Fitzpatrick
                           martin.fitzpatrick@gmail.com
                           
    Last modified: 28th October,  2013
    HISTORY
    1.00.00 09 Mar 09 -> First working version
    2.00.00 23 Mar 09 -> Added output for adjacent NaN's in signals
    2.01.00 23 Mar 09 -> Added Select input parameter
    '''
    C = NaN(B.shape[0], B.shape[1] if len(B.shape) > 1 else 1)
    B = B.reshape(1, -1)
    count = 0
    Signal = np.isnan(Signal)
    for i_el in range(0, B.shape[0]):

        ind = np.arange(B[i_el, 0], B[i_el, 1] + 1)
        in_ = Select(Signal[:, ind], axis=0)

        if np.any(in_):
            # p = diff([0 in],1,2)
            p = np.diff(np.array([0] + in_).reshape(1, -1), 1, axis=1)
            a = np.flatnonzero(p < 0) + 1
            b = np.flatnonzero(p > 0)

            if not in_[0]:
                a = cat(1, np.array([0]), a)
            else:
                b = b[1:]
            if not in_[-1]:
                b = cat(1, b, np.array([max(ind.shape) - 1]))
            a = np.unique(a)
            b = np.unique(b)

            d = ind[cat(0, a[:].reshape(1, -1), b[:].reshape(1, -1))]

            C.resize(d.shape)
            C[count:count + max(a.shape), :] = d

            count = count + max(a.shape)
        else:
            C[count, :] = B[i_el,:]
            count = count + 1

    C = C.astype(int).T
    An = C

    if flags:
        # .reshape(1,-1)
        flag = np.empty((C.shape[0], 2, Signal.shape[0]), dtype=bool)

        flag[:] = False

        Cinds = C[:, 0] > 1
        Cinds = Cinds.astype(bool)

        Cinde = C[:, 1] < Signal.shape[1]
        Cinde = Cinde.astype(bool)
        flag[Cinds, 0, :] = Signal[:, C[Cinds, 0] - 1].T

        flag[Cinde, 1, :] = Signal[:, C[Cinde, 0] - 1].T
        return An, flag
    else:
        return An


def Normalise(X, Flag=False):
    '''
    Column-wise normalise matrix
    NaN's are ignored
    
    [Xn] = Normalise(X,  Flag)
    
    INPUT
    X   : Marix
    Flag: true if any NaNs are present (optional - it saves time for large matrices)
    
    OUTPUT
    Xn: Column-wise normalised matrix
    
    Author: Giorgio Tomasi
             Giorgio.Tomasi@gmail.com
    
    Created      : 09 March,  2009; 13:18
    Last modified: 09 March,  2009; 13:50

    Python implementation: Martin Fitzpatrick
                           martin.fitzpatrick@gmail.com
                           
    Last modified: 28th October,  2013
    '''

    if not Flag:
        Patt = ~np.isnan(X)
        Flag = np.any(~Patt[:])
    else:
        Patt = not np.isnan(X)

    M, N = X.shape
    Xn = NaN(M, N)
    if Flag:
        for i_n in range(0, N):
            n = np.linalg.norm(X[Patt[:, i_n], i_n])
            if not n:
                n = 1
            Xn[Patt[:, i_n], i_n] = X[Patt[:, i_n], i_n] / n
    else:
        for i_n in range(0, N):
            n = np.linalg.norm(X[:, i_n])
            if not n:
                n = 1
            Xn[:, i_n] = X[:, i_n] / n

    return Xn


def ExtractSegments(X, Segments):
    '''
    Extract segments from signals
    
    [XSeg] = ExtractSegments(X,  Segments)
    ? [XSeg, SegNew] = ExtractSegments(X,  Segments)
    
    INPUT
    X       : (n * p) data matrix
    Segments: (s * 2) segment boundary matrix
    
    OUTPUT
    XSeg: (n * q) data matrix in which segments have been removed
    SegNew: New segment layout
        
    Author: Giorgio Tomasi
             Giorgio.Tomasi@gmail.com

    Python implementation: Martin Fitzpatrick
                           martin.fitzpatrick@gmail.com
                           
    Last modified: 28th October,  2013
    
    Created      : 23 March,  2009; 07:51
    Last modified: 23 March,  2009; 15:07

    HISTORY
    0.00.01 23 Mar 09 -> Generated function with blank help
    1.00.00 23 Mar 09 -> First working version
    '''
    n, p = X.shape
    Sd = np.diff(Segments, axis=1)

    q = np.sum(Sd + 1)
    s, t = Segments.shape

    flag_si = t != 2
    flag_in = np.any(Segments[:] != np.fix(Segments[:]))
    flag_ob = np.any(Segments[:, 0] < 1) or np.any(Segments[:, 1] > p)
    flag_ni = np.any(np.diff(Segments[:, 0]) < 0) or np.any(
        np.diff(Segments[:, 1]) < 0)
    flag_ab = np.any(Sd < 2)

    if flag_si:
        error('Segment boundary matrix must have two columns')
    if flag_in:
        error('Segment boundaries must be integers')
    if flag_ob:
        error('Segment boundaries outside of segment')
    if flag_ni:
        error('Segments boundaries must be monotonically increasing')
    if flag_ab:
        error('Segments must be at least two points long')

    XSeg = NaN(n, q)
    cdim = cat(0, np.array([[0]]), np.cumsum(Sd, axis=0))
    origin = 0
    SegNew = []
    for seg in Segments:
        data = X[:, seg[0]:seg[1] + 1]
        segment_size = data.shape[1]
        XSeg[:, origin:origin + segment_size] = data

        SegNew.append([origin, origin + segment_size - 1])
        origin = origin + segment_size

    SegNew = np.array(SegNew)
    '''    
    SegNew = np.array( SegNew )
    print SegNew
    for i_seg in range(0,s):
        # XSeg(:,cdim(i_seg) + 1:cdim(i_seg + 1)) = X(:,Segments(i_seg,1):Segments(i_seg,2));
        XSeg[:,cdim[i_seg]:cdim[i_seg+1]+1]=X[:,Segments[i_seg,0]:Segments[i_seg,1]+1]
        
        print cdim[np.arange(0,s)].reshape(1,-1)
        print cdim[np.arange(1,s+1)].reshape(1,-1)

        SegNew=cat(0, cdim[np.arange(0,s)].reshape(1,-1), cdim[np.arange(1,s+1)].reshape(1,-1) )

    '''

    return XSeg, SegNew


def scal2pts(ppmi,  ppm=[],  prec=None):
    """
    Transforms scalars in data points
    
    pts = scal2pts(values, scal)
    
    INPUT
    values: scalars whose position is sought
    scal  : vector scalars
    prec  : precision (optional) to handle endpoints
    
    OUTPUT
    pts   : position of the requested scalars (NaN if it is outside of 'scal')
    
    Author: Giorgio Tomasi
            Giorgio.Tomasi@gmail.com
    
    Created      : 12 February,  2009; 17:43
    Last modified: 11 March,  2009; 15:14

    Python implementation: Martin Fitzpatrick
                           martin.fitzpatrick@gmail.com
                           
    Last modified: 28th October,  2013

    HISTORY
    1.00.00 12 Feb 09 -> First working version
    1.01.00 11 Mar 09 -> Added input parameter check
    """
    if prec == None:
        prec = min(abs(unique(np.diff(ppm))))

    dimppmi = ppmi.shape
    ppmi = ppmi[:]
    ppm = ppm[:]
    rev = ppm[0] > ppm[1]
    if rev:
        ppm = ppm[-1:0:1]
    ubound = (ppmi - ppm[-1]) < prec & (ppmi - ppm[-2]) > 0
    lbound = (ppm[0] - ppmi) < prec & (ppm[0] - ppmi) > 0
    ppmi[(ubound - 1)] = ppm[-1]
    ppmi[(lbound - 1)] = ppm[0]
    if nargin < 2:
        error('Not enough input arguments')

    if max(ppmi.shape) > max(ppm.shape):
        warning('icoshift:scal2pts', "ppm vector is shorter than the value's")

    xxi, k = sort(ppmi[:])
    nil, j = sort(np.array([ppm[:], xxi[:]]).reshape(1, -1))
    r[j] = range(1, (max(j.shape) + 1))
    r = r[(max(ppm.shape) + 1):] - range(1, max(ppmi.shape) + 1)
    r[k] = r
    r[ppmi == ppm[-1]] = max(ppm.shape)
    ind = np.flatnonzero((r > 0) & (r <= max(ppm.shape)))
    ind = ind[:]
    pts = Inf(ppmi.shape)
    pts[ind] = r[ind]
    ptsp1 = np.min(max(ppm.shape), abs(pts + 1))
    ptsm1 = np.max(1, abs(pts - 1))
    ind = np.flatnonzero(isfinite(pts))
    dp0 = abs(ppm[pts[ind]] - ppmi[ind])
    dpp1 = abs(ppm[ptsp1[ind]] - ppmi[ind])
    dpm1 = abs(ppm[ptsm1[ind]] - ppmi[ind])
    pts[ind[dpp1 < dp0]] = pts[ind[dpp1 < dp0]] + 1
    pts[ind[dpm1 < dp0]] = pts[ind[dpm1 < dp0]] - 1
    if (0 in pts.shape):
        pts = np.array([])
    pts[~isfinite(pts)] = NaN
    if rev:
        pts = max(ppm.shape) - pts + 1
    if not np.array_equal(pts.shape, ppmi):
        pts = reshape(pts, dimppmi)

    return pts


def dscal2dpts(*varargin):
    """
    Translates an interval width from scal to the best approximation in sampling points.

    I = dppm2dpts(Delta, scal, prec)

    INPUT
    Delta: interval widths in scale units
    scal : scale
    prec : precision on the scal axes

    OUTPUT
    I: interval widths in sampling points
    
    Author: Giorgio Tomasi
            Giorgio.Tomasi@gmail.com
    
    Last modified: 21st February,  2009

    Python implementation: Martin Fitzpatrick
                           martin.fitzpatrick@gmail.com
                           
    Last modified: 28th October,  2013
    """
    nargin = len(varargin)
    if nargin > 0:
        d = varargin[0]
    if nargin > 1:
        ppm = varargin[1]
    if nargin > 2:
        varargin = varargin[2]
    if (0 in d.shape):
        I = np.array([])
        return I
    if nargin < 2:
        error('Not enough input arguments')
    if d <= 0:
        error('Delta must be positive')
    if ppm[0] < ppm[1]:
        I = scal2pts(ppm[0] + d, ppm, varargin[:]) - 1
    else:
        I = max(ppm.shape) - scal2pts(ppm[-2] + d, ppm, varargin[:]) + 1
    return I


def warning(msg):
    print msg
    return True


def error(msg):
    print msg
    return False


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def cat(dim, *args):
    return np.concatenate(args, axis=dim)


def sortrows(a, i):
    I = np.argsort(a[:, i])
    b = a[I, :]
    return b


def NaN(r, c):
    a = np.empty((r, c))
    a[:] = np.nan
    return a
