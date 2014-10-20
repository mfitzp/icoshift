from __future__ import division, print_function
import numpy
import scipy as sp
from scipy.stats import nanmean, nanmedian
from copy import copy
import sys
import logging


try:
    basestring
except NameError:
    basestring = str


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def cat(dim, *args):
    return numpy.concatenate([r for r in args if r.shape[0] > 0], axis=dim)


def sortrows(a, i):
    i = numpy.argsort(a[:, i])
    b = a[i, :]
    return b


def nans(r, c):
    a = numpy.empty((r, c))
    a[:] = numpy.nan
    return a


def min_with_indices(d):
    d = d.flatten()
    ml = numpy.min(d)
    mi = numpy.array(list(d).index(ml))
    return ml, mi


def max_with_indices(d):
    d = d.flatten()
    ml = numpy.max(d)
    mi = numpy.array(list(d).index(ml))
    return ml, mi


def icoshift(xt,  xp,  inter='whole',  n='f', scale=None, coshift_preprocessing=False,
             coshift_preprocessing_max_shift=None, fill_with_previous=True, average2_multiplier=3):
    '''
    interval Correlation Optimized shifting
    [xcs, ints, ind, target] = icoshift(xt, xp, inter[, n[, options[, scale]]])
    Splits a spectral database into "inter" intervals and coshift each vector
    left-right to get the maximum correlation toward a reference or toward an
    average spectrum in that interval. Missing parts on the edges after
    shifting are filled with "closest" value or with "NaNs".
    INPUT
    xt (1 * mt)    : target vector.
                     Use 'average' if you want to use the average spectrum as a reference
                     Use 'median' if you want to use the median spectrum as a reference
                     Use 'max' if you want to use for each segment the corresponding actual spectrum having max features as a reference
                     Use 'average2' for using the average of the average multiplied for a requested number (default=3) as a reference

    xp (np * mp)   : Matrix of sample vectors to be aligned as a sample-set
                     towards common target xt
    inter          : definition of alignment mode
                     'whole'         : it works on the whole spectra (no intervals).
                     nint            : (numeric) number of many intervals.
                     'ndata'         : (string) length of regular intervals
                                       (remainders attached to the last).
                     [I1s I1e, I2s...]: interval definition. ('i(n)s' interval
                                       n start,  'i(n)e' interval n end).
                     (refs:refe)     : shift the whole spectra according to a
                                       reference signal(s) in the region
                                       refs:refe (in sampling points)
                     'refs-refe'     : `shift the whole spectra according to a
                                       reference signal(s) in the region
                                       refs-refe (in scale units)
    n (1 * 1)      : (optional)
                     n = integer n.: maximum shift correction in data
                                     points/scale units (cf. options[4])
                                     in x/rows. It must be >0
                     n = 'b' (best): the algorithm search for the best n
                                     for each interval (it can be time consuming!)
                     n = 'f' (fast): fast search for the best n for each interval (default)
                     a logging.warn is displayed for each interval if "n" appears too small
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
                         1 :
                     (3)
                         it has to be given in scale units if option(5)=1
                     (4) 0 : intervals are given in No. of datapoints  (deafult)
                         1 : intervals are given in ppm --> use scale for inter and n
    scale           : vector of scalars used as axis for plot (optional)
    coshift_preprocessing (bool) (optional; default=False): Execute a Co-shift step before carrying out iCOshift
    coshift_preprocessing_max_shift (int) (optional): Max allowed shift for the Co-shift preprocessing
                                                      (default = equal to n if not specified)
    fill_with_previous (bool) (optional; default=True): Fill using previous point (default); set to False to np.nan
    average2_multiplier (int) (optional; default=3): Multiplier used for the average2 algorithm

    OUTPUT
    xcs  (np * mt): shift corrected vector or matrix
    ints (ni * 4) : defined intervals (Int. No.,  starting point,  ending point,  size)
    ind  (np * ni): matrix of indexes reporting how many points each spectrum
                    has been shifted for each interval (+ left,  - right)
    target (1 x mp): actual target used for the final alignment

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

    Python implementation by:
    Martin Fitzpatrick -  Rheumatology Research Group
                          Centre for Translational Inflammation Research
                          School of Immunity and Infection
                          University of Birmingham - United Kingdom
    email: martin.fitzpatrick@gmail.com

    170508 (FrSa) first working code
    211008 (FrSa) improvements and bugs correction
    111108 (Frsa) Splitting into regular intervals (number of intervals or wideness in datapoints) implemented
    141108 (GT)   FFT alignment function implemented
    171108 (FrSa) options implemented
    241108 (FrSa) Automatic search for the best or the fastest n for each interval implemented
    261108 (FrSa) Plots improved
    021208 (FrSa) 'whole' case added & robustness improved
    050309 (GT)   Implentation of interpolation modes (nan); Cosmetics; Graphics
    240309 (GT)   Fixed bug in handling missing values
    060709 (FrSa) 'max' target and output 'target' added. Some speed,  plot and robustness improvements
    241109 (GT)   Interval and band definition in units (added options[4])
    021209 (GT)   Minor debugging for the handling of options[4]
    151209 (FrSa) Cosmetics and minor debugging for the handling of options[4]
    151110 (FrSa) Option 'Max' works now also for alignment towards a reference signal
    310311 (FrSa) Bugfix for the 'whole' case when mp < 101
    030712 (FrSa) Introducing the 'average2' xt (target) for a better automatic target definition. Plots updated to include also this case
    281023 (MF)   Initial implementation of Python version of icoshift algorithm. PLOTS NOT INCLUDED
    '''

    # RETURNS [xcs, ints, ind, target]

    # Take a copy of the xp vector since we mangle it somewhat
    xp = copy(xp)


    if scale is None:
        using_custom_scale = False
        scale = numpy.array(range(0, xp.shape[1]))

    else:
        using_custom_scale = True

        dec_scale = numpy.diff(scale)
        inc_scale = scale[0] - scale[1]

        flag_scale_dir = inc_scale < 0
        flag_di_scale = numpy.any(abs(dec_scale) > 2 * numpy.min(abs(dec_scale)))

        if len(scale) != max(scale.shape):
            raise(Exception, 'scale must be a vector')

        if max(scale.shape) != xp.shape[1]:
            raise(Exception, 'x and scale are not of compatible length %d vs. %d' % (max(scale.shape), xp.shape[1]))

        if inc_scale == 0 or not numpy.all(numpy.sign(dec_scale) == - numpy.sign(inc_scale)):
            raise(Exception, 'scale must be strictly monotonic')

    if coshift_preprocessing_max_shift is None:
        coshift_preprocessing_max_shift = n

    # ERRORS CHECK
    # Constant
    # To avoid out of memory errors when 'whole',  the job is divided in
    # blocks of 32MB
    block_size = 2 ** 25


    max_flag = False
    avg2_flag = False

    xt_basis = xt

    if xt == 'average':
        xt = nanmean(xp, axis=0).reshape(1, -1)

    elif xt == 'median':
            xt = nanmedian(xp, axis=0).reshape(1, -1)

    elif xt == 'average2':
            xt = nanmean(xp, axis=0).reshape(1,-1)
            avg2_flag = True

    elif xt == 'max':
            xt = numpy.zeros((1, xp.shape[1]))
            max_flag = True

    nt, mt = xt.shape
    np, mp = xp.shape

    if mt != mp:
        raise(Exception, 'Target "xt" and sample "xp" must have the same number of columns')

    if is_number(inter):
        if inter > mp:
            raise(Exception, 'Number of intervals "inter" must be smaller than number of variables in xp')

    # Set defaults if the settings are not set
    # options = [options[oi] if oi < len(options) else d for oi, d in enumerate([1, 1, 0, 0, 0]) ]

    if using_custom_scale:
        prec = abs(numpy.min(numpy.unique(dec_scale)))
        if flag_di_scale:
            logging.warn('Scale vector is not continuous, the defined intervals might not reflect actual ranges')

    flag_coshift = (not inter == 'whole') and coshift_preprocessing

    if flag_coshift:

        if using_custom_scale:
            coshift_preprocessing_max_shift = dscal2dpts(coshift_preprocessing_max_shift, scale, prec)

        if max_flag:
            xt = nanmean(xp, axis=0)

        co_shift_scale = scale if using_custom_scale else None
        xp, nil, wint, _ = icoshift(xt, xp, 'whole',
                                    coshift_preprocessing=False,
                                    coshift_preprocessing_max_shift=coshift_preprocessing_max_shift,
                                    scale=co_shift_scale,
                                    fill_with_previous=True,
                                    average2_multiplier=average2_multiplier )

        if xt_basis == 'average' or xt_basis == 'average2':
            xt = nanmean(xp).reshape(1, -1)

        elif xt_basis == 'median':
            xt = nanmedian(xp).reshape(1, -1)

        else:  # max?
            xt = xt.reshape(1,-1)


    whole = False
    flag2 = False

    if isinstance(inter, basestring):

        if inter == 'whole':
            inter = numpy.array([0, mp - 1]).reshape(1, -1)
            whole = True

        elif '-' in inter:
            interv = regexp(inter, '(-{0,1}\\d*\\.{0,1}\\d*)-(-{0,1}\\d*\\.{0,1}\\d*)', 'tokens')
            interv = sort(scal2pts(float(cat(0, interv[:])), scale, prec))

            if interv.size != 2:
                raise(Exception, 'Invalid range for reference signal')

            inter = range(interv[0], (interv[1] + 1))
            flag2 = True

        else:

            interv = float(inter)

            if using_custom_scale:
                interv = dscal2dpts(interv, scale, prec)
            else:
                interv = round(interv)

            inter = defints(xp, interv, options[0])

    elif isinstance(inter, int):

        # Build interval list
        # e.g. 5 intervals on 32768 to match MATLAB algorithm should be
        #0, 6554, 6554, 13108, 13108, 19662, 19662, 26215, 26215, 32767

        # Distributes vars_left_over in the first "vars_left_over" intervals
        # startint     = [(1:(N+1):(remain - 1)*(N+1)+1)'; ((remain - 1) * (N + 1) + 1 + 1 + N:N:mP)'];

        remain = mp % inter
        step = int( float(mp) / inter )
        segments = []
        o = 0

        while o < mp:
            segments.extend([o, o])
            if remain > 0:
                o += 1
                remain -=1
            o += step

        # Chop of duplicate zero
        segments = segments[1:]
        segments.append(mp)  # Add on final step
        inter = numpy.array(segments, dtype=int).reshape(1, -1)

        logging.info("Calculated intervals: %s" % inter)

    elif isinstance(inter, list):  # if is a list of tuples ; add else
        inter = np.array(inter)

        flag2 = numpy.array_equal(numpy.fix(inter), inter) and max(inter.shape) > 1 and numpy.array_equal(
            numpy.array([1, numpy.max(inter) - numpy.min(inter) + 1]).reshape(1, -1), inter.shape) and numpy.array_equal(unique(numpy.diff(inter, 1, 2)), 1)

        if not flag2 and using_custom_scale:
            inter = scal2pts(inter, scale, prec)

            if numpy.any(inter[0:2:] > inter[1:2:]) and not flag_scale_dir:
                inter = flipud(numpy.reshape(inter, 2, max(inter.shape) / 2))
                inter = inter[:].T

    else:
        raise(Exception, 'The number of intervals must be "whole", an integer, or a list of tuples of integers')


    nint, mint = inter.shape
    scfl = numpy.array_equal(numpy.fix(scale), scale) and not using_custom_scale

    if isinstance(inter, basestring) and n not in ['b', 'f']:
        raise(Exception, '"n" must be a scalar b or f')

    elif isinstance(n, int) or isinstance(n, float):
        if n <= 0:
            raise(Exception, 'Shift(s) "n" must be larger than zero')

        if scfl and not isinstance(n, int):
            logging.warn('"n" must be an integer if scale is ignored; first element (i.e. %d) used' % n)
            n = numpy.round(n)
        else:
            if using_custom_scale:
                n = dscal2dpts(n, scale, prec)

        if not flag2 and numpy.any(numpy.diff(numpy.reshape(inter, (2, mint // 2)), 1, 0) < n):
            raise(Exception, 'Shift "n" must be not larger than the size of the smallest interval')

    flag = numpy.isnan(cat(0, xt.reshape(1, -1), xp))
    frag = False
    ref = lambda e: numpy.reshape(e, (2, max(e.shape) / 2)).T
    vec = lambda a: a.flatten()

    mi, pmi = min_with_indices(inter)
    ma, pma = max_with_indices(inter)

    # There are missing values in the dataset; so remove them before starting
    # if they line up between datasets
    if vec(flag).any():

        if numpy.array_equal(flag[numpy.ones((np, 1), dtype=int), :], flag[1:,:]):
            select = numpy.any
        else:
            select = numpy.all

        if flag2:
            intern_ = remove_nan(
                numpy.array([0, pma - pmi]).reshape(1, -1), cat(0, xt[:, inter], xp[:, inter]), select)
            if intern_.shape[0] != 1:
                raise(Exception, 'Reference region contains a pattern of missing that cannot be handled consistently')

            elif not numpy.array_equal(intern_, numpy.array([1, inter[-2] - inter[0] + 1]).reshape(1, -1)):
                logging.warn('The missing values at the boundaries of the reference region will be ignored')

            intern_ = range(inter[0] + intern_[0], (inter[0] + intern_[1] + 1))
        else:
            intern_, flag_nan = remove_nan(
                ref(inter), cat(0, xt, xp), select, flags=True)
            intern_ = vec(intern_.T).T

        if 0 in intern_.shape:
            raise(Exception, 'Cannot handle this pattern of missing values.')

        if max(intern_.shape) != max(inter.shape) and not flag2:
            if whole:
                if max(intern_.shape) > 2:

                    xseg, in_or = extract_segments(cat(0, xt, xp), ref(intern_))
                    InOrf = in_or.flatten()
                    inter = numpy.array([InOrf[0], InOrf[-1] - 1]).reshape(1, -1)
                    in_or = cat(1, ref(intern_), in_or)
                    xp = xseg[1:, :]
                    xt = xseg[0, :].reshape(1, -1)
                    frag = True

            else:
                logging.warn('To handle the pattern of missing values, %d segments are created/removed' % (abs(max(intern_.shape) - max(inter.shape)) / 2) )
                inter = intern_
                nint, mint = inter.shape
    xcs = xp
    mi, pmi = min_with_indices(inter)
    ma, pma = max_with_indices(inter)


    flag = max(inter.shape) > 1 and numpy.array_equal(
        numpy.array([1, pma - pmi + 1]).reshape(1, -1), inter.shape) and numpy.array_equal(unique(numpy.diff(inter, 1, 2)), 1)

    if flag:
        if n == 'b':
            logging.info('Automatic searching for the best "n" for the reference window "ref_w" enabled. That can take a longer time.')

        elif n == 'f':
            logging.info('Fast automatic searching for the best "n" for the reference window "ref_w" enabled.')

        if max_flag:
            amax, bmax = max_with_indices( numpy.sum(xp) )
            xt[mi:ma] = xp[bmax, mi:ma]

        ind = nans(np, 1)
        missind = not all(numpy.isnan(xp), 2)
        xcs[missind, :], ind[missind], _ = coshifta(xt, xp[missind,:], inter, n, fill_with_previous=fill_with_previous,
                                                    block_size=block_size)
        ints = numpy.array([1, mi, ma]).reshape(1, -1)

    else:
        if mint > 1:
            if mint % 2:
                raise(Exception, 'Wrong definition of intervals ("inter")')

            if ma > mp:
                raise(Exception, 'Intervals ("inter") exceed samples matrix dimension')

            # allint=[(1:round(mint/2))' inter(1:2:mint)' inter(2:2:mint)'];
            # allint =
            #        1           1        6555
            #        2        6555       13109
            #        3       13109       19663
            #        4       19663       26216
            #        5       26216       32768
            # ans =
            #  5     3

            inter_list = list(inter.flatten())

            allint = numpy.array([
                range(mint//2),
                inter_list[0::2],
                inter_list[1::2],
            ])

            allint = allint.T

        sinter = numpy.sort(allint, axis=0)
        intdif = numpy.diff(sinter)

        if numpy.any(intdif[1:2:max(intdif.shape)] < 0):
            logging.warn('The user-defined intervals are overlapping: is that intentional?')

        ints = allint
        ints = numpy.append(ints, ints[:, 2] - ints[:, 1])
        ind = numpy.zeros((np, allint.shape[0]))

        if n == 'b':
            logging.info('Automatic searching for the best "n" for each interval enabled. This can take a long time...')

        elif n == 'f':
            logging.info('Fast automatic searching for the best "n" for each interval enabled')

        for i in range(0, allint.shape[0]):

            if whole:
                logging.info('Co-shifting the whole %s samples...' % np)
            else:
                logging.info('Co-shifting interval no. %s of %s...' % (i, allint.shape[0]) )

            # FIXME? 0:2, or 1:2?
            intervalnow = xp[:, allint[i, 1]:allint[i, 2]]

            if max_flag:
                amax, bmax = max_with_indices(numpy.sum(intervalnow, axis=1))
                target = intervalnow[bmax, :]
                xt[0, allint[i, 1]:allint[i, 2]] = target
            else:
                target = xt[:, allint[i, 1]:allint[i, 2]]

            missind = ~numpy.all(numpy.isnan(intervalnow), axis=1)

            if not numpy.all(numpy.isnan(target)) and numpy.any(missind):

                cosh_interval, loc_ind, _ = coshifta(target, intervalnow[missind, :], 0, n,
                                                     fill_with_previous=fill_with_previous, block_size=block_size)
                xcs[missind, allint[i, 1]:allint[i, 2]] = cosh_interval
                ind[missind, i] = loc_ind.flatten()

            else:
                xcs[:, allint[i, 1]:allint[i, 1]] = intervalnow

        if avg2_flag:

            for i in range(0, allint.shape[0]):
                if whole:
                    logging.info('Co-shifting again the whole %d samples... ' % np)
                else:
                    logging.info('Co-shifting again interval no. %d of %d... ' % (i, allint.shape[0]))

                intervalnow = xp[:, allint[i, 1]:allint[i, 2]]
                target1 = numpy.mean(xcs[:, allint[i, 1]:allint[i, 2]], axis=0)
                min_interv = numpy.min(target1)
                target = (target1 - min_interv) * average2_multiplier
                missind = ~numpy.all(numpy.isnan(intervalnow), 1)

                if (not numpy.all(numpy.isnan(target))) and (numpy.sum(missind) != 0):
                    cosh_interval, loc_ind, _ = coshifta(target, intervalnow[missind, :], 0, n,
                                                         fill_with_previous=fill_with_previous, block_size=block_size)
                    xcs[missind, allint[i, 1]:allint[i, 2]] = cosh_interval

                    xt[0, allint[i, 1]:allint[i, 2]] = target
                    ind[missind, i] = loc_ind.T

                else:
                    xcs[:, allint[i, 1]:allint[i, 2]] = intervalnow

    if frag:

        xn = nans(np, mp)
        for i_sam in range(0, np):
            for i_seg in range(0, in_or.shape[0]):
                xn[i_sam, in_or[i_seg, 0]:in_or[i_seg, 1]
                    + 1] = xcs[i_sam, in_or[i_seg, 2]:in_or[i_seg, 3] + 1]
                if loc_ind[i_sam] < 0:
                    if flag_nan[i_seg, 0, i_sam]:
                        xn[i_sam, in_or[i_seg, 0]:in_or[i_seg, 0]
                            - loc_ind[i_sam, 0] + 1] = numpy.nan
                else:
                    if loc_ind[i_sam] > 0:
                        if flag_nan[i_seg, 1, i_sam]:
                            xn[i_sam, (in_or[i_seg, 1] - loc_ind[i_sam, 0] + 1):in_or[i_seg, 1]+1] = numpy.nan


        xcs = xn
    target = xt

    if flag_coshift:
        ind = ind + wint * numpy.ones( (1, ind.shape[1]) )

    return xcs, ints, ind, target


def coshifta(xt, xp, ref_w=0, n=numpy.array([1, 2, 3]), fill_with_previous=True, block_size=(2 ** 25)):

    if ref_w == 0 or ref_w.shape[0] == 0:
        ref_w = numpy.array([0])

    if numpy.all(ref_w >= 0):
        rw = max(ref_w.shape)

    else:
        rw = 1

    if fill_with_previous:
        filling = -numpy.inf

    else:
        filling = numpy.nan

    if xt == 'average':
        xt = nanmean(xp, axis=0)

    # Make two dimensional
    xt = xt.reshape(1, -1)

    nt, mt = xt.shape
    np, mp = xp.shape

    if len(ref_w.shape) > 1:
        nr, mr = ref_w.shape
    else:
        nr, mr = ref_w.shape[0], 0

    logging.info('mt=%d, mp=%d' % (mt, mp))

    if mt != mp:
        raise(Exception, 'Target "xt" and sample "xp" must be of compatible size (%d, %d)' % (mt, mp) )

    if numpy.any(n <= 0):
        raise(Exception, 'shift(s) "n" must be larger than zero')

    if nr != 1:
        raise(Exception, 'Reference windows "ref_w" must be either a single vector or 0')

    if rw > 1 and (numpy.min(ref_w) < 1) or (numpy.max(ref_w) > mt):
        raise(Exception, 'Reference window "ref_w" must be a subset of xp')

    if nt != 1:
        raise(Exception, 'Target "xt" must be a single row spectrum/chromatogram')

    auto = 0
    if n == 'b':
        auto = 1
        if rw != 1:
            n = int(0.05 * mr)
            n = 10 if n < 10 else n
            src_step = int(mr * 0.05)
        else:
            n = int(0.05 * mp)
            n = 10 if n < 10 else n
            src_step = int(mp * 0.05)
        try_last = 0

    elif n == 'f':

        auto = 1
        if rw != 1:
            n = mr - 1
            src_step = numpy.round(mr / 2) - 1
        else:
            n = mp - 1
            src_step = numpy.round(mp / 2) - 1
        try_last = 1

    if nt != 1:
        raise(Exception, 'ERROR: Target "xt" must be a single row spectrum/chromatogram')

    xw = nans(np, mp)
    ind = numpy.zeros((1, np))

    n_blocks = int(numpy.ceil(sys.getsizeof(xp) / block_size))
    sam_xblock = numpy.array([int(np / n_blocks)])

    sam_xblock = sam_xblock.T

    ind_blocks = sam_xblock[numpy.ones(n_blocks, dtype=bool)]
    ind_blocks[0:np % sam_xblock] = sam_xblock + 1
    ind_blocks = numpy.array([0, numpy.cumsum(ind_blocks, 0)]).flatten()

    if auto == 1:
        while auto == 1:
            if filling == -numpy.inf:
                xtemp = cat(1, numpy.tile(xp[:, :1], (1., n)),
                            xp, numpy.tile(xp[:, -1:, ], (1., n)))

            elif numpy.isnan(filling):
                # FIXME
                xtemp = cat(1,
                    nans(np, n), xp, nans(np, n))

            if rw == 1:
                ref_w = numpy.arange(0, mp) #.reshape(1,-1)

            ind = nans(np, 1)
            r = False


            for i_block in range(0, n_blocks):
                block_indices = range(
                    ind_blocks[i_block], ind_blocks[i_block + 1])

                _, ind[block_indices], ri = cc_fft_shift(xt[0, ref_w].reshape(1,-1), xp[block_indices, :][:, ref_w],
                                                         numpy.array([-n, n, 2, 1, filling]) )

                if not r:
                    r = numpy.empty((0, ri.shape[1]))
                r = cat(0, r, ri).T

            temp_index = range(-n, n+1)

            for i_sam in range(0, np):
                index = numpy.flatnonzero(temp_index == ind[i_sam])
                xw[i_sam, :] = xtemp[i_sam, index:index + mp]

            if (numpy.max(abs(ind)) == n) and try_last != 1:
                if n + src_step >= ref_w.shape[0]:
                    try_last = 1
                    continue
                n += src_step
                continue

            else:
                if (numpy.max(abs(ind)) < n) and n + src_step < len(ref_w) and try_last != 1:
                    n += src_step
                    try_last = 1
                    continue
                else:
                    auto = 0
                    logging.info('Best shift allowed for this interval = %d' % n)

    else:
        if filling == -numpy.inf:
            xtemp = cat(1, numpy.tile(xp[:, :1], (1., n)),
                        xp, numpy.tile(xp[:, -1:, ], (1., n)))

        elif numpy.isnan(filling):
            xtemp = cat(1,
                nans(np, n), xp, nans(np, n))


        if rw == 1:
            ref_w = numpy.arange(0, mp) #.reshape(1,-1)

        ind = nans(np, 1)
        r = numpy.array([])
        for i_block in range(n_blocks):

            block_indices = range(ind_blocks[i_block], ind_blocks[i_block + 1])
            dummy, ind[block_indices], ri = cc_fft_shift(xt[0, ref_w].reshape(1,-1), xp[block_indices, :][:, ref_w],
                                                         numpy.array([-n, n, 2, 1, filling]))
            r = cat(0, r, ri)

        temp_index = numpy.arange(-n, n+1)

        for i_sam in range(0, np):
            index = numpy.flatnonzero(temp_index == ind[i_sam])
            xw[i_sam, :] = xtemp[i_sam, index:index + mp]

        if numpy.max(abs(ind)) == n:
            logging.warn('Scrolling window size "n" may not be enough wide because extreme limit has been reached')

    return xw, ind, r


def defints(xp, interv):
    np, mp = xp.shape
    sizechk = mp / interv - round(mp / interv)
    plus = (mp / interv - round(mp / interv)) * interv
    logging.warn('The last interval will not fulfill the selected intervals size "inter" = %f' % interv)

    if plus >= 0:
        logging.warn('Size of the last interval = %d ' % plus)
    else:
        logging.warn('Size of the last interval = %d' % (interv + plus))

    if sizechk != 0:
        logging.info('The last interval will not fulfill the selected intervals size "inter"=%f.' % interv)
        logging.info('Size of the last interval = %f ' % plus)

    t = cat(1, range(0, (mp + 1), interv), mp)
    if t[-2] == t[-2]:
        t[-2] = numpy.array([])

    t = cat(0, t[0: - 1] + 1, t[1:])
    inter = t[:].T
    return inter


def cc_fft_shift(t, x=False, options=numpy.array([])):

    dim_x = numpy.array(x.shape)
    dim_t = numpy.array(t.shape)

    options_default = numpy.array([-numpy.fix(dim_t[-1] * 0.5), numpy.fix(dim_t[-1] * 0.5), len(t.shape) - 1, 1, numpy.nan])
    options = numpy.array([options[oi] if oi < len(options) else d for oi, d in enumerate(options_default)])
    options[numpy.isnan(options)] = options_default[numpy.isnan(options)]

    if options[0] > options[1]:
        raise(Exception, 'Lower bound for shift is larger than upper bound')

    time_dim = int(options[2] - 1)

    if dim_x[time_dim] != dim_t[time_dim]:
        raise(Exception, 'Target and signals do not have compatible dimensions')

    ord_ = numpy.array(
        [time_dim] +
        range(1, time_dim) +
        range(time_dim, len(x.shape) - 1) +
        [0]
    ).T

    x_fft = numpy.transpose(x, ord_)  # permute
    x_fft = numpy.reshape(x_fft, (dim_x[time_dim], numpy.prod(dim_x[ord_[1:]])))

    # FIXME? Sparse/dense switchg
    p = numpy.arange(0, numpy.prod(dim_x[ ord_[1:] ] ) )
    s = numpy.max(p) + 1
    b = sp.sparse.dia_matrix( (1.0/numpy.sqrt(numpy.nansum(x_fft ** 2, axis=0)), [0]), shape=(s,s) ).todense()
    x_fft = numpy.dot(x_fft, b)

    t = numpy.transpose(t, ord_)
    t = numpy.reshape(t, (dim_t[time_dim], numpy.prod(dim_t[ord_[1:]])))
    t = normalise(t)

    np, mp = x_fft.shape
    nt = t.shape[0]
    flag_miss = numpy.any(numpy.isnan(x_fft)) or numpy.any(numpy.isnan(t))

    if flag_miss:

        if len(x.shape) > 2:
            raise(Exception, 'Multidimensional handling of missing not implemented, yet')
        miss_off = nans(1, mp)

        for i_signal in range(0, mp):

            limits = remove_nan(
                numpy.array([0, np - 1]).reshape(1, -1), x_fft[:, i_signal].reshape(1, -1), numpy.all)

            if limits.shape != (2, 1):
                raise(Exception, 'Missing values can be handled only if leading or trailing')

            if numpy.any(cat(1, limits[0], mp - limits[1]) > numpy.max(abs(options[0:2]))):
                raise(Exception, 'Missing values band larger than largest admitted shift')

            miss_off[i_signal] = limits[0]

            if numpy.any(miss_off[i_signal-1] > 1):
                x_fft[0:limits[1] - limits[0] + 1,
                      i_signal] = x_fft[limits[0]:limits[1], i_signal]

            if limits[1] < np:
                x_fft[(limits[1] - limits[0] + 1):np, i_signal] = 0

        limits = remove_nan(numpy.array([0, nt - 1]), t.T, numpy.all)
        t[0:limits[1] - limits[0] + 1, :] = t[limits[0]:limits[1],:]
        t[limits[1] - limits[0] + 1:np, :] = 0
        miss_off = miss_off[0:mp] - limits[0]

    x_fft = cat(0, x_fft, numpy.zeros(
        (numpy.max(numpy.abs(options[0:2])), numpy.prod(dim_x[ord_[1:]], axis=0))
    ))

    t = cat(0, t, numpy.zeros(
            (numpy.max(numpy.abs(options[0:2])), numpy.prod(dim_t[ord_[1:]], axis=0))
            ))

    len_fft = max(x_fft.shape[0], t.shape[0])
    shift = numpy.arange(options[0], options[1] + 1)

    if (options[0] < 0) and (options[1] > 0):
        ind = range(int(len_fft + options[0]), int(len_fft)) + \
            range(0,  int(options[1] + 1))

    elif (options[0] < 0) and (options[1] < 0):
        ind = range(len_fft + options[0], (len_fft + options[1] + 1))

    elif (options[0] < 0) and (options[1] == 0):
        ind = range(int(len_fft + options[0]),
                    int(len_fft + options[1] + 1)) + [1]

    else:
        # ind = Options(1) + 1:Options(2) + 1
        ind = range(int(options[0]), int(options[1] + 1))

    # Pad to next ^2 for performance on the FFT
    fft_pad = int( 2**numpy.ceil( numpy.log2(len_fft) ) )

    x_fft = numpy.fft.fft(x_fft, fft_pad, axis=0)
    t_fft = numpy.fft.fft(t, fft_pad, axis=0)
    t_fft = numpy.conj(t_fft)
    t_fft = numpy.tile(t_fft, (1, dim_x[0]))

    dt = x_fft * t_fft
    cc = numpy.fft.ifft(dt, fft_pad, axis=0)

    if len(ord_[1:-1]) == 0:
        k = 1
    else:
        k = numpy.prod(dim_x[ord_[1:-1]])

    cc = numpy.reshape(cc[ind, :], ( options[1]-options[0]+1, k, dim_x[0])  )

    if options[3] == 0:
        cc = numpy.squeeze(numpy.mean(cc, axis=1))
    else:
        if options[3] == 1:
            cc = numpy.squeeze(numpy.prod(cc, axis=1))
        else:
            raise(Exception, 'Invalid options for correlation of multivariate signals')

    pos = cc.argmax(axis=0)
    values = cat(1, numpy.reshape(shift, (len(shift), 1)), cc)
    shift = shift[pos]

    if flag_miss:
        shift = shift + miss_off

    x_warp = nans(*[dim_x[0]] + list(dim_t[1:]))
    ind = numpy.tile(numpy.nan, (len(x.shape), 18))
    indw = ind

    time_dim = numpy.array([time_dim])

    for i_X in range(0, dim_x[0]):
        ind_c = i_X

        if shift[i_X] >= 0:

            ind = numpy.arange(shift[i_X], dim_x[time_dim]).reshape(1, -1)
            indw = numpy.arange(0, dim_x[time_dim] - shift[i_X]).reshape(1, -1)

            if options[4] == - numpy.inf:

                o = numpy.zeros(abs(shift[i_X])).astype(int)
                if len(o) > 0:

                    ind = cat(1,
                              ind,
                              numpy.array(dim_x[time_dim[o]] - 1).reshape(1, -1)
                              )

                    indw = cat(1,
                               indw,
                               numpy.arange(dim_x[time_dim] - shift[i_X],
                                         dim_x[time_dim]).reshape(1, -1)
                               )
        elif shift[i_X] < 0:

            ind = numpy.arange(0, dim_x[time_dim] + shift[i_X]).reshape(1, -1)
            indw = numpy.arange(-shift[i_X], dim_x[time_dim]).reshape(1, -1)

            if options[4] == - numpy.inf:

                ind = cat(1, numpy.zeros((1, -shift[i_X])), ind)
                indw = cat( 1, numpy.arange(0, -shift[i_X]).reshape(1, -1), indw)

        x_warp[ind_c, indw.astype(int)] = x[ind_c, ind.astype(int)]

    shift = numpy.reshape(shift, (len(shift), 1))

    return x_warp, shift, values


def remove_nan(b, signal, select=numpy.any, flags=False):
    '''
    Rearrange segments so that they do not include nan's

    [Bn] = remove_nan(b,  signal,  select)
    [an, flag]
    INPUT
    b     : (p * 2) Boundary matrix (i.e. [Seg_start(1) Seg_end(1); Seg_start(2) Seg_end(2);...]
    signal: (n * 2) Matrix of signals (with signals on rows)
    select: (1 * 1) function handle to selecting operator
                     e.g. numpy.any (default) eliminate a column from signal matrix
                                         if one or more elements are missing
                          numpy.all           eliminate a column from signal matrix
                                         if all elements are missing

    OUTPUT
    Bn  : (q * 2)     new Boundary matrix in which nan's are removed
    flag: (q * 2 * n) flag matrix if there are nan before (column 1) or after (column 2)
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
    2.00.00 23 Mar 09 -> Added output for adjacent nan's in signals
    2.01.00 23 Mar 09 -> Added select input parameter
    '''
    c = nans(b.shape[0], b.shape[1] if len(b.shape) > 1 else 1)
    b = b.reshape(1, -1)
    count = 0
    signal = numpy.isnan(signal)
    for i_el in range(0, b.shape[0]):

        ind = numpy.arange(b[i_el, 0], b[i_el, 1] + 1)
        in_ = select(signal[:, ind], axis=0)

        if numpy.any(in_):

            p = numpy.diff(numpy.array([0] + in_).reshape(1, -1), 1, axis=1)
            a = numpy.flatnonzero(p < 0) + 1
            b = numpy.flatnonzero(p > 0)

            if numpy.any(~in_[0]):
                a = cat(1, numpy.array([0]), a)

            else:
                b = b[1:]

            if numpy.any(~in_[-1]):
                b = cat(1, b, numpy.array([max(ind.shape) - 1]))

            a = numpy.unique(a)
            b = numpy.unique(b)

            d = ind[cat(0, a, b)]

            c.resize(d.shape)
            c[count:count + max(a.shape) + 1] = d

            count = count + max(a.shape)

        else:
            c[count, :] = b[i_el,:]
            count += 1

    c = c.astype(int).T
    an = c

    if flags:
        flag = numpy.empty((c.shape[0], 2, signal.shape[0]), dtype=bool)

        flag[:] = False

        c_inds = c[:, 0] > 1
        c_inds = c_inds.astype(bool)

        c_inde = c[:, 1] < signal.shape[1]
        c_inde = c_inde.astype(bool)
        flag[c_inds, 0, :] = signal[:, c[c_inds, 0] - 1].T

        flag[c_inde, 1, :] = signal[:, c[c_inde, 0] - 1].T
        return an, flag
    else:
        return an


def normalise(x, flag=False):
    '''
    Column-wise normalise matrix
    nan's are ignored

    [xn] = normalise(x,  flag)

    INPUT
    x   : Marix
    flag: true if any NaNs are present (optional - it saves time for large matrices)

    OUTPUT
    xn: Column-wise normalised matrix

    Author: Giorgio Tomasi
             Giorgio.Tomasi@gmail.com

    Created      : 09 March,  2009; 13:18
    Last modified: 09 March,  2009; 13:50

    Python implementation: Martin Fitzpatrick
                           martin.fitzpatrick@gmail.com

    Last modified: 28th October,  2013
    '''

    if not flag:
        p_att = ~numpy.isnan(x)
        flag = numpy.any(~p_att[:])

    else:
        p_att = ~numpy.isnan(x)

    m, n = x.shape
    xn = nans(m, n)
    if flag:
        for i_n in range(0, n):
            n = numpy.linalg.norm(x[p_att[:, i_n], i_n])
            if not n:
                n = 1
            xn[p_att[:, i_n], i_n] = x[p_att[:, i_n], i_n] / n

    else:
        for i_n in range(0, n):
            n = numpy.linalg.norm(x[:, i_n])
            if not n:
                n = 1
            xn[:, i_n] = x[:, i_n] / n

    return xn


def extract_segments(x, segments):
    '''
    Extract segments from signals

    [xseg] = extract_segments(x,  segments)
    ? [xseg, segnew] = extract_segments(x,  segments)

    INPUT
    x       : (n * p) data matrix
    segments: (s * 2) segment boundary matrix

    OUTPUT
    xseg: (n * q) data matrix in which segments have been removed
    segnew: New segment layout

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
    n, p = x.shape
    Sd = numpy.diff(segments, axis=1)

    q = numpy.sum(Sd + 1)
    s, t = segments.shape

    flag_si = t != 2
    flag_in = numpy.any(segments[:] != numpy.fix(segments[:]))
    flag_ob = numpy.any(segments[:, 0] < 1) or numpy.any(segments[:, 1] > p)
    flag_ni = numpy.any(numpy.diff(segments[:, 0]) < 0) or numpy.any(
        numpy.diff(segments[:, 1]) < 0)
    flag_ab = numpy.any(Sd < 2)

    if flag_si:
        raise(Exception, 'Segment boundary matrix must have two columns')

    if flag_in:
        raise(Exception, 'Segment boundaries must be integers')

    if flag_ob:
        raise(Exception, 'Segment boundaries outside of segment')

    if flag_ni:
        raise(Exception, 'segments boundaries must be monotonically increasing')

    if flag_ab:
        raise(Exception, 'segments must be at least two points long')

    xseg = nans(n, q)
    origin = 0
    segnew = []

    for seg in segments:
        data = x[:, seg[0]:seg[1] + 1]
        segment_size = data.shape[1]
        xseg[:, origin:origin + segment_size] = data

        segnew.append([origin, origin + segment_size - 1])
        origin = origin + segment_size

    segnew = numpy.array(segnew)

    return xseg, segnew


def find_nearest(array, value):
    idx = (numpy.abs(array-value)).argmin()
    return array[idx], idx

def scal2pts(ppmi,  ppm=[],  prec=None):
    """
    Transforms scalars into data points

    pts = scal2pts(values, scal)

    INPUT
    values: scalars whose position is sought
    scal  : vector scalars
    prec  : precision (optional) to handle endpoints

    OUTPUT
    pts   : position of the requested scalars (nan if it is outside of 'scal')

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
    rev = ppm[0] > ppm[1]

    if prec is None:
        prec = min(abs(unique(numpy.diff(ppm))))

    pts = []
    for i in ppmi:
        nearest_v, idx = find_nearest(ppm, i)
        if abs(nearest_v-i) > prec:
            pts.append(numpy.nan)
        else:
            pts.append( idx )

    return numpy.array(pts)



def dscal2dpts(d, ppm, prec=None):
    """
    Translates an interval width from scal to the best approximation in sampling points.

    i = dppm2dpts(delta, scal, prec)

    INPUT
    delta: interval width in scale units
    scal : scale
    prec : precision on the scal axes

    OUTPUT
    i: interval widths in sampling points

    Author: Giorgio Tomasi
            Giorgio.Tomasi@gmail.com

    Last modified: 21st February,  2009

    Python implementation: Martin Fitzpatrick
                           martin.fitzpatrick@gmail.com

    Last modified: 28th October,  2013
    """
    if d == 0:
        return 0

    if d <= 0:
        raise(Exception, 'delta must be positive')

    if ppm[0] < ppm[1]: # Scale in order
        i = scal2pts(numpy.array([ppm[0] + d]), ppm, prec) -1

    else:
        i = max(ppm.shape) - scal2pts(numpy.array([ppm[-1] + d]), ppm, prec) +1

    return i[0]


