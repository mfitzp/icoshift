icoshift
========

A versatile tool for the rapid alignment of 1D NMR spectra

This package is a Python implementation of the [*i*coshift](http://www.ncbi.nlm.nih.gov/pubmed/20004603) algorithm as
described by [Francesco Savorani](www.models.life.ku.dk) and [Giorgio Tomasi](www.igm.life.ku.dk). It uses correlation
shifting of spectral intervals and employs an FFT engine that aligns all spectra simultaneously.

The Matlab algorithm is demonstrated to be faster than similar methods found in the literature making full-resolution
alignment of large datasets feasible and thus avoiding down-sampling steps such as binning. The algorithm uses missing
values as a filling alternative in order to avoid spectral artifacts at the segment boundaries.

It has been converted to Python using [SMOP](http://chiselapp.com/user/victorlei/repository/smop-dev/home) followed by
hand re-coding using test datasets to check output at various steps. Better (and more complicated) test cases to come.

The interface remains identical to the Matlab version at present.

# Here Be Dragons

Conversion from one programming language to another is not straightforward. Particularly problematic from MATLAB to
Python is the change from zero-based to one-based indexing. The implementation has been fixed to work and produce
*comparable* output for all inputs, however issues with some datasets or settings may remain. Full tests to confirm
equivalence to the MATLAB algorithm to follow.

But it works.

# Thanks

Thanks to [Francesco Savorani](www.models.life.ku.dk) and [Giorgio Tomasi](www.igm.life.ku.dk) for the original neat and well documented [algorithm](http://www.ncbi.nlm.nih.gov/pubmed/20004603).