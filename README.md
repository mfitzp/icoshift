icoshift
========

A versatile tool for the rapid alignment of 1D NMR spectra

This package is a Python implementation of the [*i*coshift](http://www.ncbi.nlm.nih.gov/pubmed/20004603) algorithm as described by [Francesco Savorani](www.models.life.ku.dk) and [Giorgio Tomasi](www.igm.life.ku.dk). It uses correlation shifting of spectral intervals and employs an FFT engine that aligns all spectra simultaneously.

The Matlab algorithm is demonstrated to be faster than similar methods found in the literature making full-resolution alignment of large datasets feasible and thus avoiding down-sampling steps such as binning. The algorithm uses missing values as a filling alternative in order to avoid spectral artifacts at the segment boundaries.

It has been converted to Python using [SMOP](http://chiselapp.com/user/victorlei/repository/smop-dev/home) followed by hand re-coding using test datasets to check output at various steps. Better (and more complicated) test cases to come.

The interface remains identical to the Matlab version at present.

# Here Be Dragons

Conversion from one programming language to another is not straightforward. Particularly problematic here was the different indexing - zero-index vs. one-indexed arrays - in Python vs. Matlab. Simple to fix in situ, but less so when downstream code depends on it, after various matrix transformations.

At present this algorithm handles the basic *default* settings and no more. Contributions, bugfixes and - most importantly - Pythonification of the code is most welcome. It is duck ugly as it stands.

But it works.

# Thanks

Thanks to [Francesco Savorani](www.models.life.ku.dk) and [Giorgio Tomasi](www.igm.life.ku.dk) for the original neat and well documented [algorithm](http://www.ncbi.nlm.nih.gov/pubmed/20004603).