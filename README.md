# patchmatch
Python implementation of a customized version of generalized Patchmatch
(see the paper: Barnes C., Shechtman E., Goldman D.B., Finkelstein A. (2010) The Generalized PatchMatch Correspondence Algorithm.)
Our version is "generalized" only in the sense it looks for approximated k nearest neighbours of a given patch, instead of only looking for *the* approximated nearest neighbour (like in Barnes, C., Shechtman, E., Finkelstein, A., & Goldman, D. (2009). PatchMatch: A randomized correspondence algorithm for structural image editing. ACM Transactions on Graphics-TOG, 28(3), 24.).

The implementation of the algorithm is in file patchmatch_code.py, and a test script is provided by test.py.

The function to call in order to run the algorithm is patch_match(im, m, knn), where im is a grayscale image, m the "half" size of a patch (then a patch is 2*m+1 x 2*m+1), and knn is the number of nearest neighbours to look for.

The function patch_match returns a set of offsets and a set of weights.
Both are n x knn matrices, where n is the number of pixels of the input image and knn is the number of nearest neighbours to look for.
Each row corresponds to one pixel.

In offsets, column j gives the offset for the j-th nearest neighbour. For example, for the patch around pixel i, the 3rd nearest neighbour is the patch around pixel i+offsets[i, 2].

In weights, column j gives a weight that is linked to the distance between the patch around pixel i and the patch around pixel i+offsets[i, j-1]. In this implementation, weights are negative, defined as w_ij = - alpha * d(patch_i, patch_j) where d(., .) is a Gaussian-weighted Euclidean distance, computed by the function patch_dissimilarity, and alpha is a positive parameter set to 0.01 here.

Therefore one can easily modify the code to return the distances between patches instead of weights - just devide the weights by -alpha.
However, note that the algorithm assumes that patch_dissimilarity returns a negative quantity, that is closer to zero when two patches are similar. The algorithm should still work as long as the returned quantity is higher for similar patches than for dissimilar ones, negativity is not necessary, but this order must be preserved.

In the words of the original paper Barnes, C., Shechtman, E., Finkelstein, A., & Goldman, D. (2009). PatchMatch: A randomized correspondence algorithm for structural image editing. ACM Transactions on Graphics-TOG, 28(3), 24,
this implementation performs one propagation followed by a random search and another propagation.

