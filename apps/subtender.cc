/*
   Copyright (c) 2019, Triad National Security, LLC. All rights reserved.

   This is open source software; you can redistribute it and/or modify it
   under the terms of the BSD-3 License. If software is modified to produce
   derivative works, such modified software should be clearly marked, so as
   not to confuse it with the version available from LANL. Full text of the
   BSD-3 License can be found in the LICENSE file of the repository.
*/

/**
  \file subtender.cc

  \brief Inventory subroutines and subroutine calls and suggest potentially
  external code.

  Demonstration program to inventory subroutine names and calls and match
  them against common external libraries from which source code may have been
  imported. This assists with managing legacy code by detecting code that may
  be externally maintained and with intellectual property protection
 */

#include "subtender.hh"
#include "flpr/flpr.hh"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <iostream>
#include <map>
#include <set>
#include <unistd.h>
#include <utility>
#include <vector>

/*--------------------------------------------------------------------------*/

#define TAG(T) FLPR::Syntax_Tags::T

using File = FLPR::Parsed_File<>;
using Cursor = typename File::Parse_Tree::cursor_t;
using Procedure = FLPR::Procedure<File>;

// using vec_str = std::vector<std::string>;
using set_str = std::set<std::string>;
using pair_ss = std::pair<std::string, std::string>;
// using vec_ssp = std::vector<std::pair<std::string, std::string>>;
using vec_ssp = std::vector<pair_ss>;
using map_ss = std::map<std::string, std::string>;
/*--------------------------------------------------------------------------*/

SubScanConfigurator sscfg;

// sublib structures contain lower-case subroutine name followed by a brief
// description 

/// List of F77 subroutines from Numerical Recipes, 1st Ed. (F77)
map_ss sublib_NR1 {};

/// List of Fortran (F77? F90?) subroutines from Numerical Recipes, 2nd Ed.
map_ss sublib_NR2 {
  {"addint", "Interpolate and add, used by mglin [19.6]"},
  {"airy", "Airy functions [6.7]"},
  {"amebsa", "Simulated annealing in continuous spaces [10.9]"},
  {"amoeba", "Minimize in N-dimensions by downhill simplex method [10.4]"},
  {"amotry", "Evaluate a trial point, used by amoeba [10.4]"},
  {"amotsa", "Evaluate a trial point, used by amebsa [10.9]"},
  {"anneal", "Traveling salesman problem by simulated annealing [10.9]"},
  {"anorm2", "Utility used by mgfas [19.6]"},
  {"arcmak", "Construct an arithmetic code [20.5]"},
  {"arcode", "Encode or decode a character using arithmetic coding [20.5]"},
  {"arcsum", "Add integer to byte string, used by arcode [20.5]"},
  {"asolve", "Used by linbcg for preconditioner [2.7]"},
  {"atimes", "Used by linbcg for sparse multiplication [2.7]"},
  {"avevar", "Calculate mean and variance of a data set [14.2]"},
  {"badluk", "Friday the 13th when the moon is full [1.1]"},
  {"balanc", "Balance a nonsymmetric matrix [11.5]"},
  {"banbks", "Band diagonal systems, backsubstitution [2.4]"},
  {"bandec", "Band diagonal systems, decomposition [2.4]"},
  {"banmul", "Multiply vector by band diagonal matrix [2.4]"},
  {"bcucof", "Construct two-dimensional bicubic [3.6]"},
  {"bcuint", "Two-dimensional bicubic interpolation [3.6]"},
  {"beschb", "Chebyshev expansion used by bessjy [6.7]"},
  {"bessi0", "Modified Bessel function I_0 [6.6]"},
  {"bessi1", "Modified Bessel function I_1 [6.6]"},
  {"bessi", "Modified Bessel function I of integer order [6.6]"},
  {"bessik", "Modified Bessel functions of fractional order [6.7]"},
  {"bessj0", "Bessel function J_0 [6.5]"},
  {"bessj1", "Bessel function J_1 [6.5]"},
  {"bessj", "Bessel function J of general integer order [6.5]"},
  {"bessjy", "Bessel functions of fractional order [6.7]"},
  {"bessk0", "Modified Bessel function K_0 [6.6]"},
  {"bessk1", "Modified Bessel function K_1 [6.6]"},
  {"bessk", "Modified Bessel function K of integer order [6.6]"},
  {"bessy0", "Bessel function Y_0 [6.5]"},
  {"bessy1", "Bessel function Y_1 [6.5]"},
  {"bessy", "Bessel function Y of general integer order [6.5]"},
  {"beta", "Beta function [6.1]"},
  {"betacf", "Continued fraction used by betai [6.4]"},
  {"betai", "Incomplete beta function [6.4]"},
  {"bico", "Binomial coefficients function [6.1]"},
  {"bksub", "Backsubstitution, used by SOLVDE [17.3]"},
  {"bnldev", "Binomial distributed random deviates [7.3]"},
  {"brent", "Find minimum of a function by Brent's method [10.2]"},
  {"broydn", "Secant method for systems of equations [9.7]"},
  {"bsstep", "Integrate ODEs, Bulirsch-Stoer step [16.4]"},
  {"caldat", "Calendar date from Julian day number [1.1]"},
  {"chder", "Derivative of a function already Chebyshev fitted [5.9]"},
  {"chebev", "Chebyshev polynomial evaluation [5.8]"},
  {"chebft", "Fit a Chebyshev polynomial to a function [5.8]"},
  {"chebpc", "Polynomial coefficients from a Chebyshev fit [5.10]"},
  {"chint", "Integrate a function already Chebyshev fitted [5.9]"},
  {"chixy", "Used by fitexy to calculate a chi-squared [15.3]"},
  {"choldc", "Cholesky decomposition [2.9]"},
  {"cholsl", "Cholesky backsubstitution [2.9]"},
  {"chsone", "Chi-square test for difference between data and model [14.3]"},
  {"chstwo", "Chi-square test for difference between two data sets [14.3]"},
  {"cisi", "Cosine and sine integrals Ci and Si [6.9]"},
  {"cntab1", "Contingency table analysis using chi-square [14.4]"},
  {"cntab2", "Contingency table analysis using entropy measure [14.4]"},
  {"convlv", "Convolution or deconvolution of data using FFT [13.1]"},
  {"copy", "Utility used by mglin, mgfas [19.6]"},
  {"correl", "Correlation or autocorrelation of data using FFT [13.2]"},
  {"cosft1", "Fast cosine transform with endpoints [12.3]"},
  {"cosft2", "``staggered'' fast cosine transform [12.3]"},
  {"covsrt", "Rearrange covariance matrix, used by lfit [15.4]"},
  {"crank", "Replaces array elements by their rank [14.6]"},
  {"cyclic", "Solution of cyclic tridiagonal systems [2.7]"},
  {"daub4", "Daubechies 4-coefficient wavelet filter [13.10]"},
  {"dawson", "Dawson's integral [6.10]"},
  {"dbrent", "Find minimum of a function using derivative information [10.3]"},
  {"ddpoly", "Evaluate a polynomial and its derivatives [5.3]"},
  {"decchk", "Decimal check digit calculation or verification [20.3]"},
  {"derivs", "Sample derivatives routine for stiff [16.6]"},
  {"df1dim", "Alternative function used by linmin [10.6]"},
  {"dfpmin", "Minimize in N-dimensions by variable metric method [10.7]"},
  {"dfridr", "Numerical derivative by Ridders' method [5.7]"},
  {"dftcor", "Compute endpoint corrections for Fourier integrals [13.9]"},
  {"dftint", "High-accuracy Fourier integrals [13.9]"},
  {"difeq", "Spheroidal matrix coefficients, used by SFROID [17.4]"},
  {"eclass", "Determine equivalence classes from list [8.6]"},
  {"eclazz", "Determine equivalence classes from procedure [8.6]"},
  {"ei", "Exponential integral Ei [6.3]"},
  {"eigsrt", "Eigenvectors, sorts into order by eigenvalue [11.1]"},
  {"elle", "Legendre elliptic integral of the second kind [6.11]"},
  {"ellf", "Legendre elliptic integral of the first kind [6.11]"},
  {"ellpi", "Legendre elliptic integral of the third kind [6.11]"},
  {"elmhes", "Reduce a general matrix to Hessenberg form [11.5]"},
  {"erf", "Error function [6.2]"},
  {"erfc", "Complementary error function [6.2]"},
  {"erfcc", "Complementary error function, concise routine [6.2]"},
  {"eulsum", "Sum a series by Euler--van Wijngaarden algorithm [5.1]"},
  {"evlmem", "Power spectral estimation from MEM coefficients [13.7]"},
  {"expdev", "Exponential random deviates [7.2]"},
  {"expint", "Exponential integral E_n [6.3]"},
  {"f1dim", "Function used by linmin [10.5]"},
  {"factln", "Logarithm of factorial function [6.1]"},
  {"factrl", "Factorial function [6.1]"},
  {"fasper", "Power spectrum of unevenly sampled larger data sets [13.8]"},
  {"fdjac", "Finite-difference Jacobian, used by newt [9.7]"},
  {"fgauss", "Fit a sum of Gaussians using mrqmin [15.5]"},
  {"fill0", "Utility used by mglin [19.6]"},
  {"fit", "Least-squares fit data to a straight line [15.2]"},
  {"fitexy", "Fit data to a straight line, errors in both x and y [15.3]"},
  {"fixrts", "Reflect roots of a polynomial into unit circle [13.6]"},
  {"fleg", "Fit a Legendre polynomial using lfit or svdfit [15.4]"},
  {"flmoon", "Calculate phases of the moon by date [1.0]"},
  {"fmin", "Norm of a vector function, used by newt [9.7]"},
  {"four1", "Fast Fourier transform (FFT) in one dimension [12.2]"},
  {"fourew", "Rewind and permute files, used by fourfs [12.6]"},
  {"fourfs", "FFT for huge data sets on external media [12.6]"},
  {"fourn", "Fast Fourier transform in multidimensions [12.4]"},
  {"fpoly", "Fit a polynomial using lfit or svdfit [15.4]"},
  {"fred2", "Solve linear Fredholm equations of the second kind [18.1]"},
  {"fredex", "Example of solving a singular Fredholm equation [18.3]"},
  {"fredin", "Interpolate solutions obtained with fred2 [18.1]"},
  {"frenel", "Fresnel integrals S(x) and C(x) [6.9]"},
  {"frprmn", "Minimize in N-dimensions by conjugate gradient [10.6]"},
  {"ftest", "F-test for difference of variances [14.2]"},
  {"gamdev", "Gamma-law distribution random deviates [7.3]"},
  {"gammln", "Logarithm of gamma function [6.1]"},
  {"gammp", "Incomplete gamma function [6.2]"},
  {"gammq", "Complement of incomplete gamma function [6.2]"},
  {"gasdev", "Normally distributed random deviates [7.2]"},
  {"gaucof", "Quadrature weights from orthogonal polynomials [4.5]"},
  {"gauher", "Gauss-Hermite weights and abscissas [4.5]"},
  {"gaujac", "Gauss-Jacobi weights and abscissas [4.5]"},
  {"gaulag", "Gauss-Laguerre weights and abscissas [4.5]"},
  {"gauleg", "Gauss-Legendre weights and abscissas [4.5]"},
  {"gaussj", "Gauss-Jordan matrix inversion and linear equation solution [2.1]"},
  {"gcf", "Continued fraction used by gammp and gammq [6.2]"},
  {"golden", "Find minimum of a function by golden section search [10.1]"},
  {"gser", "Series used by gammp and gammq [6.2]"},
  {"hpsel", "Find M largest values, without altering an array [8.5]"},
  {"hpsort", "Sort an array by heapsort method [8.3]"},
  {"hqr", "Eigenvalues of a Hessenberg matrix [11.6]"},
  {"hufapp", "Append bits to a Huffman code, used by hufmak [20.4]"},
  {"hufdec", "Use Huffman code to decode and decompress a character [20.4]"},
  {"hufenc", "Use Huffman code to encode and compress a character [20.4]"},
  {"hufmak", "Construct a Huffman code [20.4]"},
  {"hunt", "Search a table when calls are correlated [3.4]"},
  {"hypdrv", "Complex hypergeometric function, derivative of [6.12]"},
  {"hypgeo", "Complex hypergeometric function [6.12]"},
  {"hypser", "Complex hypergeometric function, series evaluation [6.12]"},
  {"icrc1", "Cyclic redundancy checksum, used by icrc [20.3]"},
  {"icrc", "Cyclic redundancy checksum [20.3]"},
  {"igray", "Gray code and its inverse [20.2]"},
  {"indexx", "Construct an index for an array [8.4]"},
  {"interp", "Bilinear prolongation, used by mglin, mgfas [19.6]"},
  {"irbit1", "Random bit sequence [7.4]"},
  {"irbit2", "Random bit sequence [7.4]"},
  {"jacobi", "Eigenvalues and eigenvectors of a symmetric matrix [11.1]"},
  {"jacobn", "Sample Jacobian routine for stiff [16.6]"},
  {"julday", "Julian Day number from calendar date [1.1]"},
  {"kendl1", "Correlation between two data sets, Kendall's tau [14.6]"},
  {"kendl2", "Contingency table analysis using Kendall's tau [14.6]"},
  {"kermom", "Sample routine for moments of a singular kernel [18.3]"},
  {"ks2d1s", "K--S test in two dimensions, data vs. model [14.7]"},
  {"ks2d2s", "K--S test in two dimensions, data vs. data [14.7]"},
  {"ksone", "Kolmogorov-Smirnov test of data against model [14.3]"},
  {"kstwo", "Kolmogorov-Smirnov test between two data sets [14.3]"},
  {"laguer", "Find a root of a polynomial by Laguerre's method [9.5]"},
  {"lfit", "General linear least-squares fit by normal equations [15.4]"},
  {"linbcg", "Biconjugate gradient solution of sparse systems [2.7]"},
  {"linmin", "Minimum of a function along a ray in N-dimensions [10.5]"},
  {"lnsrch", "Search along a line, used by newt [9.7]"},
  {"locate", "Search an ordered table by bisection [3.4]"},
  {"lop", "Applies nonlinear operator, used by mgfas [19.6]"},
  {"lubksb", "Linear equation solution, backsubstitution [2.3]"},
  {"ludcmp", "Linear equation solution, LU decomposition [2.3]"},
  {"machar", "Diagnose computer's floating arithmetic [20.1]"},
  {"maloc", "Memory allocation utility used by mglin, mgfas [19.6]"},
  {"matadd", "Utility used by mgfas [19.6]"},
  {"matsub", "Utility used by mgfas [19.6]"},
  {"medfit", "Fit data to a straight line robustly, least absolute deviation [15.7]"},
  {"memcof", "Evaluate maximum entropy (MEM) coefficients [13.6]"},
  {"metrop", "Metropolis algorithm, used by anneal [10.9]"},
  {"mgfas", "Nonlinear elliptic PDE solved by multigrid method [19.6]"},
  {"mglin", "Linear elliptic PDE solved by multigrid method [19.6]"},
  {"midexp", "Integrate a function that decreases exponentially [4.4]"},
  {"midinf", "Integrate a function on a semi-infinite interval [4.4]"},
  {"midpnt", "Extended midpoint rule [4.4]"},
  {"midsql", "Integrate a function with lower square-root singularity [4.4]"},
  {"midsqu", "Integrate a function with upper square-root singularity [4.4]"},
  {"miser", "Recursive multidimensional Monte Carlo integration [7.8]"},
  {"mmid", "Integrate ODEs by modified midpoint method [16.3]"},
  {"mnbrak", "Bracket the minimum of a function [10.1]"},
  {"mnewt", "Newton's method for systems of equations [9.6]"},
  {"moment", "Calculate moments of a data set [14.1]"},
  {"mp2dfr", "Multiple precision conversion to decimal base [20.6]"},
  {"mpdiv", "Multiple precision divide and remainder [20.6]"},
  {"mpinv", "Multiple precision reciprocal [20.6]"},
  {"mpmul", "Multiple precision multiply, using FFT methods [20.6]"},
  {"mpops", "Multiple precision arithmetic, simpler operations [20.6]"},
  {"mppi", "Multiple precision example, compute many digits of pi [20.6]"},
  {"mprove", "Linear equation solution, iterative improvement [2.5]"},
  {"mpsqrt", "Multiple precision square root [20.6]"},
  {"mrqcof", "Used by mrqmin to evaluate coefficients [15.5]"},
  {"mrqmin", "Nonlinear least-squares fit, Marquardt's method [15.5]"},
  {"newt", "Globally convergent multi-dimensional Newton's method [9.7]"},
  {"odeint", "Integrate ODEs with accuracy monitoring [16.2]"},
  {"orthog", "Construct nonclassical orthogonal polynomials [4.5]"},
  {"pade", "Pad'e approximant from power series coefficients [5.12]"},
  {"pccheb", "Inverse of chebpc; use to economize power series [5.11]"},
  {"pcshft", "Polynomial coefficients of a shifted polynomial [5.10]"},
  {"pearsn", "Pearson's correlation between two data sets [14.5]"},
  {"period", "Power spectrum of unevenly sampled data [13.8]"},
  {"piksr2", "Sort two arrays by straight insertion [8.1]"},
  {"piksrt", "Sort an array by straight insertion [8.1]"},
  {"pinvs", "Diagonalize a sub-block, used by SOLVDE [17.3]"},
  {"plgndr", "Legendre polynomials, associated (spherical harmonics) [6.8]"},
  {"poidev", "Poisson distributed random deviates [7.3]"},
  {"polcoe", "Polynomial coefficients from table of values [3.5]"},
  {"polcof", "Polynomial coefficients from table of values [3.5]"},
  {"poldiv", "Divide one polynomial by another [5.3]"},
  {"polin2", "Two-dimensional polynomial interpolation [3.6]"},
  {"polint", "Polynomial interpolation [3.1]"},
  {"powell", "Minimize in N-dimensions by  Powell's method [10.5]"},
  {"predic", "Linear prediction using MEM coefficients [13.6]"},
  {"probks", "Kolmogorov-Smirnov probability function [14.3]"},
  {"psdes", "``pseudo-DES'' hashing of 64 bits [7.5]"},
  {"pwt", "Partial wavelet transform [13.10]"},
  {"pwtset", "Initialize coefficients for pwt [13.10]"},
  {"pythag", "Calculate (a^2+b^2)^{1/2} without overflow [2.6]"},
  {"pzextr", "Polynomial extrapolation, used by bsstep [16.4]"},
  {"qgaus", "Integrate a function by Gaussian quadratures [4.5]"},
  {"qrdcmp", "QR decomposition [2.10]"},
  {"qromb", "Integrate using Romberg adaptive method [4.3]"},
  {"qromo", "Integrate using open Romberg adaptive method [4.4]"},
  {"qroot", "Complex or double root of a polynomial, Bairstow [9.5]"},
  {"qrsolv", "QR backsubstitution [2.10]"},
  {"qrupdt", "Update a QR decomposition [2.10]"},
  {"qsimp", "Integrate using Simpson's rule [4.2]"},
  {"qtrap", "Integrate using trapezoidal rule [4.2]"},
  {"quad3d", "Integrate a function over a three-dimensional space [4.6]"},
  {"quadct", "Count points by quadrants, used by ks2d1s [14.7]"},
  {"quadmx", "Sample routine for a quadrature matrix [18.3]"},
  {"quadvl", "Quadrant probabilities, used by ks2d1s [14.7]"},
  {"ran0", "Random deviate by Park and Miller minimal standard [7.1]"},
  {"ran1", "Random deviate, minimal standard plus shuffle [7.1]"},
  {"ran2", "Random deviate by L'Ecuyer long period plus shuffle [7.1]"},
  {"ran3", "Random deviate by Knuth subtractive method [7.1]"},
  {"ran4", "Random deviates from DES-like hashing [7.5]"},
  {"rank", "Construct a rank table for an array [8.4]"},
  {"ranpt", "Get random point, used by miser [7.8]"},
  {"ratint", "Rational function interpolation [3.2]"},
  {"ratlsq", "Rational fit by least-squares method [5.13]"},
  {"ratval", "Evaluate a rational function [5.3]"},
  {"rc", "Carlson's degenerate elliptic integral [6.11]"},
  {"rd", "Carlson's elliptic integral of the second kind [6.11]"},
  {"realft", "Fast Fourier transform of a single real function [12.3]"},
  {"rebin", "Sample rebinning used by vegas [7.8]"},
  {"red", "Reduce columns of a matrix, used by SOLVDE [17.3]"},
  {"relax2", "Gauss-Seidel relaxation, used by mgfas [19.6]"},
  {"relax", "Gauss-Seidel relaxation, used by mglin [19.6]"},
  {"resid", "Calculate residual, used by mglin [19.6]"},
  {"revcst", "Cost of a reversal, used by anneal [10.9]"},
  {"revers", "Do a reversal, used by anneal [10.9]"},
  {"rf", "Carlson's elliptic integral of the first kind [6.11]"},
  {"rj", "Carlson's elliptic integral of the third kind [6.11]"},
  {"rk4", "Integrate one step of ODEs, fourth-order Runge-Kutta [16.1]"},
  {"rkck", "Cash-Karp-Runge-Kutta step used by rkqs [16.2]"},
  {"rkdumb", "Integrate ODEs by fourth-order Runge-Kutta [16.1]"},
  {"rkqs", "Integrate one step of ODEs with accuracy monitoring [16.2]"},
  {"rlft3", "FFT of real data in two or three dimensions [12.5]"},
  {"rotate", "Jacobi rotation used by qrupdt [2.10]"},
  {"rsolv", "Right triangular backsubstitution [2.10]"},
  {"rstrct", "Half-weighting restriction, used by mglin, mgfas [19.6]"},
  {"rtbis", "Find root of a function by bisection [9.1]"},
  {"rtflsp", "Find root of a function by false-position [9.2]"},
  {"rtnewt", "Find root of a function by Newton-Raphson [9.4]"},
  {"rtsafe", "Find root of a function by Newton-Raphson and bisection [9.4]"},
  {"rtsec", "Find root of a function by secant method [9.2]"},
  {"rzextr", "Rational function extrapolation, used by bsstep [16.4]"},
  {"savgol", "Savitzky-Golay smoothing coefficients [14.8]"},
  {"scrsho", "Graph a function to search for roots [9.0]"},
  {"select", "Find the Nth largest in an array [8.5]"},
  {"selip", "Find the Nth largest, without altering an array [8.5]"},
  {"sfroid", "Spheroidal functions by method of SOLVDE [17.4]"},
  {"shell", "Sort an array by Shell's method [8.1]"},
  {"shoot", "Solve two point boundary value problem by shooting [17.1]"},
  {"shootf", "Ditto, by shooting to a fitting point [17.2]"},
  {"simp1", "Linear programming, used by simplx [10.8]"},
  {"simp2", "Linear programming, used by simplx [10.8]"},
  {"simp3", "Linear programming, used by simplx [10.8]"},
  {"simplx", "Linear programming maximization of a linear function [10.8]"},
  {"simpr", "Integrate stiff ODEs by semi-implicit midpoint rule [16.6]"},
  {"sinft", "Fast sine transform [12.3]"},
  {"slvsm2", "Solve on coarsest grid, used by mgfas [19.6]"},
  {"slvsml", "Solve on coarsest grid, used by mglin [19.6]"},
  {"sncndn", "Jacobian elliptic functions [6.11]"},
  {"snrm", "Used by linbcg for vector norm [2.7]"},
  {"sobseq", "Sobol's quasi-random sequence [7.7]"},
  {"solvde", "Two point boundary value problem, solve by relaxation [17.3]"},
  {"sor", "Elliptic PDE solved by successive overrelaxation method [19.5]"},
  {"sort2", "Sort two arrays by quicksort method [8.2]"},
  {"sort3", "Sort, use an index to sort 3 or more arrays [8.4]"},
  {"sort", "Sort an array by quicksort method [8.2]"},
  {"spctrm", "Power spectrum estimation using FFT [13.4]"},
  {"spear", "Spearman's rank correlation between two data sets [14.6]"},
  {"sphbes", "Spherical Bessel functions j_n and y_n [6.7]"},
  {"sphfpt", "Spheroidal functions by method of shootf [17.4]"},
  {"sphoot", "Spheroidal functions by method of shoot [17.4]"},
  {"splie2", "Construct two-dimensional spline [3.6]"},
  {"splin2", "Two-dimensional spline interpolation [3.6]"},
  {"spline", "Construct a cubic spline [3.3]"},
  {"splint", "Cubic spline interpolation [3.3]"},
  {"spread", "Extirpolate value into array, used by fasper [13.8]"},
  {"sprsax", "Product of sparse matrix and vector [2.7]"},
  {"sprsin", "Convert matrix to sparse format [2.7]"},
  {"sprspm", "Pattern multiply two sparse matrices [2.7]"},
  {"sprstm", "Threshold multiply two sparse matrices [2.7]"},
  {"sprstp", "Transpose of sparse matrix [2.7]"},
  {"sprstx", "Product of transpose sparse matrix and vector [2.7]"},
  {"stifbs", "Integrate stiff ODEs, Bulirsch-Stoer step [16.6]"},
  {"stiff", "Integrate stiff ODEs by fourth-order Rosenbrock [16.6]"},
  {"stoerm", "Integrate conservative second-order ODEs [16.5]"},
  {"svbksb", "Singular value backsubstitution [2.6]"},
  {"svdcmp", "Singular value decomposition of a matrix [2.6]"},
  {"svdfit", "Linear least-squares fit by singular value decomposition [15.4]"},
  {"svdvar", "Variances from singular value decomposition [15.4]"},
  {"toeplz", "Solve Toeplitz systems [2.8]"},
  {"tptest", "Student's t-test for means, case of paired data [14.2]"},
  {"tqli", "Eigensolution of a symmetric tridiagonal matrix [11.3]"},
  {"trapzd", "Trapezoidal rule [4.2]"},
  {"tred2", "Householder reduction of a real, symmetric matrix [11.2]"},
  {"tridag", "Solution of tridiagonal systems [2.4]"},
  {"trncst", "Cost of a transposition, used by anneal [10.9]"},
  {"trnspt", "Do a transposition, used by anneal [10.9]"},
  {"ttest", "Student's t-test for difference of means [14.2]"},
  {"tutest", "Student's t-test for means, case of unequal variances [14.2]"},
  {"twofft", "Fast Fourier transform of two real functions [12.3]"},
  {"vander", "Solve Vandermonde systems [2.8]"},
  {"vegas", "Adaptive multidimensional Monte Carlo integration [7.8]"},
  {"voltra", "Linear Volterra equations of the second kind [18.2]"},
  {"wt1", "One-dimensional discrete wavelet transform [13.10]"},
  {"wtn", "Multidimensional discrete wavelet transform [13.10]"},
  {"wwghts", "Quadrature weights for an arbitrarily singular kernel [18.3]"},
  {"zbrac", "Outward search for brackets on roots [9.1]"},
  {"zbrak", "Inward search for brackets on roots [9.1]"},
  {"zbrent", "Find root of a function by Brent's method [9.3]"},
  {"zrhqr", "Roots of a polynomial by eigenvalue methods [9.5]"},
  {"zriddr", "Find root of a function by Ridders' method [9.2]"},
  {"zroots", "Roots of a polynomial by Laguerre's method with deflation [9.5]"}
};

/// List of subroutines from IBM Scientific Software Package
map_ss sublib_IBM_SSP {};

/// List of subroutines from UNIVAC scienfic library
map_ss sublib_UNIVAC_sci {};

/// List of subroutines from UNIVAC statistical library
map_ss sublib_UNIVAC_stat {};

/// List of subroutines from BLAS
map_ss sublib_BLAS {};

/// List of subroutines from LINPACK
map_ss sublib_LINPACK {};

/// List of subroutines from EISPACK
map_ss sublib_EISPACK {};

/// List of subroutines from LAPACK
map_ss sublib_LAPACK {};

/// List of subroutines from ORNL Programmer's Handbook (K-1728)
map_ss sublib_ORNL_PH {
  // A.  Interpolation

  {"ylag", "Lagrangian interpolation for a function of one variable. p. 17, ORD 9050"},
  {"dlag", "Lagrangian interpolation for a function of two variables. p. 23, ORD 9051"},
  {"foura", "Fourier approximation. p. 29, ORD 9052"},
  {"fouras", "Fourier approximation. p. 29, ORD 9052"},
  {"fourac", "Fourier approximation. p. 29, ORD 9052"},

  // B.  Numerical Differentiation

  {"deriv1", "Derivative of least squares fit. p. 39, ORD 9053"},
  {"deriv2", "Derivative of least squares fit. p. 39, ORD 9053"},
  {"dxlagr", "Derivative of Lagrangian fit. p. 45, ORD 9054"},

  // C.  Numerical Integration

  {"intgr", "Gaussian integration. p. 53, ORD 9055"},
  {"simp", "Simpson integration. p. 57, ORD 9056"},
  {"simpun", "Simpson type integration for uneven spacing. p. 61, ORD 9057"},
  {"laguer", "Laguerre-Gauss integration. p. 67, ORD 9058"},
  {"hermit", "Hermite-Gauss integration. p. 71, ORD 9059"},

  // D.  Ordinary Differential Equations

  {"kutta", "Self-adjusting Runge-Kutta. p. 77, ORD 9060"},
  {"difeq", "Fixed interval Runge-Kutta. p. 85, ORD 9061"},
  {"dife2", "Runge-Kutta for second order systems. p. 89, ORD 9062"},

  // E.  Zeros of Functions

  {"abcisa", "Linear fractional statement function. p. 95"},
  {"cest", "Muller's iteration. p. 97, ORD 9063"},
  {"polsol", "Modified Lehmer's method for polynomial equations. p. 101, ORD 9064"},
  {"zeroin", "Secant method for systems of equations. p. 111, ORD 9066"},
  {"nonlin", "Secant method for systems of equations. p. 119, ORD 9065"},

  // F. Solution of Matrix Equations and Determinants

  {"dmateq", "Gaussian elimination for real systems. p. 131, ORD 9067"},
  {"cmateq", "Gaussian elimination for complex systems. p. 131, ORD 9067"},
  {"symin1", "Inversion of symmetric matrix. p. 139, ORD 9068"},
  {"symin2", "Inversion of symmetric matrix. p. 139, ORD 9068"},
  {"bansol", "Solution of band systems of linear equations. p. 147, ORD 9069"}, 
  {"trisol", "Solution of tridiagonal systems of linear equations. p. 153, ORD 9070"},
  {"cdet", "Determinant of complex matrix. p. 157, ORD 9071"},
  {"hdet", "Determinant of Hermitian matrix. p. 161, ORD 9072"},

  // G. Solution of Eigensystems

  {"matgt", "Generates test matrices. p. 169, ORD 9073"},
  {"matg1", "Generates test matrices. p. 169, ORD 9073"},
  {"symmat", "Symmetric matrix eigensystem solver. p. 173, ORD 9074"},
  {"bigmat", "Symmetric matrix eigensystem solver. p. 173, ORD 9074"},
  {"qrvec", "Symmetric matrix eigensystem solver. p. 187, ORD 9075"},
  {"tdag", "Symmetric matrix eigensystem solver. p. 195, ORD 9076"},
  {"jacobh", "Hermitian matrix eigensystem solver. p. 201, ORD 9077"},
  {"rilmat", "Real matrix eigensystem solver. p. 209, ORD 9078"},
  {"allmat", "Complex matrix eigensystem solver. p. 225, ORD 9079"},

  // H. Least Squares

  {"lsqfit", "Linear least squares. p. 239, ORD 9080"},
  {"alsq", "Linear least squares. p. 243, ORD 9081"},
  {"blsq", "Linear least squares. p. 243, ORD 9081"},

  // I. Statistics

  // Seven distributions of random numbers. p. 257, ORD 9082

  // Z. Miscellaneous

  {"axty", "Inner product of vectors. p. 267, ORD 9083"},
  {"dscale", "Scale vectors. p. 271, ORD 9084"},
  {"array", "Array listers. p. 277, ORD 9085"},
  {"arrayd", "Array listers. p. 277, ORD 9085"},
  {"arrayc", "Array listers. p. 277, ORD 9085"},
  {"araydc", "Array listers. p. 277, ORD 9085"},
  {"sort", "Sorting routine. p. 285, ORD 9086"},
  {"ngcd", "Greatest common divisors. p. 289, ORD 9087"},
  {"lcm", "Least common multiples. p. 289, ORD 9087"},
};

/// Default libraries and routine groups containing target routine names
std::map<std::string, map_ss> map_default_sublibs {
  // {"NR1",
  {"NR2", sublib_NR2},
  // {"IBM_SSP",     sublib_IBM_SSP},
  // {"UNIVAC_sci",  sublib_UNIVAC_sci},
  // {"UNIVAC_stat",  sublib_UNIVAC_stat},
  // {"BLAS",  sublib_BLAS},
  // {"LINPACK",  sublib_LINPACK},
  // {"EISPACK",  sublib_EISPACK},
  // {"LAPACK",  sublib_LAPACK},
  {"ORNL_PH",  sublib_ORNL_PH},
};

// Note: Derive set_default_sublibs from keys of map_default_sublibs
/// Default libraries and routine groups containing target routine names
set_str set_default_sublibs {
  // "NR1",
  "NR2",
  // "IBM_SSP",
  // "UNIVAC_sci",
  // "UNIVAC_stat",
  // "BLAS",
  // "LINPACK",
  // "EISPACK",
  // "LAPACK",
  "ORNL_PH",
};

/**
 *  \brief Local subroutine definition record
 */
struct LocalSub {

  /// Subroutine name
  std::string name {};

  /// Source file containing subroutine definition
  std::string filename {};

  /// Line number of subroutine definition
  int lineno {};

  /// Suspected origin of local subroutine
  std::string origin {};

  /// Constructor
  LocalSub(std::string SubroutineName,
           std::string FileName,
           int LineNumber,
           std::string OriginLibrary) :
           name(SubroutineName),   
           filename(FileName), 
           lineno(LineNumber),           
           origin(OriginLibrary)
  {
  }
}; // LocalSub

/// Directory of local subroutines and their suspected origins
std::vector<LocalSub> LocalSubDb {};

/*--------------------------------------------------------------------------*/

/**
 *  \brief Called subroutine definition record
 */
struct CalledSub {

  /// Subroutine name
  std::string name {};

  /// Source file containing subroutine call
  std::string filename {};

  /// Program unit containin the subroutine call
  std::string parent_routine {};

  /// Line number of subroutine call
  int lineno {};

  /// Suspected origin of called subroutine
  std::string origin {};

  /// Constructor
  CalledSub(std::string SubroutineName,
            std::string FileName,
            std::string ParentRoutine,
            int LineNumber,
            std::string OriginLibrary) :
            name(SubroutineName),   
            filename(FileName), 
            parent_routine(ParentRoutine),   
            lineno(LineNumber),           
            origin(OriginLibrary)
  {
  }
}; // CalledSub

/// Directory of called subroutines and their suspected origins
std::vector<CalledSub> CalledSubDb {};

/*--------------------------------------------------------------------------*/

/**
 *  \brief Return the lower-case value of a given character
 *  \param[in] in Character
 */
char chartolower(char in) {
  return std::tolower(in);
}

/*--------------------------------------------------------------------------*/

/**
 *  \brief Print code usage message to given output stream
 *  \param[inout] os Output stream
 */
void print_usage(std::ostream &os) {
  os << "Usage:\tsubtender [-n] [-x <subroutine>] [-L <library>] [-X <library>]"
        " <filename> ... \n";
  os << "\tsubtender -h\n";
  os << "\t-n\t\tSet dry-run mode; show actions that will occur but do not execute them\n";
  os << "\t-x\t\tExclude subroutine name from library search\n";
  os << "\t-L\t\tLimit libraries searched to this library\n";
  os << "\t-X\t\tExclude library from default libraries searched\n";
  os << "\t-h\t\tPrint usage message\n";
  os << "\t<subroutine>\tName of subroutine (case-insensitive)\n";
  os << "\t<library>\tName of library or file group\n";
  os << "\t<filename>\tFortran source file\n";

  // TODO: Show list of libraries available to search
}

/*--------------------------------------------------------------------------*/

/**
 *  \brief Print code usage message and default searchable libraries 
 *  to given output stream
 * 
 *  \param[inout] os Output stream
 *  \param[in] sscfg Configuration object
 */
void print_usage(std::ostream &os,
                 const SubScanConfigurator& sscfg) {
  // Show basic usage
  print_usage(os);

  // Show list of libraries available to search
  os << std::endl << "\tSubroutine libraries and groups include:" << std::endl;
  for (auto const& sublib: sscfg.default_libs) {
    os << "\t\t" << sublib << std::endl;
  }
}

/*--------------------------------------------------------------------------*/

/**
 *  \brief Parse command line options and arguments
 *  \param[in] argc Argument count
 *  \param[in] argv List of argument values
 *  \param[inout] sscfg Configuration object
 */
void parse_cmd_line(int argc,
                    char* const* argv,
                    SubScanConfigurator& sscfg)
{
  int ch;
  while ((ch = getopt(argc, argv, "hnx:L:X:")) != -1) {
    switch (ch) {
      case 'h':
      {
        // help: print usage and exit normally
        print_usage(std::cerr, sscfg);
        exit(0);
      }
      break;

      case 'n':
      {
        // dry-run: Set options and show intended operation but don't 
        // actually scan files
        sscfg.dry_run = true;
      }
      break;

      case 'x':
      {
        // Exclude subroutine
        std::string exsub = std::string(optarg);
        std::transform(exsub.begin(), exsub.end(), exsub.begin(), chartolower);

        sscfg.excluded_subs.insert(exsub);
        std::cerr << "Excluding subroutine " << exsub << std::endl;
      }
      break;

      case 'L':
      {
        // Include library
        // std::string inclib {optarg};
        std::string inclib = std::string(optarg);
        sscfg.included_libs.insert(inclib);
        std::cerr << "Including library " << inclib << std::endl;
      }
      break;

      case 'X':
      {
        // Exclude library
        // std::string exlib {optarg};
        std::string exlib = std::string(optarg);
        sscfg.excluded_libs.insert(exlib);
        std::cerr << "Excluding library " << exlib << std::endl;
      }
      break;

      default:
      {
        // unknown: print usage and exit with error
        print_usage(std::cerr, sscfg);
        exit(1);
      }
    }
  }

  // Remaining arguments are considered Fortran source files
  int idx = optind;
  std::cerr << "Found " << (argc - idx) << " files to check:" << std::endl;
  if (idx < argc) {
    for (int i = idx; i < argc; ++i) {
      std::string srcfile {argv[i]};
      sscfg.srcfiles.insert(srcfile);
//      std::cerr << "Scanning source file " << srcfile << std::endl;
      // fortran_filenames.emplace_back(std::string{argv[i]});
    }
  } else {
    // Print usage and exit with error if no files are specified
    print_usage(std::cerr, sscfg);
    exit(1);
  }

  return;
}

/*--------------------------------------------------------------------------*/

/**
 *  \brief Main subtender code driver
 *  \param[in] argc Argument count
 *  \param[in] argv List of argument values
 *  \returns Status code
 */
int main(int argc,
         char* const* argv) {
  int err {0};
  // SubScanConfigurator sscfg;
  sscfg.default_libs = set_default_sublibs;

  if (argc < 2) {
    print_usage(std::cerr, sscfg);
    err = 1;
  } else {

    parse_cmd_line(argc, argv, sscfg);

    if (sscfg.dry_run) {
      std::cerr << "Dry-run mode enabled." << std::endl;
    }

    if (sscfg.included_libs.size() > 0) {
      for (const auto& sublib : sscfg.included_libs) {
        if (sscfg.default_libs.count(sublib) > 0) {
          std::cerr << "Searching for routines in " << sublib << std::endl;
          sscfg.scan_libs.insert(sublib);
        } else {
          std::cerr << "Warning: Cannot find " << sublib << " in master library list." << std::endl;
        }
      }
    } else {
      for (const auto& sublib : sscfg.default_libs) {
        if (sscfg.excluded_libs.count(sublib) == 0) {
          std::cerr << "Searching for routines in " << sublib << std::endl;
          sscfg.scan_libs.insert(sublib);
        }
      }
    }

    if (sscfg.scan_libs.size() > 0) {
      std::cerr << "Searching for routines in:" << std::endl;
      for (const auto& sublib : sscfg.scan_libs) {
        std::cerr << "  " << sublib << std::endl;
      }

      for (const auto& ifn : sscfg.srcfiles) {
        std::cerr << "  Scanning " << ifn << std::endl;

        if (!sscfg.dry_run) {
          if (!subtender_file(ifn)) {
            err = 2;
            std::cerr << "Error scanning " << ifn << std::endl;
          }
        }
      }
    } else {
      err = 3;
      std::cerr << "Error: No valid libraries specified!" << std::endl;
    }
  }
  return err;
}

/*--------------------------------------------------------------------------*/

/**
 *  \brief Apply subtender transform to source file
 *  \param[in] filename Name of target file
 *  \returns Changed status; true if target file has been changed
 */
bool subtender_file(std::string const &filename) {
  std::cerr << "  sf: Scanning " << filename << std::endl;

  File file(filename);
  if (!file)
    return false;

  FLPR::Procedure_Visitor puv(file, subtender_procedure);
  bool const scanned = puv.visit();
  // if (changed) {
  //   write_file(std::cout, file);
  // }
  return scanned;
}

/*--------------------------------------------------------------------------*/

/**
 *  \brief Check for subroutine name in library
 *
 *  \param[in] lname Subroutine name, previously converted to lower case
 *  \param[in] sublib Name of library to search
 *  \returns True if subroutine `lname` was found in library `sublib`,
 *           false otherwise
 */
bool detect_subroutine(const std::string& lname,
                       const std::string& sublib)
{
  bool found_in_sublib {false};

  std::map<std::string, map_ss>::iterator it; 
  it = map_default_sublibs.find(sublib);

  bool found_sublib_in_master {it != map_default_sublibs.end()};

  if (found_sublib_in_master) {
    map_ss curlib;
    curlib = it->second;

    map_ss::iterator lib_it;
    lib_it = curlib.find(lname);

    found_in_sublib = (lib_it != curlib.end());
  }
  return found_in_sublib;
}

/*--------------------------------------------------------------------------*/

/**
 *  \brief Apply Subtender transform to a given procedure.
 *
 *  Subtender transform will be applied to procedures which
 *  can be `ingest()`ed and have an executable body and are not otherwise
 *  explicitly excluded from processing.
 *
 *  \param[inout] file Target source file object
 *  \param[in] c Cursor pointing to current procedure within `file`
 *  \param[in] internal_procedure Flag indicating the current procedure is an
 *             internal procedure, `contain`ed within another function or
 *             subroutine
 *  \param[in] module_procedure Flag indicating the current procedure is a
 *             module procedure, `contain`ed within a module
 *  \returns Changed status; true if target file has been changed
 */
bool subtender_procedure(File &file,
                         Cursor c,
                         bool internal_procedure,
                         bool module_procedure) {
  bool changed {true};
  Procedure proc(file);

  if (!proc.ingest(c)) {
    std::cerr << "\n******** Unable to ingest procedure *******\n"
              << std::endl;
    return false;
  }

//   if (internal_procedure) {
//     std::cerr << "skipping " << proc.name() << ": internal procedure"
//               << std::endl;
//     return false;
//   }

  if (!proc.has_region(Procedure::EXECUTION_PART)) {
    std::cerr << "skipping " << proc.name() << ": no execution part"
              << std::endl;
    return false;
  }

  if (exclude_procedure(proc)) {
    std::cerr << "skipping " << proc.name() << ": excluded" << std::endl;
    return false;
  }

  std::string lname {proc.name()};
  std::transform(lname.begin(), lname.end(), lname.begin(), chartolower);

  // for (auto const& sublib: sscfg.excluded_subs) {
  //   std::cerr << "Excluding: " << sublib << std::endl;
  // }

  // std::cerr << "Checking " << lname << " against exclusion list." << std::endl;
  if (sscfg.excluded_subs.count(lname) > 0) {
    std::cerr << "  sp: Skipping " << proc.name() << " as " << lname << std::endl;
  } else {
    std::cerr << "  sp: Scanning " << proc.name() << " as " << lname << std::endl;

    for (auto const& sublib: sscfg.scan_libs) {
      if (detect_subroutine(lname, sublib)) {
          // Hit -- Write to Db
          LocalSubDb.emplace_back(LocalSub(lname, "MysteryFile", 404, sublib));        
          std::cerr << "      sp: *** Found " << lname << " in library " << sublib << std::endl;
          std::cerr << "      sp: " << lname << " -> " << lib_it->second << std::endl;
      }
      // std::map<std::string, map_ss>::iterator it; 
      // it = map_default_sublibs.find(sublib);
      // bool found_sublib_in_master {it != map_default_sublibs.end()};
      // if (found_sublib_in_master) {
      //   std::cerr << "    sp: Looking for " << lname << " in library " << sublib << std::endl;
      //   map_ss::iterator lib_it;
      //   map_ss curlib;
      //   curlib = it->second;
      //   lib_it = curlib.find(lname);
      //   bool found_in_sublib {lib_it != curlib.end()};
      //   if (found_in_sublib) {
      //     // Hit -- Write to Db
      //     LocalSubDb.emplace_back(LocalSub(lname, "MysteryFile", 404, sublib));        
      //     std::cerr << "      sp: *** Found " << lname << " in library " << sublib << std::endl;
      //     std::cerr << "      sp: " << lname << " -> " << lib_it->second << std::endl;
      //   }
      // }
    }

  }

  /* Detect defined and called subroutines, check them against all selected
   * libraries and record matches in the appopriate stats objects
   */

  // Q1: Does proc.name() match any sublib routine names?

  // Set range of proc corresponding to Procedure::EXECUTION_PART
  auto execution_part{proc.crange(Procedure::EXECUTION_PART)};
  
  for (auto const &stmt : execution_part) {
    // Detect called procs and get their names, lcased for scanning
    // Q2: Does called proc name match any sublib routine names? 
    bool found_continue {false}; 
    int const stmt_tag = stmt.syntax_tag();
    found_continue = (   TAG(SG_CONTINUE_STMT) == stmt_tag
                      || TAG(KW_CONTINUE) == stmt_tag);
  }

  return changed;
}

/*--------------------------------------------------------------------------*/

/**
 *  \brief Test if a given procedure is excluded from processing
 *  \param[in] proc Procedure to check
 *  \returns Exclusion status; true if procedure is excluded from processing
 */
bool exclude_procedure(Procedure const &proc) {
  if (proc.headless_main_program()) {
    return false;
  }
  using Stmt_Const_Cursor = typename Procedure::Stmt_Const_Cursor;
  Stmt_Const_Cursor s =
      proc.range_cursor(Procedure::PROC_BEGIN)->stmt_tree().ccursor();
  s.down();
  if (TAG(SG_PREFIX) == s->syntag && s.has_down()) {
    s.down();
    do {
      assert(TAG(SG_PREFIX_SPEC) == s->syntag);
      s.down();
      if (TAG(KW_PURE) == s->syntag || TAG(KW_ELEMENTAL) == s->syntag)
        return true;
      s.up();
    } while (s.try_next());
    s.up();
  }
  return false;
}

/*--------------------------------------------------------------------------*/

/**
 *  \brief Write modified file contents (logical lines)
 *  \param[inout] os Output stream
 *  \param[in] f Source file object
 */
void write_file(std::ostream &os,
                File const &f) {
  for (auto const &ll : f.logical_lines()) {
    os << ll;
  }
}
