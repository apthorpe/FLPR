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
using Stmt_Cursor = typename File::Stmt_Cursor;

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

/// List of subroutines from IBM 1130 Scientific Subroutine Package
map_ss sublib_IBM_SSP {
// STATISTICS
// Data Screening
  {"tally", "Totals, means, standard deviations, minimums, and maximums, p. 13"},
  {"bound", "Selection of observations within bounds, p. 14"},
  {"subst", "Subset selection from observation matrix, p. 15"},
  {"absnt", "Detection of missing data, p. 16"},
  {"tab1", "Tabulation of data (1 variable), p. 16"},
  {"tab2", "Tabulation of data (2 variables), p. 18"},
  {"submx", "Build subset matrix, p. 20"},
// Elementary Statistics
  {"momen", "First four moments, p. 20"},
  {"ttstt", "Tests on population means, p. 21"},
// Correlation
  {"corre", "Means, standard deviations, and correlations, p. 23"},
// Multiple Linear Regression
  {"order", "Rearrangement of intercorrelations, p. 25"},
  {"multr", "Multiple regression and correlation, p. 26"},
// Polynomial Regression
  {"gdata", "Data generation, p. 28"},
// Canonical Correlation
  {"canor", "Canonical correlation, p. 30"},
  {"nroot", "Eigenvalues and eigenvectors of a special nonsymmetric matrix, p. 32"},
// Analysis of Variance
  {"avdat", "Data storage allocation, p. 34"},
  {"avcal", "Sigma and delta operation, p. 35"},
  {"meanq", "Mean square operation, p. 36"},
// Discriminant Analysis                                   
  {"dmatx", "Means and dispersion matrix, p. 38"},
  {"discr", "Discriminant functions, p. 39"},
// Factor Analysis                                         
  {"trace", "Cumulative percentage of eigenvalues, p. 41"},
  {"load", "Factor loading, p. 42"},
  {"varmx", "Varimax rotation, p. 43"},
// Time Series                                             
  {"auto", "Autocovariances, p. 46"},
  {"cross", "Crosscovariances, p. 47"},
  {"smo", "Application of filter coefficients (weights), p. 48"},
  {"exsmo", "Triple exponential smoothing, p. 49"},
// Nonparametric Statistics                                
  {"chisq", "Chi-squared test for a contingency table, p. 50"},
  {"utest", "Mann-Whitney U-test, p. 52"},
  {"twoav", "Friedman two-way analysis of variance, p. 53"},
  {"qtest", "Cochran Q-test, p. 54"},
  {"srank", "Spearman rank correlation, p. 55"},
  {"krank", "Kendall rank correlation, p. 56"},
  {"wtest", "Kendall coefficient of concordance, p. 58"},
  {"rank", "Rank observations, p. 59"},
  {"tie", "Calculation of ties in ranked observations, p. 59"},
// Random Number Generators
  {"randu", "Uniform random numbers, p. 60"},
  {"gauss", "Normal random numbers, p. 60"},
// MATHEMATICS OPERATIONS
// Special Matrix Operations
  {"minv", "Matrix inversion, p. 61"},
  {"eigen", "Eigenvalues and eigenvectors of a real, symmetric matrix, p. 62"},
// Matrices                                                                
  {"gmadd", "Add two general matrices, p. 64"},
  {"gmsub", "Subtract two general matrices, p. 64"},
  {"gmprd", "Product of two general matrices, p. 65"},
  {"gmtra", "Transpose of a general matrix, p. 65"},
  {"gtprd", "Transpose product of two general matrices, p. 66"},
  {"madd", "Add two matrices, p. 66"},
  {"msub", "Subtract two matrices, p. 67"},
  {"mprd", "Matrix product (row into column), p. 67"},
  {"mtra", "Transpose a matrix, p. 68"},
  {"tprd", "Transpose product, p. 68"},
  {"mata", "Transpose product of matrix by itself, p. 69"},
  {"sadd", "Add scalar to matrix, p. 69"},
  {"ssub", "Subtract scalar from a matrix, p. 70"},
  {"smpy", "Matrix multiplied by a scalar, p. 70"},
  {"sdiv", "Matrix divided by a scalar, p. 71"},
  {"radd", "Add row of one matrix to row of another matrix, p. 71"},
  {"cadd", "Add column of one matrix to column of another matrix, p. 72"},
  {"srma", "Scalar multiply row and add to another row, p. 72"},
  {"scma", "Scalar multiply column and add to another column, p. 73"},
  {"rint", "Interchange two rows, p. 73"},
  {"cint", "Interchange two columns, p. 74"},
  {"rsum", "Sum the rows of a matrix, p. 74"},
  {"csum", "Sum the columns of a matrix, p. 75"},
  {"rtab", "Tabulate the rows of a matrix, p. 75"},
  {"ctab", "Tabulate the columns of a matrix, p. 76"},
  {"rsrt", "Sort matrix rows, p. 77"},
  {"csrt", "Sort matrix columns, p. 78"},
  {"rcut", "Partition row-wise, p. 79"},
  {"ccut", "Partition column-wise, p. 79"},
  {"rtie", "Adjoin two matrices row-wise, p. 80"},
  {"ctie", "Adjoin two matrices column-wise, p. 80"},
  {"mcpy", "Matrix copy, p. 81"},
  {"xcpy", "Copy submatrix from given matrix, p. 81"},
  {"rcpy", "Copy row of matrix into vector, p. 82"},
  {"ccpy", "Copy column of matrix into vector, p. 82"},
  {"dcpy", "Copy diagonal of matrix into vector, p. 83"},
  {"scla", "Matrix clear and add scalar, p. 83"},
  {"dcla", "Replace diagonal with scalar, p. 84"},
  {"mstr", "Storage conversion, p. 84"},
  {"mfun", "Matrix transformation by a function, p. 85"},
  {"recp", "Reciprocal function for MFUN, p. 85"},
  {"loc", "Location in compressed-stored matrix, p. 86"},
  {"array", "Vector storage-double dimensioned storage conversion, p. 86"},
// Integration and Differentiation                                                                      
  {"qsf", "Integral of equidistantly tabulated function by Simpson's Rule, p. 87"},
  {"qatr", "Integral of given function by trapezoidal rule using Romberg's extrapolation method, p. 88"},
// Ordinary Differential Equations
  {"rk1", "Integral of first-order differential equation by Runge-Kutta method, p. 90"},
  {"rk2", "Tabulated integral of first-order differential equation by Runge-Kutta method, p. 91"},
  {"rkgs", "Solution of a system of first-order differential equations with given initial values by the Runge-Kutta method, p. 92"},
// Fourier Analysis
  {"forif", "Fourier analysis of a given function, p. 95"},
  {"forit", "Fourier analysis of a tabulated function, p. 96"},
// Special Operations and Functions
  {"gamma", "Gamma function, p. 97"}, 
  {"lep", "Legendre polynomial, p. 98"}, 
  {"besj", "J Bessel function, p. 99"}, 
  {"besy", "Y Bessel function, p. 101"},
  {"besi", "I Bessel function, p. 103"},
  {"besk", "K Bessel function, p. 104"},
  {"cell", "Elliptic integral of the first kind, p. 105"},
  {"cel2", "Elliptic integral of the second kind, p. 106"},
  {"expi", "Exponential integral, p. 108"},
  {"sici", "Sine cosine integral, p. 110"},
  {"cs", "Fresnel integrals, p. 112"},
// Linear Equations
  {"simq", "Solution of simultaneous linear, algebraic equations, p. 115"},
// Nonlinear Equations
  {"rtwi", "Refine estimate of root by Wegstein's iteration, p. 116"},
  {"rtmi", "Determine root within a range by Mueller's iteration, p. 117"},
  {"rtni", "Refine estimate of root by Newton's iteration, p. 119"},
// Roots of Polynomial                                                           
  {"polrt", "Real and complex roots of a real polynomial, p. 120"},
// Polynomial Operations                                                         
  {"padd", "Add two polynomials, p. 122"},
  {"paddm", "Multiply polynomial by constant and add to another polynomial, p. 122"},
  {"pcla", "Replace one polynomial by another, p. 122"},
  {"psub", "Subtract one polynomial from another, p. 123"},
  {"pmpy", "Multiply two polynomials, p. 123"},
  {"pdiv", "Divide one polynomial by another, p. 124"},
  {"pqsd", "Quadratic synthetic division of a polynomial, p. 124"},
  {"pval", "Value of a polynomial, p. 124"},
  {"pvsub", "Substitute variable of polynomial by another polynomial, p. 125"},
  {"pcld", "Complete linear synthetic division, p. 125"},
  {"pild", "Evaluate polynomial and its first derivative, p. 125"},
  {"pder", "Derivative of a polynomial, p. 126"},
  {"pint", "Integral of a polynomial, p. 126"},
  {"pgcd", "Greatest common divisor of two polynomials, p. 126"},
  {"pnorm", "Normalize coefficient vector of polynomial, p. 127"}
};

/// List of PDP-11 system routines; see "PDP-11 FORTRAN-77 User's Guide", AA-V194B-TK, Digital Equipment Corporation, 1988
map_ss sublib_PDP11_SYSTEM {
  {"assign", "Specified, at run time, device and/or file name information to be associated with a logical unit number, D.2"},
  {"close", "Closes a file on a specified logical unit, D.3"},
  {"date", "Returns a 9-byte string containing the ASCII representation of the current date, D.4"},
  {"idate", "Returns three integer values representing the current month, day, and year, D.5"},
  {"errset", "Specifies the action to be taken on the detection of certain errors, D.6"},
  {"errsns", "Returns information about the most recently detected error condition, D.7"},
  {"errtst", "Returns information about whether a specific error condition has occurred during program execution, D.8"},
  {"exit", "Terminates the execution of a program, reports termination status information, and returns control to the operating system, D.9"},
  {"userex", "Specifies a user subprogram to be called immediately prior to task termination, D.10"},
  {"fdbset", "Specifies special I/O options to be associated with a logical unit, D.11"},
  {"irad50", "Converts Hollerith strings to Radix-50 representation, D.12"},
  {"rad50", "Converts 6-character Hollerith strings to Radix-50 representation and returns the result as a function value, D.13"},
  {"r50asc", "Converts Radix-50 strings to Hollerith strings, D.14"},
  {"secnds", "Provides system time of day or elapsed time as a floating-point function value, in seconds, D.15"},
  {"time", "Returns an 8-byte string containing the ASCII representation of the current time in hours, minutes, and seconds, D.16"}
};

/// List of subroutines from UNIVAC scienfic library
map_ss sublib_UNIVAC_MATH_PACK {
// Interpolation
  {"gnint", "Gregory-newton interpolation"},
  {"gnext", "Gregory-newton extrapolation"},
  {"gnpol", "Gregory-newton polynomial evaluation"},
  {"besint", "Bessel interpolation"},
  {"stint", "Stirling interpolation"},
  {"cdint", "Gauss central-difference interpolation"},
  {"aitint", "Aitken interpolation"},
  {"ylgint", "Lagrange interpolation"},
  {"spln1", "Spline interpolation"},
  {"spln2", "Spline interpolation"},
// Numerical Integration
  {"trapni", "Trapezoidal rule"},
  {"sim1ni", "Simpson 1/3 rule"},
  {"sim3ni", "Simpson 3/8 rule"},
  {"stepni", "Variable step integration"},
  {"genni", "Generalized numerical quadrature"},
  {"doubni", "Double integration"},
  {"lgauss", "Gauss quadrature abscissas and weights"},
  {"simpts", "Simpson 1/3 rule abscissas and weights"},
// Solution of Equations
  {"newtit", "Newton-raphson iteration"},
  {"wegit", "Wegstein (altken) iteration"},
  {"aitit", "Aitken iteration"},
  {"rootcp", "Real and complex roots of a real or complex polynomial"},
// Differentiation
  {"deriv1", "First derivative approximation"},
  {"deriv2", "Second derivative approximation"},
  {"nthder", "Nth derivative of a polynomial"},
// Polynomial Manipulation
  {"givzrs", "Polynomial coefficients given its zeros"},
  {"cvalue", "Complex polynomial evaluation"},
  {"polyx", "Real polynomial multiplication"},
  {"cpolyx", "Complex polynomial multiplication"},
// Matrix Manipulation: Real Matrices
  {"mxadd", "Matrix addition"},
  {"mxsub", "Matrix subtraction"},
  {"mxtrn", "Matrix transposition"},
  {"mxmlt", "Matrix multiplication"},
  {"mxsca", "Matrix multiplication by a scalar"},
  {"mxmdig", "Matrix multiplication by diagonal matrix stored as a vector"},
  {"gjr", "Determinant; inverse; solution of simultaneous equations"},
  {"mxhoi", "Inverse accuracy improvement"},
// Matrix Manipulation: Complex Matrices
  {"cmxadd", "Matrix addition"},
  {"cmxsub", "Matrix subtraction"},
  {"cmxtrn", "Matrix transposition"},
  {"cmxmlt", "Matrix multiplication"},
  {"cmxsca", "Matrix multiplication by a scalar"},
  {"cgjr", "Determinant; inverse; solution of simultaneous equations"},
// Matrix Manipulation: Elgenvalues and Eigenvectors
  {"tridmx", "Tridiagonalization of real symmetric matrix"},
  {"eigval", "Elgenvalues of tridiagonal matrix by sturm sequences"},
  {"eigvec", "Elgenvectors of tridiagonal matrix by wilkinson's method"},
// Matrix Manipulation: Miscellaneous
  {"dgjr", "Double-precision determinant; inverse; solution of simultaneous equations"},
  {"pmxtri", "Polynomial matrix triangularization"},
  {"scale", "Polynomial matrix scaling"},
  {"mxrot", "Matrix rotation"},
// Ordinary Differential Equations
  {"eulde", "Euler's method"},
  {"hamde", "Hamming's method"},
  {"inval", "Initial values for differential equation solution"},
  {"rkde", "Runge-kutta method"},
  {"sode", "Second-order equations"},
  {"mrkde", "Reduction of mth order system to system of m first-order equations"},
// Systems of Equations
  {"hjacmx", "Jacobi iteration to determine eigenvalues and eigenvectors of hermitian matrix"},
  {"jacmx", "Jacobi iteration to determine eigenvalues and eigenvectors of symmetric matrix"},
  {"lsimeq", "Solution to a set of linear simultaneous equations"},
  {"nsimeq", "Functional iteration to determine solution to set of nonlinear equations"},
// Curve Fitting
  {"cfsrie", "Coefficients of fourier series on a continuous range"},
  {"dfsrie", "Coefficients of fourier series on a discrete range"},
  {"ftrans", "Fourier transform"},
  {"fito", "Fitted value and derivative values for a least-squares polynomial"},
  {"orthls", "Orthogonal polynomial least-squares curve fitting"},
  {"fity", "Fitted values for a least-squares polynomial"},
  {"coefs", "Coefficients of a least-squares polynomial"},
// Pseudo Random Number Generators
  {"nrand", "Interval (0,2**35) generator"},
  {"mrand", "Modified generator"},
  {"randu", "Uniform distribution"},
  {"randn", "Normal distribution"},
  {"randex", "Exponential distribution"},
// Specific Functions
  {"bssl", "Zero- and first-order bessel functions"},
  {"besj", "Regular (irregular) bessel functions of real argument"},
  {"besy", "Irregular bessel functions of real argument"},
  {"besi", "Regular (irregular) bessel functions of imaginary argument"},
  {"besk", "Irregular bessel functions of imaginary argument"},
  {"gamma", "Gamma function evaluation"},
  {"legen", "Legendre polynomial evaluation"},
  {"arctnq", "Arctangent of a quotient"},
};

/// List of subroutines from UNIVAC statistical library
map_ss sublib_UNIVAC_STAT_PACK {
// Descriptive Statistics
  {"freqp", "Frequency polygon"},
  {"hist", "Histogram"},
  {"mhist", "Multivariate histogram"},
  {"group", "Grouping of data"},
// Elementary Population Statistics
  {"amean", "Arithmetic mean"},
  {"gmean", "Geometric mean"},
  {"hmean", "Harmonic mean"},
  {"median", "Median"},
  {"mode", "Mode"},
  {"quant", "Quantiles"},
  {"ogive", "Distribution curve"},
  {"iqrng", "Interpercentile range"},
  {"range", "Range"},
  {"mndev", "Mean deviation"},
  {"stdev", "Standard deviation"},
  {"cvar", "Coefficient of variation"},
  {"order", "Order and rank statistics"},
  {"cmont", "Central moments"},
  {"amont", "Absolute moments"},
  {"cumlt", "Cumulants"},
  {"shpcor", "Sheppard's corrections"},
  {"kursk", "Skewness and kurtosis"},
// Distribution, Fitting, and Plotting
  {"binom", "Binomial distribution"},
  {"poison", "Poisson distribution"},
  {"hyper", "Hypergeometric distribution"},
  {"pnorm", "Normal distribution"},
  {"afser", "Arne-fisher series"},
// Chi-Square Tests
  {"chi21s", "Chi-square test of sample proportion for one sample"},
  {"chi2js", "Chi-square test of sample proportion for j samples"},
  {"chi2p", "Chi-square test of fit to poisson distribution"},
  {"chi2n", "Chi-square test of normality"},
  {"chisam", "Chi-square test of homogeneity"},
  {"chicnt", "Chi-square test for independence"},
  {"gengof", "Chi-square test of general goodness of fit"},
// Significance Tests
  {"sigprp", "Test of significance of proportion of successes"},
  {"sigmn", "Test of significance of a mean"},
  {"sigdmn", "Test of significance of the difference between two means"},
  {"sigdvr", "Test of significance of the ratio between two variances"},
// Confidence Intervals
  {"cfdmkv", "Confidence interval for the mean: known variance"},
  {"cfdmuv", "Confidence interval for the mean: unknown variance"},
  {"cfdmsu", "Confidence interval for the difference between two means"},
  {"cfdvar", "Confidence interval for variance"},
  {"tolint", "Tolerance intervals"},
// Analysis of Variance
  {"anov1", "One-way cross-classification"},
  {"anov2", "Two-way cross-classification"},
  {"anov3", "Three-way cross-classification"},
  {"misdat", "Missing data"},
  {"vtrans", "Variable transformations"},
  {"anovrb", "Randomized blocks"},
  {"anovls", "Latin squares"},
  {"anovsp", "Split-plot design"},
  {"anossp", "Split-split plot design"},
  {"anovn2", "Two-way nested design"},
  {"anovn3", "Three-way nested design"},
  {"anoco", "Analysis of covariance"},
  {"glh", "General linear hypotheses"},
// Regression Analysis
  {"restem", "Stepwise multiple regression"},
  {"rebsom", "Back solution multiple regression"},
  {"coran", "Correlation analysis"},
// Time Series Analysis
  {"movavg", "Moving averages"},
  {"seashi", "Shiskin's seasonality factors"},
  {"wemav", "Weighted moving averages"},
  {"trels", "Trend analysis by least squares"},
  {"vadime", "Variate difference method"},
  {"tsfarg", "Autoregressive model"},
  {"gexsmo", "Generalized exponential smoothing"},
  {"auxcor", "Auto-correlation and cross-correlation analysis"},
  {"powden", "Power density functions"},
  {"rcprob", "Residual probabilities"},
// Multivariate Analysis
  {"genvar", "Generalized variance"},
  {"dishot", "Hotelling's distribution"},
  {"dsq", "Mahalanobis' distribution"},
  {"sigtmn", "Significance of a set of means"},
  {"discra", "Discriminant analysis"},
  {"factan", "Factor and principal components analysis"},
// Distribution Functions
  {"rnorm", "Normal distribution"},
  {"chi", "Chi-square distribution"},
  {"stud", "Student's distribution"},
  {"fish", "Fisher's distribution"},
  {"pois", "Poisson distribution"},
  {"bin", "Binomial distribution"},
  {"hygeo", "Hypergeometric distribution"},
  {"gamin", "Incomplete gamma distribution"},
  {"betinc", "Incomplete beta distribution"},
// Inverse Distribution Functions
  {"tinorm", "Inverse normal distribution"},
  {"studin", "Inverse student's distribution"},
  {"fishin", "Inverse fisher's distribution"},
  {"chin", "Inverse chi-square distribution"},
// Miscellaneous Subroutines
  {"plot1", "Plot of one line"},
  {"jim", "Matrix inversion"},
  {"mxtmlt", "Left multiplication of a matrix by its transpose"},
};

/// List of subroutines from BLAS, Level 1
map_ss sublib_BLAS1 {
  {"caxpy", "Compute a scalar times a vector plus a vector, y = ax + y, all complex. (Level 1 BLAS)."},
  {"ccopy", "Copy a vector X to a vector Y, both complex. (Level 1 BLAS)."},
  {"cdotc", "Compute complex dot product using conjugated vector components. (Level 1 BLAS)."},
  {"cdotu", "Compute complex dot product using unconjugated vector components. (Level 1 BLAS)."},
  {"cscal", "Multiply a vector by a scalar, y = ay, both complex. (Level 1 BLAS)."},
  {"csrot", "Applies Givens plane rotation to complex matrix. (Level 1 BLAS)."},
  {"csscal", "Multiply a complex vector by a scalar, y = ay. (Level 1 BLAS)."},
  {"cswap", "Interchange vectors X and Y, both complex. (Level 1 BLAS)."},
  {"dasum", "Compute sum of absolute values of a vector. (Level 1 BLAS)."},
  {"daxpy", "Compute the scalar times a vector plus a vector, y = ax + y. (Level 1 BLAS)."},
  {"dcopy", "Copy a vector X to a vector Y. (Level 1 BLAS)."},
  {"ddot", "Compute dot product of two vectors. (Level 1 BLAS)."},
  {"dnrm2", "Compute the Euclidean length or L2 norm of a vector. (Level 1 BLAS)."},
  {"drot", "Apply Givens plane rotation to a vector. (Level 1 BLAS)."},
  {"drotg", "Construct Givens plane rotation of a matrix. (Level 1 BLAS)."},
  {"dscal", "Compute a constant times a vector, y = ay. (Level 1 BLAS)."},
  {"dswap", "Interchange vectors X and Y. (Level 1 BLAS)."},
  {"dzasum", "Compute complex sum of absolute values of components of complex vector. (Level 1 BLAS)."},
  {"dznrm2", "Compute the Euclidean length or L2 norm of a complex vector. (Level 1 BLAS)."},
  {"icamax", "Find smallest index of maximum magnitude component of a complex vector. (Level 1 BLAS)."},
  {"idamax", "Find smallest index of maximum magnitude component of a vector. (Level 1 BLAS)."},
  {"isamax", "Find smallest index of maximum magnitude component of a vector. (Level 1 BLAS)."},
  {"izamax", "Find smallest index of maximum magnitude component of a complex vector. (Level 1 BLAS)."},
  {"sasum", "Compute sum of absolute values of components of vector. (Level 1 BLAS)."},
  {"saxpy", "Compute a constant times a vector plus a vector. (Level 1 BLAS)."},
  {"scasum", "Compute complex sum of absolute values of components of vector. (Level 1 BLAS)."},
  {"scnrm2", "Compute the Euclidean length or L2 norm of a complex vector. (Level 1 BLAS)."},
  {"scopy", "Copy a vector X to a vector Y. (Level 1 BLAS)."},
  {"sdot", "Compute dot product of two vectors. (Level 1 BLAS)."},
  {"snrm2", "Compute the Euclidean length or L2 norm of a vector, without underflow or overflow. (Level 1 BLAS)."},
  {"srot", "Apply Givens plane rotation to a vector. (Level 1 BLAS)."},
  {"srotg", "Construct Givens plane rotation of a matrix. (Level 1 BLAS)."},
  {"sscal", "Multiply a vector by a scalar, y =ay. (Level 1 BLAS)."},
  {"sswap", "Interchange vectors X and Y. (Level 1 BLAS)."},
  {"zaxpy", "Compute a scalar times a vector plus a vector, y = ax + y, all complex. (Level 1 BLAS)."},
  {"zcopy", "Copy a vector X to a vector Y, both complex. (Level 1 BLAS)."},
  {"zdotc", "Compute complex dot product using conjugated vector components. (Level 1 BLAS)."},
  {"zdotu", "Compute complex dot product using unconjugated vector components. (Level 1 BLAS)."},
  {"zdrot", "Applies Givens plane rotation to complex matrix. (Level 1 BLAS)."},
  {"zdscal", "Multiply a complex vector by a double precision scalar, y = ay. (Level 1 BLAS)."},
  {"zscal", "Multiply a vector by a scalar, y = ay, both complex. (Level 1 BLAS)."},
  {"zswap", "Interchange vectors X and Y, both complex. (Level 1 BLAS)."}
};

/// List of subroutines from BLAS, Level 2
map_ss sublib_BLAS2 {
  {"cgbmv", "Performs one of the matrix-vector operations y := alpha*A*x + beta*y, or y := alpha*A''*x + beta*y, or y := alpha*conjg( A'' )*x + beta*y, where alpha and beta are complex scalars, x and y are vectors and A is an m by n complex band matrix, with kl sub-diagonals and ku super-diagonals. (Level 2 BLAS)."},
  {"cgemv", "Performs one of the matrix-vector operations y := alpha*A*x + beta*y, or y := alpha*A''*x + beta*y, or y := alpha*conjg( A'' )*x + beta*y, where alpha and beta are complex scalars, x and y are complex vectors and A is an m by n complex matrix. (Level 2 BLAS)."},
  {"cgerc", "Performs conjugated rank 1 update of a complex general matrix. (Level 2 BLAS)."},
  {"cgeru", "Performs unconjugated rank 1 update of a complex general matrix. (Level 2 BLAS)."},
  {"chbmv", "Performs the matrix-vector operation y := alpha*A*x + beta*y, where alpha and beta are complex scalars, x and y are n element complex vectors and A is an n by n complex Hermitian band matrix, with k super-diagonals. (Level 2 BLAS).#."},
  {"chemv", "Performs the matrix-vector operation y := alpha*A*x + beta*y, where alpha and beta are complex scalars, x and y are n element complex vectors and A is an n by n complex Hermitian matrix. (Level 2 BLAS)."},
  {"cher", "Performs Hermitian rank 1 update of a complex Hermitian matrix. (Level 2 BLAS)."},
  {"cher2", "Performs Hermitian rank 2 update of a complex Hermitian matrix. (Level 2 BLAS)."},
  {"chpmv", "Performs the matrix-vector operation y := alpha*A*x + beta*y, where alpha and beta are complex scalars, x and y are n element complex vectors and A is an n by n complex Hermitian matrix, supplied in packed form. (Level 2 BLAS)."},
  {"chpr", "Performs Hermitian rank 1 update of a packed complex Hermitian matrix. (Level 2 BLAS)."},
  {"chpr2", "Performs Hermitian rank 2 update of a packed complex Hermitian matrix. (Level 2 BLAS)."},
  {"ctbmv", "Performs one of the matrix-vector operations x := A*x, or x := A''*x, or x := conjg( A'' )*x, where x is an n element complex vector and A is an n by n unit, or non-unit, upper or lower triangular complex band matrix, with k+1 diagonals. (Level 2 BLAS)."},
  {"ctbsv", "Solves a complex triangular banded system of equations. (Level 2 BLAS)."},
  {"ctpmv", "Performs one of the matrix-vector operations x := A*x, or x := A''*x, or x := conjg( A'' )*x, where x is an n element complex vector and A is an n by n unit, or non-unit, upper or lower triangular complex matrix, supplied in packed form. (Level 2 BLAS)."},
  {"ctpsv", "Solves a complex triangular packed system of equations. (Level 2 BLAS)."},
  {"ctrmv", "Performs one of the matrix-vector operations x := A*x, or x := A''*x, or x := conjg( A'' )*x, where x is an n element complex vector and A is an n by n unit, or non-unit, upper or lower triangular complex matrix. (Level 2 BLAS)."},
  {"ctrsv", "Solves a complex triangular system of equations. (Level 2 BLAS)."},
  {"dgbmv", "Multiplies a real vector by a real general band matrix. (Level 2 BLAS)."},
  {"dgemv", "Performs one of the matrix-vector operations y := alpha*A*x + beta*y, or y := alpha*A''*x + beta*y, where alpha and beta are scalars, x and y are vectors and A is an m by n matrix. (Level 2 BLAS)."},
  {"dger", "Performs rank 1 update of a real general matrix. (Level 2 BLAS)."},
  {"dsbmv", "Performs the following matrix-vector operation: y <-- alpha*A*x + beta*y, where alpha and beta are scalars, x and y are n-element vectors, and A is an n-by-n symmetric band matrix. (Level 2 BLAS)."},
  {"dspmv", "Performs the following matrix-vector operation: y <-- alpha*A*x + beta*y, where alpha and beta are real scalars, x and y are n-element vectors, and A is an n-by-n symmetric packed matrix. (Level 2 BLAS)."},
  {"dspr", "Performs symmetric rank 1 update of a real symmetric packed matrix. (Level 2 BLAS)."},
  {"dspr2", "Performs symmetric rank 2 update of a real symmetric packed matrix. (Level 2 BLAS)."},
  {"dsymv", "Performs the following matrix-vector operation: y <-- alpha*A*x + beta*y, where alpha and beta are scalars, x and y are n-element vectors, and A is an n-by-n symmetric matrix. (Level 2 BLAS)."},
  {"dsyr", "Performs symmetric rank 1 update of a real symmetric matrix. (Level 2 BLAS)."},
  {"dsyr2", "Performs symmetric rank 2 update of a real symmetric matrix. (Level 2 BLAS)."},
  {"dtbmv", "Performs one of the following matrix-vector operations: x <-- A*x, x <-- (transpose of A)*x, where x is an n-element vector, and A is an n-by-n unit, or non-unit, upper or lower triangular band matrix with k+1 diagonals. (Level 2 BLAS)."},
  {"dtbsv", "Solves a real triangular banded system of equations. (Level 2 BLAS)."},
  {"dtpmv", "Performs one of the following matrix-vector operations: x <-- A*x, or x <-- (transpose of A)*x, where x is an n-element vector and A is an n-by-n unit, or non-unit, upper or lower triangular matrix. (Level 2 BLAS)."},
  {"dtpsv", "Solves a real triangular packed system of equations. (Level 2 BLAS)."},
  {"dtrmv", "Performs one of the following matrix-vector operations: x <-- A*x or x <-- (transpose of A)*x, where x is an n-element vector, A is an n-by-n unit, or non-unit, upper or lower triangular matrix. (Level 2 BLAS)."},
  {"dtrsv", "Solves a real triangular system of equations. (Level 2 BLAS)."},
  {"sgbmv", "Multiplies a real vector by a real general band matrix. (Level 2 BLAS)."},
  {"sgemv", "Performs one of the matrix-vector operations y := alpha*A*x + beta*y, or y := alpha*A''*x + beta*y, where alpha and beta are scalars, x and y are vectors and A is an m by n matrix. (Level 2 BLAS)."},
  {"sger", "Performs rank 1 update of a real general matrix. (Level 2 BLAS)."},
  {"ssbmv", "Performs the following matrix-vector operation: y <-- alpha*A*x + beta*y, where alpha and beta are scalars, x and y are n-element vectors, and A is an n-by-n symmetric band matrix. (Level 2 BLAS)."},
  {"sspmv", "Performs the following matrix-vector operation: y <-- alpha*A*x + beta*y, where alpha and beta are real scalars, x and y are n-element vectors, and A is an n-by-n symmetric packed matrix. (Level 2 BLAS)."},
  {"sspr", "Performs symmetric rank 1 update of a real symmetric packed matrix. (Level 2 BLAS)."},
  {"sspr2", "Performs symmetric rank 2 update of a real symmetric packed matrix. (Level 2 BLAS)."},
  {"ssymv", "Performs the following matrix-vector operation: y <-- alpha*A*x + beta*y, where alpha and beta are scalars, x and y are n-element vectors, and A is an n-by-n symmetric matrix. (Level 2 BLAS)."},
  {"ssyr", "Performs symmetric rank 1 update of a real symmetric matrix. (Level 2 BLAS)."},
  {"ssyr2", "Performs symmetric rank 2 update of a real symmetric matrix. (Level 2 BLAS)."},
  {"stbmv", "Performs one of the following matrix-vector operations: x <-- A*x, x <-- (transpose of A)*x, where x is an n-element vector, and A is an n-by-n unit, or non-unit, upper or lower triangular band matrix with k+1 diagonals. (Level 2 BLAS)."},
  {"stbsv", "Solves a real triangular banded system of equations. (Level 2 BLAS)."},
  {"stpmv", "Performs one of the following matrix-vector operations: x <-- A*x, or x <-- (transpose of A)*x, where x is an n-element vector and A is an n-by-n unit, or non-unit, upper or lower triangular matrix. (Level 2 BLAS)."},
  {"stpsv", "Solves a real triangular packed system of equations. (Level 2 BLAS)."},
  {"strmv", "Performs one of the following matrix-vector operations: x <-- A*x or x <-- (transpose of A)*x, where x is an n-element vector, A is an n-by-n unit, or non-unit, upper or lower triangular matrix. (Level 2 BLAS)."},
  {"strsv", "Solves a real triangular system of equations. (Level 2 BLAS)."},
  {"zgbmv", "Performs one of the matrix-vector operations y := alpha*A*x + beta*y, or y := alpha*A''*x + beta*y, or y := alpha*conjg( A'' )*x + beta*y, where alpha and beta are complex scalars, x and y are vectors and A is an m by n complex band matrix, with kl sub-diagonals and ku super-diagonals. (Level 2 BLAS)."},
  {"zgemv", "Performs one of the matrix-vector operations y := alpha*A*x + beta*y, or y := alpha*A''*x + beta*y, or y := alpha*conjg( A'' )*x + beta*y, where alpha and beta are complex scalars, x and y are complex vectors and A is an m by n complex matrix. (Level 2 BLAS)."},
  {"zgerc", "Performs conjugated rank 1 update of a complex general matrix. (Level 2 BLAS)."},
  {"zgeru", "Performs unconjugated rank 1 update of a complex general matrix. (Level 2 BLAS)."},
  {"zhbmv", "Performs the matrix-vector operation y := alpha*A*x + beta*y, where alpha and beta are complex scalars, x and y are n element complex vectors and A is an n by n complex Hermitian band matrix, with k super-diagonals. (Level 2 BLAS).#."},
  {"zhemv", "Performs the matrix-vector operation y := alpha*A*x + beta*y, where alpha and beta are complex scalars, x and y are n element complex vectors and A is an n by n complex Hermitian matrix. (Level 2 BLAS)."},
  {"zher", "Performs Hermitian rank 1 update of a complex Hermitian matrix. (Level 2 BLAS)."},
  {"zher2", "Performs Hermitian rank 2 update of a complex Hermitian matrix. (Level 2 BLAS)."},
  {"zhpmv", "Performs the matrix-vector operation y := alpha*A*x + beta*y, where alpha and beta are complex scalars, x and y are n element complex vectors and A is an n by n complex Hermitian matrix, supplied in packed form. (Level 2 BLAS)."},
  {"zhpr", "Performs Hermitian rank 1 update of a packed complex Hermitian matrix. (Level 2 BLAS)."},
  {"zhpr2", "Performs Hermitian rank 2 update of a packed complex Hermitian matrix. (Level 2 BLAS)."},
  {"ztbmv", "Performs one of the matrix-vector operations x := A*x, or x := A''*x, or x := conjg( A'' )*x, where x is an n element complex vector and A is an n by n unit, or non-unit, upper or lower triangular complex band matrix, with k+1 diagonals. (Level 2 BLAS)."},
  {"ztbsv", "Solves a complex triangular banded system of equations. (Level 2 BLAS)."},
  {"ztpmv", "Performs one of the matrix-vector operations x := A*x, or x := A''*x, or x := conjg( A'' )*x, where x is an n element complex vector and A is an n by n unit, or non-unit, upper or lower triangular complex matrix, supplied in packed form. (Level 2 BLAS)."},
  {"ztpsv", "Solves a complex triangular packed system of equations. (Level 2 BLAS)."},
  {"ztrmv", "Performs one of the matrix-vector operations x := A*x, or x := A''*x, or x := conjg( A'' )*x, where x is an n element complex vector and A is an n by n unit, or non-unit, upper or lower triangular complex matrix. (Level 2 BLAS)."},
  {"ztrsv", "Solves a complex triangular system of equations. (Level 2 BLAS)."}
};

/// List of subroutines from BLAS, Level 3
map_ss sublib_BLAS3 {
  {"cgemm", "Performs one of the matrix-matrix operations C := alpha*op( A )*op( B ) + beta*C, where op( X ) is one of op( X ) = X or op( X ) = X'' or op( X ) = conjg( X'' ), alpha and beta are complex scalars, and A, B and C are complex matrices, with op( A ) an m by k matrix, op( B ) a k by n matrix and C an m by n matrix. (Level 3 BLAS)."},
  {"chemm", "Performs one of the matrix-matrix operations C := alpha*A*B + beta*C, or C := alpha*B*A + beta*C, where alpha and beta are complex scalars, A is a complex Hermitian matrix and B and C are m by n complex matrices. (Level 3 BLAS)."},
  {"cher2k", "Performs Hermitian rank 2k update of a complex Hermitian matrix. (Level 3 BLAS)."},
  {"cherk", "Performs Hermitian rank k update of a complex Hermitian matrix. (Level 3 BLAS)."},
  {"csymm", "Performs one of the matrix-matrix operations C := alpha*A*B + beta*C, or C := alpha*B*A + beta*C, where alpha and beta are complex scalars, A is a complex symmetric matrix and B and C are m by n complex matrices. (Level 3 BLAS)."},
  {"csyr2k", "Performs symmetric rank 2k update of a complex symmetric matrix. (Level 3 BLAS)."},
  {"csyrk", "Performs symmetric rank k update of a complex symmetric matrix. (Level 3 BLAS)."},
  {"ctrmm", "Performs one of the matrix-matrix operations B := alpha*op( A )*B, or B := alpha*B*op( A ) where alpha is a complex scalar, B is an m by n complex matrix, A is a unit, or non-unit, upper or lower triangular complex matrix and op( A ) is one of op( A ) = A, or op( A ) = A'', or op( A ) = conjg( A''). (Level 3 BLAS)."},
  {"ctrsm", "Solves a complex triangular system of equations with multiple right-hand sides. (Level 3 BLAS)."},
  {"dgemm", "Performs one of the matrix-matrix operations C := alpha*op( A )*op( B ) + beta*C, where op( X ) is one of op( X ) = X, or op( X ) = X'', alpha and beta are scalars, and A, B and C are matrices, with op( A ) an m by k matrix, op( B ) is a k by n matrix and C an m by n matrix. (Level 3 BLAS)."},
  {"dsymm", "Performs one of the following matrix-matrix operations: C <-- alpha*A*B + beta*C, or C <-- alpha*B*A + beta*C, where alpha and beta are scalars, A is a symmetric matrix, and B and C are m-by-n matrices. (Level 3 BLAS)."},
  {"dsyr2k", "Performs symmetric rank 2k update of a real symmetric matrix. (Level 3 BLAS)."},
  {"dsyrk", "Performs symmetric rank k update of a real symmetric matrix. (Level 3 BLAS)."},
  {"dtrmm", "Perform one of the matrix-matrix operations: B <-- alpha*op(a)*B, or B <-- alpha*B*op(a), where alpha is a scalar, B is an m-by-n matrix, A is a unit, or non-unit, upper or lower triangular matrix, and op(A) is one of the following: op(A) = A, or op(A) = transpose of A. (Level 3 BLAS)."},
  {"dtrsm", "Solves a real triangular system of equations with multiple right-hand sides. (Level 3 BLAS)."},
  {"sgemm", "Performs one of the matrix-matrix operations C := alpha*op( A )*op( B ) + beta*C, where op( X ) is one of op( X ) = X, or op( X ) = X'', alpha and beta are scalars, and A, B and C are matrices, with op( A ) an m by k matrix, op( B ) is a k by n matrix and C an m by n matrix. (Level 3 BLAS)."},
  {"ssymm", "Performs one of the following matrix-matrix operations: C <-- alpha*A*B + beta*C, or C <-- alpha*B*A + beta*C, where alpha and beta are scalars, A is a symmetric matrix, and B and C are m-by-n matrices. (Level 3 BLAS)."},
  {"ssyr2k", "Performs symmetric rank 2k update of a real symmetric matrix. (Level 3 BLAS)."},
  {"ssyrk", "Performs symmetric rank k update of a real symmetric matrix. (Level 3 BLAS)."},
  {"strmm", "Perform one of the matrix-matrix operations: B <-- alpha*op(a)*B, or B <-- alpha*B*op(a), where alpha is a scalar, B is an m-by-n matrix, A is a unit, or non-unit, upper or lower triangular matrix, and op(A) is one of the following: op(A) = A, or op(A) = transpose of A. (Level 3 BLAS)."},
  {"strsm", "Solves a real triangular system of equations with multiple right-hand sides. (Level 3 BLAS)."},
  {"zgemm", "Performs one of the matrix-matrix operations C := alpha*op( A )*op( B ) + beta*C, where op( X ) is one of op( X ) = X or op( X ) = X'' or op( X ) = conjg( X'' ), alpha and beta are complex scalars, and A, B and C are complex matrices, with op( A ) an m by k matrix, op( B ) a k by n matrix and C an m by n matrix. (Level 3 BLAS)."},
  {"zhemm", "Performs one of the matrix-matrix operations C := alpha*A*B + beta*C, or C := alpha*B*A + beta*C, where alpha and beta are complex scalars, A is a complex Hermitian matrix and B and C are m by n complex matrices. (Level 3 BLAS)."},
  {"zher2k", "Performs Hermitian rank 2k update of a complex Hermitian matrix. (Level 3 BLAS)."},
  {"zherk", "Performs Hermitian rank k update of a complex Hermitian matrix. (Level 3 BLAS)."},
  {"zsymm", "Performs one of the matrix-matrix operations C := alpha*A*B + beta*C, or C := alpha*B*A + beta*C, where alpha and beta are complex scalars, A is a complex symmetric matrix and B and C are m by n complex matrices. (Level 3 BLAS)."},
  {"zsyr2k", "Performs symmetric rank 2k update of a complex symmetric matrix. (Level 3 BLAS)."},
  {"zsyrk", "Performs symmetric rank k update of a complex symmetric matrix. (Level 3 BLAS)."},
  {"ztrmm", "Performs one of the matrix-matrix operations B := alpha*op( A )*B, or B := alpha*B*op( A ) where alpha is a complex scalar, B is an m by n complex matrix, A is a unit, or non-unit, upper or lower triangular complex matrix and op( A ) is one of op( A ) = A, or op( A ) = A'', or op( A ) = conjg( A''). (Level 3 BLAS)."},
  {"ztrsm", "Solves a complex triangular system of equations with multiple right-hand sides. (Level 3 BLAS)."}
};

/// List of subroutines from LINPACK
map_ss sublib_LINPACK {
  {"cchdc", "Compute Cholesky decomposition of complex positive definite matrix with optional pivoting."},
  {"cchdd", "Downdates Cholesky factorization of positive definite complex matrix."},
  {"cchex", "Updates Cholesky factorization of positive definite complex matrix."},
  {"cchud", "Updates Cholesky factorization of positive definite matrix."},
  {"cgbco", "Compute LU factorization of complex band matrix and estimate its condition."},
  {"cgbdi", "Compute determinant of complex band matrix from its LU factors. (No provision for computing inverse directly.)."},
  {"cgbfa", "Compute LU factorization of general complex band matrix."},
  {"cgbsl", "Uses LU factorization of complex band matrix to solve systems."},
  {"cgeco", "Compute LU factorization of general complex matrix and estimate its condition."},
  {"cgedi", "Compute determinant and/or inverse of general complex matrix from its LU factors."},
  {"cgefa", "Compute LU factorization of general complex matrix."},
  {"cgesl", "Use LU factorization of general complex matrix to solve systems."},
  {"cgtsl", "Solves systems with general complex tridiagonal matrix."},
  {"chico", "Computes factorization of complex Hermitian indefinite matrix and estimates its condition."},
  {"chidi", "Uses factorization of complex Hermitian indefinite matrix to compute its inertia, determinant, and/or inverse."},
  {"chifa", "Computes factorization of complex Hermitian indefinite matrix."},
  {"chisl", "Uses factorization of complex Hermitian indefinite matrix to solve systems."},
  {"chpco", "Computes factorization of complex Hermitian indefinite matrix stored in packed form and estimates its condition."},
  {"chpdi", "Uses factorization of complex Hermitian indefinite matrix stored in packed form to compute its inertia, determinant, and inverse."},
  {"chpfa", "Computes factorization of complex Hermitian indefinite matrix stored in packed form."},
  {"chpsl", "Uses factorization of complex Hermitian indefinite matrix stored in packed form to solve systems."},
  {"cpbco", "Uses Cholesky algorithm to compute factorization of complex positive definite band matrix and estimates its condition."},
  {"cpbdi", "Uses factorization of complex positive definite band matrix to compute determinant. (No provision for computing inverse.)."},
  {"cpbfa", "Uses Cholesky algorithm to compute factorization of complex positive definite band matrix."},
  {"cpbsl", "Uses factorization of complex positive definite band matrix to solve systems."},
  {"cpoco", "Uses Cholesky algorithm to compute factorization of complex positive definite matrix and estimates its condition."},
  {"cpodi", "Uses factorization of complex positive definite matrix to compute its determinant and/or inverse."},
  {"cpofa", "Uses Cholesky algorithm to compute factorization of complex positive definite matrix."},
  {"cposl", "Uses factorization of complex positive definite matrix to solve systems."},
  {"cppco", "Uses Cholesky algorithm to factor complex positive definite matrix stored in packed form."},
  {"cppdi", "Uses factorization of complex positive definite matrix stored in packed form to compute determinant and/or inverse."},
  {"cppfa", "Uses Cholesky algorithm to factor complex positive definite matrix stored in packed form."},
  {"cppsl", "Uses factorization of complex positive definite matrix stored in packed form to solve systems."},
  {"cptsl", "Solves systems with complex positive definite tridiagonal matrix."},
  {"cqrdc", "Computes QR decomposition of general complex matrix."},
  {"cqrsl", "Applies the output of CQRDC to compute coordinate transformations, projections, and least squares solutions (general complex matrix)."},
  {"csico", "Computes factorization of complex symmetric indefinite matrix and estimates its condition."},
  {"csidi", "Uses factorization of complex symmetric indefinite matrix to compute its determinant and/or inverse."},
  {"csifa", "Computes factorization of complex symmetric indefinite matrix."},
  {"csisl", "Uses factorization of complex symmetric indefinite matrix to solve systems."},
  {"cspco", "Computes factorization of complex symmetric indefinite matrix stored in packed form and computes its condition."},
  {"cspdi", "Uses factorization of complex symmetric indefinite matrix stored in packed form to compute its determinant and/or inverse."},
  {"cspfa", "Computes factorization of complex symmetric indefinite matrix stored in packed form."},
  {"cspsl", "Uses factorization of complex symmetric indefinite matrix stored in packed form to solve systems."},
  {"csvdc", "Computes singular value decomposition of general complex matrix."},
  {"ctrco", "Estimates condition of complex triangular matrix."},
  {"ctrdi", "Computes determinant and/or inverse of complex triangular matrix."},
  {"ctrsl", "Solves systems with complex triangular matrix."},
  {"dchdc", "Compute Cholesky decomposition of positive definite double precision matrix with optional pivoting."},
  {"dchdd", "Downdates Cholesky factorization of positive definite double precision matrix."},
  {"dchex", "Updates Cholesky factorization of positive definite double precision matrix."},
  {"dchud", "Updates Cholesky factorization of positive definite double precision matrix."},
  {"dgbco", "Computes LU factorization of general double precision band matrix and estimates its condition."},
  {"dgbdi", "Uses LU factorization of general double precision band matrix to compute its determinant. (No provision for inverse computation.)."},
  {"dgbfa", "Computes LU factorization of general double precision band matrix."},
  {"dgbsl", "Uses LU factorization of general double precision band matrix to solve systems."},
  {"dgeco", "Compute LU factorization of general double precision matrix and estimate its condition."},
  {"dgedi", "Uses LU factorization of general double precision matrix to compute its determinant and/or inverse."},
  {"dgefa", "Compute LU factorization of general double precision matrix."},
  {"dgesl", "Uses LU factorization of general double precision matrix to solve systems."},
  {"dgtsl", "Solve systems with tridiagonal double precision matrix."},
  {"dpbco", "Compute LU factorization of double precision positive definite band matrix and estimate its condition."},
  {"dpbdi", "Use LU factorization of double precision positive definite band matrix to compute determinant. (No provision for inverse.)."},
  {"dpbfa", "Computes LU factorization of double precision positive definite band matrix."},
  {"dpbsl", "Uses LU factorization of double precision positive definite band matrix to solve systems."},
  {"dpoco", "Use Cholesky algorithm to factor double precision positive definite matrix and estimate its condition."},
  {"dpodi", "Use factorization of double precision positive definite matrix to compute determinant and/or inverse."},
  {"dpofa", "Use Cholesky algorithm to factor double precision positive definite matrix."},
  {"dposl", "Use factorization of double precision positive definite matrix to solve systems."},
  {"dppco", "Use Cholesky algorithm to factor double precision positive definite matrix stored in packed form and estimate its condition."},
  {"dppdi", "Use factorization of double precision positive definite matrix stored in packed form to compute determinant and/or inverse."},
  {"dppfa", "Use Cholesky algorithm to factor double precision positive definite matrix stored in packed form."},
  {"dppsl", "Use factorization of double precision positive definite matrix stored in packed form to solve systems."},
  {"dptsl", "Decomposes double precision symmetric positive definite tridiagonal matrix and simultaneously solve a system."},
  {"dqrdc", "Compute QR decomposition of general double precision matrix."},
  {"dqrsl", "Applies the output of DQRDC to compute coordinate transformations, projections, and least squares solutions (general double precision matrix)."},
  {"dsico", "Computes factorization of double precision symmetric indefinite matrix and estimate its condition."},
  {"dsidi", "Use factorization of double precision symmetric indefinite matrix to compute determinant and/or inverse."},
  {"dsifa", "Compute factorization of double precision symmetric indefinite matrix."},
  {"dsisl", "Use factorization of double precision symmetric indefinite matrix to solve systems."},
  {"dspco", "Compute factorization of double precision symmetric indefinite matrix stored in packed form and estimate its condition."},
  {"dspdi", "Use factorization of double precision symmetric indefinite matrix stored in packed form to compute determinant and/or inverse."},
  {"dspfa", "Compute factorization of double precision symmetric indefinite matrix stored in packed form."},
  {"dspsl", "Use factorization of double precision symmetric indefinite matrix stored in packed form to solve systems."},
  {"dsvdc", "Compute Singular Value Decomposition of double precision matrix. Has options to allow computation of only the singular values, or singular values and associated decomposition matrices."},
  {"dtrco", "Estimates condition of double precision triangular matrix."},
  {"dtrdi", "Computes determinant and/or inverse of double precision triangular matrix."},
  {"dtrsl", "Solves systems with double precision triangular matrix."},
  {"schdc", "Compute Cholesky decomposition of real positive definite matrix with optional pivoting."},
  {"schdd", "Downdates Cholesky factorization of real positive definite matrix."},
  {"schex", "Updates Cholesky factorization of real positive definite matrix."},
  {"schud", "Updates Cholesky factorization of real positive definite matrix."},
  {"sgbco", "Computes LU factorization of real band matrix and estimates its condition."},
  {"sgbdi", "Uses LU factorization of real band matrix to compute its determinant. (No provision for computing matrix inverse.)."},
  {"sgbfa", "Computes LU factorization of real band matrix."},
  {"sgbsl", "Uses LU factorization of real band matrix to solve systems."},
  {"sgeco", "Computes LU factorization of real general matrix and estimates its condition."},
  {"sgedi", "Uses LU factorization of real general matrix to compute its determinant and/or inverse."},
  {"sgefa", "Computes LU factorization of real general matrix."},
  {"sgesl", "Uses LU factorization of real general matrix to solve systems."},
  {"sgtsl", "Factors a real tridiagonal matrix and simultaneously solves a system."},
  {"spbco", "Uses Cholesky algorithm to compute factorization of real positive definite band matrix and estimates its condition."},
  {"spbdi", "Uses factorization of real positive definite band matrix to compute its determinant. (No provision for matrix inverse.)."},
  {"spbfa", "Uses Cholesky algorithm to compute factorization of real positive definite band matrix."},
  {"spbsl", "Uses factorization of real positive definite band matrix to solve systems."},
  {"spoco", "Uses Cholesky algorithm to factor real positive definite matrix and estimate its condition."},
  {"spodi", "Uses factorization of real positive definite matrix to compute its determinant and/or inverse."},
  {"spofa", "Uses Cholesky algorithm to factor real positive definite matrix."},
  {"sposl", "Uses factorization of real positive definite matrix to solve systems."},
  {"sppco", "Uses Cholesky algorithm to factor real positive definite matrix stored in packed form and estimate its condition."},
  {"sppdi", "Uses factorization of real positive definite matrix stored in packed form to compute its determinant and/or inverse."},
  {"sppfa", "Uses Cholesky algorithm to factor real positive definite matrix stored in packed form."},
  {"sppsl", "Uses factorization of real positive definite matrix stored in packed form to solve systems."},
  {"sptsl", "Decomposes real symmetric positive definite tridiagonal matrix and simultaneously solves a system."},
  {"sqrdc", "Computes QR decomposition of real general matrix."},
  {"sqrsl", "Applies the output of SQRDC to compute coordinate transformations, projections, and least squares solutions (general real matrix)."},
  {"ssico", "Computes factorization of real symmetric indefinite matrix and estimates its condition."},
  {"ssidi", "Uses factorization of real symmetric indefinite matrix to compute its determinant and/or inverse."},
  {"ssifa", "Computes factorization of real symmetric indefinite matrix."},
  {"ssisl", "Uses factorization of real symmetric indefinite matrix to solve systems."},
  {"sspco", "Computes factorization of real symmetric indefinite matrix stored in packed form and estimates its condition."},
  {"sspdi", "Uses factorization of real symmetric indefinite matrix stored in packed form to compute its determinant and/or inverse."},
  {"sspfa", "Computes factorization of real symmetric indefinite matrix stored in packed form."},
  {"sspsl", "Uses factorization of real symmetric indefinite matrix stored in packed form to solve systems."},
  {"ssvdc", "Computes the singular value decomposition of a real n-by-p matrix X, dimensioned X(LDX, P). Has options to allow computation of only the singular values, or singular values and associated decomposition matrices."},
  {"strco", "Estimates the condition of real triangular matrix."},
  {"strdi", "Computes determinant and/or inverse of real triangular matrix."},
  {"strsl", "Solves systems with real triangular matrix."},
  {"zchdc", "Compute Cholesky decomposition of complex positive definite matrix with optional pivoting."},
  {"zchdd", "Downdates Cholesky factorization of positive definite complex matrix."},
  {"zchex", "Updates Cholesky factorization of positive definite complex matrix."},
  {"zchud", "Updates Cholesky factorization of positive definite matrix."},
  {"zgbco", "Compute LU factorization of complex band matrix and estimate its condition."},
  {"zgbdi", "Compute determinant of complex band matrix from its LU factors. (No provision for computing inverse directly.)."},
  {"zgbfa", "Compute LU factorization of general complex band matrix."},
  {"zgbsl", "Uses LU factorization of complex band matrix to solve systems."},
  {"zgeco", "Compute LU factorization of general complex matrix and estimate its condition."},
  {"zgedi", "Compute determinant and/or inverse of general complex matrix from its LU factors."},
  {"zgefa", "Compute LU factorization of general complex matrix."},
  {"zgesl", "Use LU factorization of general complex matrix to solve systems."},
  {"zgtsl", "Solves systems with general complex tridiagonal matrix."},
  {"zhico", "Computes factorization of complex Hermitian indefinite matrix and estimates its condition."},
  {"zhidi", "Uses factorization of complex Hermitian indefinite matrix to compute its inertia, determinant, and/or inverse."},
  {"zhifa", "Computes factorization of complex Hermitian indefinite matrix."},
  {"zhisl", "Uses factorization of complex Hermitian indefinite matrix to solve systems."},
  {"zhpco", "Computes factorization of complex Hermitian indefinite matrix stored in packed form and estimates its condition."},
  {"zhpdi", "Uses factorization of complex Hermitian indefinite matrix stored in packed form to compute its inertia, determinant, and inverse."},
  {"zhpfa", "Computes factorization of complex Hermitian indefinite matrix stored in packed form."},
  {"zhpsl", "Uses factorization of complex Hermitian indefinite matrix stored in packed form to solve systems."},
  {"zpbco", "Uses Cholesky algorithm to compute factorization of complex positive definite band matrix and estimates its condition."},
  {"zpbdi", "Uses factorization of complex positive definite band matrix to compute determinant. (No provision for computing inverse.)."},
  {"zpbfa", "Uses Cholesky algorithm to compute factorization of complex positive definite band matrix."},
  {"zpbsl", "Uses factorization of complex positive definite band matrix to solve systems."},
  {"zpoco", "Uses Cholesky algorithm to compute factorization of complex positive definite matrix and estimates its condition."},
  {"zpodi", "Uses factorization of complex positive definite matrix to compute its determinant and/or inverse."},
  {"zpofa", "Uses Cholesky algorithm to compute factorization of complex positive definite matrix."},
  {"zposl", "Uses factorization of complex positive definite matrix to solve systems."},
  {"zppco", "Uses Cholesky algorithm to factor complex positive definite matrix stored in packed form."},
  {"zppdi", "Uses factorization of complex positive definite matrix stored in packed form to compute determinant and/or inverse."},
  {"zppfa", "Uses Cholesky algorithm to factor complex positive definite matrix stored in packed form."},
  {"zppsl", "Uses factorization of complex positive definite matrix stored in packed form to solve systems."},
  {"zptsl", "Solves systems with complex positive definite tridiagonal matrix."},
  {"zqrdc", "Computes QR decomposition of general complex matrix."},
  {"zqrsl", "Applies the output of ZQRDC to compute coordinate transformations, projections, and least squares solutions (general complex matrix)."},
  {"zsico", "Computes factorization of complex symmetric indefinite matrix and estimates its condition."},
  {"zsidi", "Uses factorization of complex symmetric indefinite matrix to compute its determinant and/or inverse."},
  {"zsifa", "Computes factorization of complex symmetric indefinite matrix."},
  {"zsisl", "Uses factorization of complex symmetric indefinite matrix to solve systems."},
  {"zspco", "Computes factorization of complex symmetric indefinite matrix stored in packed form and computes its condition."},
  {"zspdi", "Uses factorization of complex symmetric indefinite matrix stored in packed form to compute its determinant and/or inverse."},
  {"zspfa", "Computes factorization of complex symmetric indefinite matrix stored in packed form."},
  {"zspsl", "Uses factorization of complex symmetric indefinite matrix stored in packed form to solve systems."},
  {"zsvdc", "Computes singular value decomposition of general complex matrix."},
  {"ztrco", "Estimates condition of complex triangular matrix."},
  {"ztrdi", "Computes determinant and/or inverse of complex triangular matrix."},
  {"ztrsl", "Solves systems with complex triangular matrix."}
};

/// List of subroutines from EISPACK
map_ss sublib_EISPACK {
  {"bakvec", "Forms eigenvectors of certain real non-symmetric tridiagonal matrix from symmetric tridiagonal matrix output from FIGI."},
  {"balanc", "Balances a general real matrix and isolates eigenvalues whenever possible."},
  {"balbak", "Forms eigenvectors of real general matrix from eigenvectors of matrix output from BALANC."},
  {"bandr", "Reduces real symmetric band matrix to symmetric tridiagonal matrix and, optionally, accumulates orthogonal similarity transformations."},
  {"bandv", "Forms eigenvectors of real symmetric band matrix associated with a set of ordered approximate eigenvalues by inverse iteration."},
  {"bisect", "Compute eigenvalues of symmetric tridiagonal matrix in given interval using Sturm sequencing."},
  {"bqr", "Computes some of the eigenvalues of a real symmetric band matrix using the QR method with shifts of origin."},
  {"cbabk2", "Forms eigenvectors of complex general matrix from eigenvectors of matrix output from CBAL."},
  {"cbal", "Balances a complex general matrix and isolates eigenvalues whenever possible."},
  {"cg", "Computes the eigenvalues and, optionally, the eigenvectors of a complex general matrix."},
  {"ch", "Computes the eigenvalues and, optionally, eigenvectors of a complex Hermitian matrix."},
  {"cinvit", "Computes eigenvectors of a complex upper Hessenberg matrix associated with specified eigenvalues using inverse iteration."},
  {"combak", "Forms eigenvectors of complex general matrix from eigenvectors of upper Hessenberg matrix output from COMHES."},
  {"comhes", "Reduces complex general matrix to complex upper Hessenberg form using stabilized elementary similarity transformations."},
  {"comlr", "Computes eigenvalues of a complex upper Hessenberg matrix using the modified LR method."},
  {"comlr2", "Computes eigenvalues and eigenvectors of complex upper Hessenberg matrix using modified LR method."},
  {"comqr", "Computes eigenvalues of complex upper Hessenberg matrix using the QR method."},
  {"comqr2", "Computes eigenvalues and eigenvectors of complex upper Hessenberg matrix."},
  {"cortb", "Forms eigenvectors of complex general matrix from eigenvectors of upper Hessenberg matrix output from CORTH."},
  {"corth", "Reduces complex general matrix to complex upper Hessenberg using unitary similarity transformations."},
  {"elmbak", "Forms eigenvectors of real general matrix from eigenvectors of upper Hessenberg matrix output from ELMHES."},
  {"elmhes", "Reduces real general matrix to upper Hessenberg form using stabilized elementary similarity transformations."},
  {"eltran", "Accumulates the stabilized elementary similarity transformations used in the reduction of a real general matrix to upper Hessenberg form by ELMHES."},
  {"figi", "Transforms certain real non-symmetric tridiagonal matrix to symmetric tridiagonal matrix."},
  {"figi2", "Transforms certain real non-symmetric tridiagonal matrix to symmetric tridiagonal matrix."},
  {"hqr", "Computes eigenvalues of a real upper Hessenberg matrix using the QR method."},
  {"hqr2", "Computes eigenvalues and eigenvectors of real upper Hessenberg matrix using QR method."},
  {"htrib3", "Computes eigenvectors of complex Hermitian matrix from eigenvectors of real symmetric tridiagonal matrix output from HTRID3."},
  {"htribk", "Forms eigenvectors of complex Hermitian matrix from eigenvectors of real symmetric tridiagonal matrix output from HTRIDI."},
  {"htrid3", "Reduces complex Hermitian (packed) matrix to real symmetric tridiagonal matrix by unitary similarity transformations."},
  {"htridi", "Reduces complex Hermitian matrix to real symmetric tridiagonal matrix using unitary similarity transformations."},
  {"imtql1", "Computes eigenvalues of symmetric tridiagonal matrix using implicit QL method."},
  {"imtql2", "Computes eigenvalues and eigenvectors of symmetric tridiagonal matrix using implicit QL method."},
  {"imtqlv", "Computes eigenvalues of symmetric tridiagonal matrix using implicit QL method. Eigenvectors may be computed later."},
  {"invit", "Computes eigenvectors of upper Hessenberg (real) matrix associated with specified eigenvalues by inverse iteration."},
  {"minfit", "Compute singular value decomposition of rectangular real matrix and solve related linear least squares problem."},
  {"ortbak", "Forms eigenvectors of general real matrix from eigenvectors of upper Hessenberg matrix output from ORTHES."},
  {"orthes", "Reduces real general matrix to upper Hessenberg form using orthogonal similarity transformations."},
  {"ortran", "Accumulates orthogonal similarity transformations in reduction of real general matrix by ORTHES."},
  {"qzhes", "The first step of the QZ algorithm for solving generalized matrix eigenproblems. Accepts a pair of real general matrices and reduces one of them to upper Hessenberg form and the other to upper triangular form using orthogonal transformations. Usually followed by QZIT, QZVAL, QZVEC."},
  {"qzit", "The second step of the QZ algorithm for generalized eigenproblems. Accepts an upper Hessenberg and an upper triangular matrix and reduces the former to quasi-triangular form while preserving the form of the latter. Usually preceded by QZHES and followed by QZVAL and QZVEC."},
  {"qzval", "The third step of the QZ algorithm for generalized eigenproblems. Accepts a pair of real matrices, one in quasi-triangular form and the other in upper triangular form and computes the eigenvalues of the associated eigenproblem. Usually preceded by QZHES, QZIT, and followed by QZVEC."},
  {"qzvec", "The optional fourth step of the QZ algorithm for generalized eigenproblems. Accepts a matrix in quasi-triangular form and another in upper triangular form and computes the eigenvectors of the triangular problem and transforms them back to the original coordinates. Usually preceded by QZHES, QZIT, QZVAL."},
  {"ratqr", "Computes largest or smallest eigenvalues of symmetric tridiagonal matrix using rational QR method with Newton correction."},
  {"rebak", "Forms eigenvectors of generalized symmetric eigensystem from eigenvectors of derived matrix output from REDUC or REDUC2."},
  {"rebakb", "Forms eigenvectors of generalized symmetric eigensystem from eigenvectors of derived matrix output from REDUC2."},
  {"reduc", "Reduces generalized symmetric eigenproblem Ax=(lambda)Bx, to standard symmetric eigenproblem, using Cholesky factorization."},
  {"reduc2", "Reduces certain generalized symmetric eigenproblems to standard symmetric eigenproblem, using Cholesky factorization."},
  {"rg", "Computes eigenvalues and, optionally, eigenvectors of a real general matrix."},
  {"rgg", "Computes eigenvalues and eigenvectors for real generalized eigenproblem: Ax=(lambda)Bx."},
  {"rs", "Computes eigenvalues and, optionally, eigenvectors of a real symmetric matrix."},
  {"rsb", "Computes eigenvalues and, optionally, eigenvectors of real symmetric band matrix."},
  {"rsg", "Computes eigenvalues and, optionally, eigenvectors of real symmetric generalized eigenproblem: Ax=(lambda)Bx."},
  {"rsgab", "Computes eigenvalues and, optionally, eigenvectors of real symmetric generalized eigenproblem: ABx=(lambda)x."},
  {"rsgba", "Computes eigenvalues and, optionally, eigenvectors of real symmetric generalized eigenproblem: BAx=(lambda)x."},
  {"rsp", "Compute eigenvalues and, optionally, eigenvectors of a real symmetric matrix packed into a one dimensional array."},
  {"rst", "Compute eigenvalues and, optionally, eigenvectors of a real symmetric tridiagonal matrix."},
  {"rt", "Compute eigenvalues and eigenvectors of a special real tridiagonal matrix."},
  {"svd", "Computes singular value decomposition of arbitrary real rectangular matrix."},
  {"tinvit", "Eigenvectors of symmetric tridiagonal matrix corresponding to some specified eigenvalues, using inverse iteration."},
  {"tql1", "Compute eigenvalues of symmetric tridiagonal matrix by QL method."},
  {"tql2", "Compute eigenvalues and eigenvectors of symmetric tridiagonal matrix."},
  {"tqlrat", "Computes eigenvalues of symmetric tridiagonal matrix using a rational variant of the QL method."},
  {"trbak1", "Forms the eigenvectors of real symmetric matrix from eigenvectors of symmetric tridiagonal matrix formed by TRED1."},
  {"trbak3", "Forms eigenvectors of real symmetric matrix from the eigenvectors of symmetric tridiagonal matrix formed by TRED3."},
  {"tred1", "Reduce real symmetric matrix to symmetric tridiagonal matrix using orthogonal similarity transformations."},
  {"tred2", "Reduce real symmetric matrix to symmetric tridiagonal matrix using and accumulating orthogonal transformations."},
  {"tred3", "Reduce real symmetric matrix stored in packed form to symmetric tridiagonal matrix using orthogonal transformations."},
  {"tridib", "Computes eigenvalues of symmetric tridiagonal matrix in given interval using Sturm sequencing."},
  {"tsturm", "Computes eigenvalues of symmetric tridiagonal matrix in given interval and eigenvectors by Sturm sequencing."}
};

/// List of subroutines from LAPACK
map_ss sublib_LAPACK {
  {"cbdsqr", "Computes the singular value decomposition (SVD) of a real bidiagonal matrix, using the bidiagonal QR algorithm. (Computational routine.)."},
  {"cgbbrd", "Reduces a complex general m-by-n band matrix A to real upper bidiagonal form B by a unitary transformation: Q'' * A * P = B. The routine computes B and optionally forms Q or P'', or computes Q'' * C for a given matrix C."},
  {"cgbcon", "Estimates the reciprocal of the condition number of a general band matrix, in either the 1-norm or the infinity-norm, using the LU factorization computed by CGBTRF. (Computational routine.)."},
  {"cgbequ", "Computes row and column scalings to equilibrate a complex general band matrix and reduce its condition number. (Computational routine.)."},
  {"cgbrfs", "Improves the computed solution to a complex general banded system of linear equations AX=B, A**T X=B or A**H X=B, and provides forward and backward error bounds for the solution. (Computational routine.)."},
  {"cgbsv", "Solves a complex general banded system of linear equations AX=B. (Simple driver.)."},
  {"cgbsvx", "Solves a complex general banded system of linear equations AX=B, A**T X=B or A**H X=B, and provides an estimate of the condition number and error bounds on the solution. (Expert driver.)."},
  {"cgbtrf", "Computes an LU factorization of a complex general band matrix, using partial pivoting with row interchanges. (Computational routine.)."},
  {"cgbtrs", "Solves a complex general banded system of linear equations AX=B, A**T X=B or A**H X=B, using the LU factorization computed by CGBTRF. (Computational routine.)."},
  {"cgebak", "Transforms eigenvectors of a balanced matrix to those of the original matrix supplied to CGEBAL. (Computational routine.)."},
  {"cgebal", "Balances a complex general matrix in order to improve the accuracy of computed eigenvalues. (Computational routine.)."},
  {"cgebrd", "Reduces a complex general rectangular matrix to real bidiagonal form by an unitary transformation. (Computational routine.)."},
  {"cgecon", "Estimates the reciprocal of the condition number of a general matrix, in either the 1-norm or the infinity-norm, using the LU factorization computed by CGETRF. (Computational routine.)."},
  {"cgeequ", "Computes row and column scalings to equilibrate a general rectangular matrix and reduce its condition number. (Computational routine.)."},
  {"cgees", "Computes the eigenvalues and Schur factorization of a general matrix, and orders the factorization so that selected eigenvalues are at the top left of the Schur form. (Simple driver.)."},
  {"cgeesx", "Computes the eigenvalues and Schur factorization of a general matrix, orders the factorization so that selected eigenvalues are at the top left of the Schur form, and computes reciprocal condition numbers for the average of the selected eigenvalues, and for the associated right invariant subspace. (Expert driver.)."},
  {"cgeev", "Computes the eigenvalues and left and right eigenvectors of a complex general matrix. (Simple driver.)."},
  {"cgeevx", "Computes the eigenvalues and left and right eigenvectors of a complex general matrix, with preliminary balancing of the matrix, and computes reciprocal condition numbers for the eigenvalues and right eigenvectors. (Expert driver.)."},
  {"cgehrd", "Reduces a complex general matrix to upper Hessenberg form by an unitary similarity transformation. (Computational routine.)."},
  {"cgelqf", "Computes an LQ factorization of a complex general rectangular matrix. (Computational routine.)."},
  {"cgels", "Computes the least squares solution to an over-determined system of linear equations, A X=B or A**H X=B, or the minimum norm solution of an under-determined system, where A is a general rectangular matrix of full rank, using a QR or LQ factorization of A. (Simple driver.)."},
  {"cgelss", "Computes the minimum norm least squares solution to an over- or under-determined system of linear equations A X=B, using the singular value decomposition of A. (Simple driver.)."},
  {"cgelsx", "Computes the minimum norm least squares solution to an over- or under-determined system of linear equations A X=B, using a complete orthogonal factorization of A. (Expert driver.)."},
  {"cgeqlf", "Computes a QL factorization of a complex general rectangular matrix. (Computational routine.)."},
  {"cgeqpf", "Computes a QR factorization with column pivoting of a general rectangular matrix. (Computational routine.)."},
  {"cgeqrf", "Computes a QR factorization of a complex general rectangular matrix. (Computational routine.)."},
  {"cgerfs", "Improves the computed solution to a complex general system of linear equations AX=B, A**T X=B or A**H X=B, and provides forward and backward error bounds for the solution. (Computational routine.)."},
  {"cgerqf", "Computes an RQ factorization of a complex general rectangular matrix. (Computational routine.)."},
  {"cgesv", "Solves a complex general system of linear equations AX=B. (Simple driver.)."},
  {"cgesvd", "Computes the singular value decomposition (SVD) of a general rectangular matrix. (Simple driver.)."},
  {"cgesvx", "Solves a complex general system of linear equations AX=B, A**T X=B or A**H X=B, and provides an estimate of the condition number and error bounds on the solution. (Expert driver.)."},
  {"cgetrf", "Computes an LU factorization of a complex general matrix, using partial pivoting with row interchanges. (Computational routine.)."},
  {"cgetri", "Computes the inverse of a complex general matrix, using the LU factorization computed by CGETRF. (Computational routine.)."},
  {"cgetrs", "Solves a complex general system of linear equations AX=B, A**T X=B or A**H X=B, using the LU factorization computed by CGETRF. (Computational routine.)."},
  {"cgtcon", "Estimates the reciprocal of the condition number of a general tridiagonal matrix, in either the 1-norm or the infinity-norm, using the LU factorization computed by CGTTRF. (Computational routine.)."},
  {"cgtrfs", "Improves the computed solution to a complex general tridiagonal system of linear equations AX=B, A**T X=B or A**H X=B, and provides forward and backward error bounds for the solution. (Computational routine.)."},
  {"cgtsv", "Solves a complex general tridiagonal system of linear equations AX=B. (Simple driver.)."},
  {"cgtsvx", "Solves a complex general tridiagonal system of linear equations AX=B, A**T X=B or A**H X=B, and provides an estimate of the condition number and error bounds on the solution. (Expert driver.)."},
  {"cgttrf", "Computes an LU factorization of a complex general tridiagonal matrix, using partial pivoting with row interchanges. (Computational routine.)."},
  {"cgttrs", "Solves a complex general tridiagonal system of linear equations AX=B, A**T X=B or A**H X=B, using the LU factorization computed by CGTTRF. (Computational routine.)."},
  {"chbev", "Computes all eigenvalues and eigenvectors of a complex Hermitian band matrix.  (Simple driver.)"},
  {"chbevd", "Computes all eigenvalues and, optionally, eigenvectors of a complex Hermitian band matrix."},
  {"chbevx", "Computes selected eigenvalues and eigenvectors of a complex Hermitian band matrix.  (Expert driver.)"},
  {"chbgst", "Reduces a complex Hermitian definite banded generalized eigenproblem A * x = lambda * B * x to standard form C * y = lambda * y, such that C has the same bandwidth as A. B must have been previously factorized by CPBSTF."},
  {"chbgv", "Computes all the eigenvalues and, optionally, the eigenvectors, of a complex generalized Hermitian definite banded eigenproblem, of the form A * x = lambda * B * x. A and B are assumed to be Hermitian and banded, and B is also positive definite."},
  {"chbtrd", "Reduces a complex Hermitian band matrix to real symmetric tridiagonal form by an unitary similarity transformation.  (Computational routine.)"},
  {"checon", "Estimates the reciprocal of the condition number of a real symmetric/complex symmetric/complex Hermitian indefinite matrix, using the factorization computed by CHETRF. (Computational routine.)."},
  {"cheev", "Computes all eigenvalues and eigenvectors of a complex Hermitian matrix. (Simple driver.)"},
  {"cheevd", "Computes all eigenvalues and, optionally, eigenvectors of a complex Hermitian matrix."},
  {"cheevx", "Computes selected eigenvalues and eigenvectors of a complex Hermitian matrix.  (Expert driver.)"},
  {"chegst", "Reduces a complex Hermitian-definite generalized eigenproblem Ax= lambda Bx, ABx= lambda x, or BAx= lambda x, to standard form, where B has been factorized by CPOTRF.  (Computational routine.)"},
  {"chegv", "Computes all eigenvalues and the eigenvectors of a generalized complex Hermitian-definite generalized eigenproblem, Ax= lambda Bx, ABx= lambda x, or BAx= lambda x.  (Simple driver.)"},
  {"cherfs", "Improves the computed solution to a real symmetric/complex symmetric/complex Hermitian indefinite system of linear equations AX=B, and provides forward and backward error bounds for the solution. (Computational routine.)."},
  {"chesv", "Solves a complex Hermitian indefinite system of linear equations AX=B. (Simple driver.)."},
  {"chesvx", "Solves a complex Hermitian indefinite system of linear equations AX=B, and provides an estimate of the condition number and error bounds on the solution. (Expert driver.)."},
  {"chetrd", "Reduces a complex Hermitian matrix to real symmetric tridiagonal form by an unitary similarity transformation.  (Computational routine.)"},
  {"chetrf", "Computes the factorization of a real symmetric/complex symmetric/complex Hermitian-indefinite matrix, using the diagonal pivoting method. (Computational routine.)."},
  {"chetri", "Computes the inverse of a real symmetric/complex symmetric/complex Hermitian indefinite matrix, using the factorization computed by CHETRF. (Computational routine.)."},
  {"chetrs", "Solves a complex Hermitian indefinite system of linear equations AX=B, using the factorization computed by CHPTRF. (Computational routine.)."},
  {"chpcon", "Estimates the reciprocal of the condition number of a real symmetric/complex symmetric/complex Hermitian indefinite matrix in packed storage, using the factorization computed by CHPTRF. (Computational routine.)."},
  {"chpev", "Computes all eigenvalues and eigenvectors of a complex Hermitian matrix in packed storage.  (Simple driver.)"},
  {"chpevd", "Computes all eigenvalues and, optionally, eigenvectors of a complex Hermitian matrix in packed storage."},
  {"chpevx", "Computes selected eigenvalues and eigenvectors of a complex Hermitian matrix in packed storage.  (Expert driver.)"},
  {"chpgst", "Reduces a complex Hermitian-definite generalized eigenproblem Ax= lambda Bx, ABx= lambda x, or BAx= lambda x, to standard form, where A and B are held in packed storage, and B has been factorized by CPPTRF. (Computational routine.)"},
  {"chpgv", "Computes all eigenvalues and eigenvectors of a generalized complex Hermitian-definite generalized eigenproblem, Ax= lambda Bx, ABx= lambda x, or BAx= lambda x, where A and B are in packed storage.  (Simple driver.)"},
  {"chprfs", "Improves the computed solution to a real symmetric/complex symmetric/complex Hermitian indefinite system of linear equations AX=B, where A is held in packed storage, and provides forward and backward error bounds for the solution. (Computational routine.)."},
  {"chpsv", "Solves a complex Hermitian indefinite system of linear equations AX=B, where A is held in packed storage. (Simple driver.)."},
  {"chpsvx", "Solves a complex Hermitian indefinite system of linear equations AX=B, where A is held in packed storage, and provides an estimate of the condition number and error bounds on the solution. (Expert driver.)."},
  {"chptrd", "Reduces a complex Hermitian matrix in packed storage to real symmetric tridiagonal form by an unitary similarity transformation.  (Computational routine.)"},
  {"chptrf", "Computes the factorization of a real symmetric/complex symmetric/complex Hermitian-indefinite matrix in packed storage, using the diagonal pivoting method. (Computational routine.)."},
  {"chptri", "Computes the inverse of a real symmetric/complex symmetric/complex Hermitian indefinite matrix in packed storage, using the factorization computed by CHPTRF. (Computational routine.)."},
  {"chptrs", "Solves a complex Hermitian indefinite system of linear equations AX=B, where A is held in packed storage, using the factorization computed by CHPTRF. (Computational routine.)."},
  {"chsein", "Computes specified right and/or left eigenvectors of an upper Hessenberg matrix by inverse iteration. (Computational routine.)."},
  {"chseqr", "Computes the eigenvalues and Schur factorization of an upper Hessenberg matrix, using the multishift QR algorithm. (Computational routine.)."},
  {"cpbcon", "Estimates the reciprocal of the condition number of a complex Hermitian positive definite band matrix, using the Cholesky factorization computed by CPBTRF. (Computational routine.)."},
  {"cpbequ", "Computes row and column scalings to equilibrate a complex Hermitian positive definite band matrix and reduce its condition number. (Computational routine.)."},
  {"cpbrfs", "Improves the computed solution to a complex Hermitian positive definite banded system of linear equations AX=B, and provides forward and backward error bounds for the solution. (Computational routine.)."},
  {"cpbstf", "Computes a split Cholesky factorization of a complex Hermitian positive definite band matrix. This routine is designed to be used in conjunction with CHBGST."},
  {"cpbsv", "Solves a complex Hermitian positive definite banded system of linear equations AX=B. (Simple driver.)."},
  {"cpbsvx", "Solves a complex Hermitian positive definite banded system of linear equations AX=B, and provides an estimate of the condition number and error bounds on the solution. (Expert driver.)."},
  {"cpbtrf", "Computes the Cholesky factorization of a complex Hermitian positive definite band matrix. (Computational routine.)."},
  {"cpbtrs", "Solves a complex Hermitian positive definite banded system of linear equations AX=B, using the Cholesky factorization computed by CPBTRF. (Computational routine.)."},
  {"cpocon", "Estimates the reciprocal of the condition number of a complex Hermitian positive definite matrix, using the Cholesky factorization computed by SPOTRF/CPOTRF. by SPOTRF/CPOTRF. (Computational routine.)."},
  {"cpoequ", "Computes row and column scalings to equilibrate a complex Hermitian positive definite matrix and reduce its condition number. (Computational routine.)."},
  {"cporfs", "Improves the computed solution to a complex Hermitian positive definite system of linear equations AX=B, and provides forward and backward error bounds for the solution. (Computational routine.)."},
  {"cposv", "Solves a complex Hermitian positive definite system of linear equations AX=B. (Simple driver.)."},
  {"cposvx", "Solves a complex Hermitian positive definite system of linear equations AX=B, and provides an estimate of the condition number and error bounds on the solution. (Expert driver.)."},
  {"cpotrf", "Computes the Cholesky factorization of a complex Hermitian positive definite matrix. (Computational routine.)."},
  {"cpotri", "Computes the inverse of a complex Hermitian positive definite matrix, using the Cholesky factorization computed by CPOTRF. (Computational routine.)."},
  {"cpotrs", "Solves a complex Hermitian positive definite system of linear equations AX=B, using the Cholesky factorization computed by CPOTRF. (Computational routine.)."},
  {"cppcon", "Estimates the reciprocal of the condition number of a complex Hermitian positive definite matrix in packed storage, using the Cholesky factorization computed by CPPTRF. (Computational routine.)."},
  {"cppequ", "Computes row and column scalings to equilibrate a complex Hermitian positive definite matrix in packed storage and reduce its condition number. (Computational routine.)."},
  {"cpprfs", "Improves the computed solution to a complex Hermitian positive definite system of linear equations AX=B, where A is held in packed storage, and provides forward and backward error bounds for the solution. (Computational routine.)."},
  {"cppsv", "Solves a complex Hermitian positive definite system of linear equations AX=B, where A is held in packed storage. (Simple driver.)."},
  {"cppsvx", "Solves a complex Hermitian positive definite system of linear equations AX=B, where A is held in packed storage, and provides an estimate of the condition number and error bounds on the solution. (Expert driver.)."},
  {"cpptrf", "Computes the Cholesky factorization of a complex Hermitian positive definite matrix in packed storage. (Computational routine.)."},
  {"cpptri", "Computes the inverse of a complex Hermitian positive definite matrix in packed storage, using the Cholesky factorization computed by CPPTRF. (Computational routine.)."},
  {"cpptrs", "Solves a complex Hermitian positive definite system of linear equations AX=B, where A is held in packed storage, using the Cholesky factorization computed by CPPTRF. (Computational routine.)."},
  {"cptcon", "Computes the reciprocal of the condition number of a complex Hermitian positive definite tridiagonal matrix, using the LDL**H factorization computed by CPTTRF. (Computational routine.)."},
  {"cpteqr", "Computes all eigenvalues and eigenvectors of a real symmetric positive definite tridiagonal matrix, by computing the SVD of its bidiagonal Cholesky factor. (Computational routine.)."},
  {"cptrfs", "Improves the computed solution to a complex Hermitian positive definite tridiagonal system of linear equations AX=B, and provides forward and backward error bounds for the solution. (Computational routine.)."},
  {"cptsv", "Solves a complex Hermitian positive definite tridiagonal system of linear equations AX=B. (Simple driver.)."},
  {"cptsvx", "Solves a complex Hermitian positive definite tridiagonal system of linear equations AX=B, and provides an estimate of the condition number and error bounds on the solution. (Expert driver.)."},
  {"cpttrf", "Computes the LDL**H factorization of a complex Hermitian positive definite tridiagonal matrix. (Computational routine.)."},
  {"cpttrs", "Solves a complex Hermitian positive definite tridiagonal system of linear equations, using the LDL**H factorization computed by CPTTRF. (Computational routine.)."},
  {"cspcon", "Estimates the reciprocal of the condition number of a real symmetric/complex symmetric/complex Hermitian indefinite matrix in packed storage, using the factorization computed by CSPTRF. (Computational routine.)."},
  {"csprfs", "Improves the computed solution to a real symmetric/complex symmetric/complex Hermitian indefinite system of linear equations AX=B, where A is held in packed storage, and provides forward and backward error bounds for the solution. (Computational routine.)."},
  {"cspsv", "Solves a complex symmetric indefinite system of linear equations AX=B, where A is held in packed storage. (Simple driver.)."},
  {"cspsvx", "Solves a complex symmetric indefinite system of linear equations AX=B, where A is held in packed storage, and provides an estimate of the condition number and error bounds on the solution. (Expert driver.)."},
  {"csptrf", "Computes the factorization of a real symmetric/complex symmetric/complex Hermitian-indefinite matrix in packed storage, using the diagonal pivoting method. (Computational routine.)."},
  {"csptri", "Computes the inverse of a real symmetric/complex symmetric/complex Hermitian indefinite matrix in packed storage, using the factorization computed by CSPTRF. (Computational routine.)."},
  {"csptrs", "Solves a complex symmetric indefinite system of linear equations AX=B, where A is held in packed storage, using the factorization computed by CSPTRF. (Computational routine.)."},
  {"cstein", "Computes selected eigenvectors of a real symmetric tridiagonal matrix by inverse iteration. (Computational routine.)."},
  {"csteqr", "Computes all eigenvalues and eigenvectors of a real symmetric tridiagonal matrix, using the implicit QL or QR algorithm. (Computational routine.)."},
  {"csycon", "Estimates the reciprocal of the condition number of a real symmetric/complex symmetric/complex Hermitian indefinite matrix, using the factorization computed by CSYTRF. (Computational routine.)."},
  {"csyrfs", "Improves the computed solution to a real symmetric/complex symmetric/complex Hermitian indefinite system of linear equations AX=B, and provides forward and backward error bounds for the solution. (Computational routine.)."},
  {"csysv", "Solves a complex symmetric indefinite system of linear equations AX=B. (Simple driver.)."},
  {"csysvx", "Solves a complex symmetric indefinite system of linear equations AX=B, and provides an estimate of the condition number and error bounds on the solution. (Expert driver.)."},
  {"csytrf", "Computes the factorization of a real symmetric/complex symmetric/complex Hermitian-indefinite matrix, using the diagonal pivoting method. (Computational routine.)."},
  {"csytri", "Computes the inverse of a real symmetric/complex symmetric/complex Hermitian indefinite matrix, using the factorization computed by CSYTRF. (Computational routine.)."},
  {"csytrs", "Solves a complex symmetric indefinite system of linear equations AX=B, using the factorization computed by CSPTRF. (Computational routine.)."},
  {"ctbcon", "Estimates the reciprocal of the condition number of a complex triangular band matrix, in either the 1-norm or the infinity-norm. (Computational routine.)."},
  {"ctbrfs", "Provides forward and backward error bounds for the solution of a complex triangular banded system of linear equations AX=B, A**T X=B or A**H X=B. (Computational routine.)."},
  {"ctbtrs", "Solves a complex triangular banded system of linear equations AX=B, A**T X=B or A**H X=B. (Computational routine.)."},
  {"ctpcon", "Estimates the reciprocal of the condition number of a complex triangular matrix in packed storage, in either the 1-norm or the infinity-norm. (Computational routine.)."},
  {"ctprfs", "Provides forward and backward error bounds for the solution of a complex triangular system of linear equations AX=B, A**T X=B or A**H X=B, where A is held in packed storage. (Computational routine.)."},
  {"ctptri", "Computes the inverse of a complex triangular matrix in packed storage. (Computational routine.)."},
  {"ctptrs", "Solves a complex triangular system of linear equations AX=B, A**T X=B or A**H X=B, where A is held in packed storage. (Computational routine.)."},
  {"ctrcon", "Estimates the reciprocal of the condition number of a complex triangular matrix, in either the 1-norm or the infinity-norm. (Computational routine.)."},
  {"ctrevc", "Computes left and right eigenvectors of a complex upper triangular matrix. (Computational routine.)."},
  {"ctrexc", "Reorders the Schur factorization of a complex matrix by a unitary similarity transformation. (Computational routine.)."},
  {"ctrrfs", "Provides forward and backward error bounds for the solution of a complex triangular system of linear equations A X=B, A**T X=B or A**H X=B. (Computational routine.)."},
  {"ctrsen", "Reorders the Schur factorization of a complex matrix in order to find an orthonormal basis of a right invariant subspace corresponding to selected eigenvalues, and returns reciprocal condition numbers (sensitivities) of the average of the cluster of eigenvalues and of the invariant subspace. (Computational routine.)."},
  {"ctrsna", "Estimates the reciprocal condition numbers (sensitivities) of selected eigenvalues and eigenvectors of a complex upper triangular matrix. (Computational routine.)."},
  {"ctrsyl", "Solves the a complex Sylvester matrix equation A X +/- X B=C where A and B are upper triangular, and may be transposed. (Computational routine.)."},
  {"ctrtri", "Computes the inverse of a complex triangular matrix. (Computational routine.)."},
  {"ctrtrs", "Solves a complex triangular system of linear equations AX=B, A**T X=B or A**H X=B. (Computational routine.)."},
  {"ctzrqf", "Computes an RQ factorization of a complex upper trapezoidal matrix. (Computational routine.)."},
  {"cungbr", "Generates the unitary transformation matrices from a reduction to bidiagonal form determined by CGEBRD. (Computational routine.)."},
  {"cunghr", "Generates the unitary transformation matrix from a reduction to Hessenberg form determined by CGEHRD. (Computational routine.)."},
  {"cunglq", "Generates all or part of the unitary matrix Q from an LQ factorization determined by CGELQF. (Computational routine.)."},
  {"cungql", "Generates all or part of the unitary matrix Q from a QL factorization determined by CGEQLF. (Computational routine.)."},
  {"cungqr", "Generates all or part of the unitary matrix Q from a QR factorization determined by CGEQRF. (Computational routine.)."},
  {"cungrq", "Generates all or part of the unitary matrix Q from an RQ factorization determined by CGERQF. (Computational routine.)."},
  {"cungtr", "Generates the unitary transformation matrix from a reduction to tridiagonal form determined by CHETRD. (Computational routine.)."},
  {"cunmbr", "Multiplies a complex general matrix by one of the unitary transformation matrices from a reduction to bidiagonal form determined by CGEBRD. (Computational routine.)."},
  {"cunmhr", "Multiplies a complex general matrix by the unitary transformation matrix from a reduction to Hessenberg form determined by CGEHRD. (Computational routine.)."},
  {"cunmlq", "Multiplies a complex general matrix by the unitary matrix from an LQ factorization determined by CGELQF. (Computational routine.)."},
  {"cunmql", "Multiplies a complex general matrix by the unitary matrix from a QL factorization determined by CGEQLF. (Computational routine.)."},
  {"cunmqr", "Multiplies a complex general matrix by the unitary matrix from a QR factorization determined by CGEQRF. (Computational routine.)."},
  {"cunmrq", "Multiplies a complex general matrix by the unitary matrix from an RQ factorization determined by CGERQF. (Computational routine.)."},
  {"cunmtr", "Multiplies a complex general matrix by the unitary transformation matrix from a reduction to tridiagonal form determined by CHETRD. (Computational routine.)."},
  {"cupgtr", "Generates the unitary transformation matrix from a reduction to tridiagonal form determined by CHPTRD. (Computational routine.)."},
  {"cupmtr", "Multiplies a complex general matrix by the unitary transformation matrix from a reduction to tridiagonal form determined by CHPTRD. (Computational routine.)."},
  {"dbdsqr", "Computes the singular value decomposition (SVD) of a real bidiagonal matrix, using the bidiagonal QR algorithm. (Computational routine.)."},
  {"dgbbrd", "Reduces a real general m-by-n band matrix A to upper bidiagonal form B by an orthogonal transformation: Q'' * A * P = B. The routine computes B and optionally forms Q or P'', or computes Q'' * C for a given matrix C."},
  {"dgbcon", "Estimates the reciprocal of the condition number of a general band matrix, in either the 1-norm or the infinity-norm, using the LU factorization computed by DGBTRF. (Computational routine.)."},
  {"dgbequ", "Computes row and column scalings to equilibrate a real general band matrix and reduce its condition number. (Computational routine.)."},
  {"dgbrfs", "Improves the computed solution to a real general banded system of linear equations AX=B, A**T X=B or A**H X=B, and provides forward and backward error bounds for the solution. (Computational routine.)."},
  {"dgbsv", "Solves a real general banded system of linear equations AX=B. (Simple driver.)."},
  {"dgbsvx", "Solves a real general banded system of linear equations AX=B, A**T X=B or A**H X=B, and provides an estimate of the condition number and error bounds on the solution. (Expert driver.)."},
  {"dgbtrf", "Computes an LU factorization of a real general band matrix, using partial pivoting with row interchanges. (Computational routine.)."},
  {"dgbtrs", "Solves a real general banded system of linear equations AX=B, A**T X=B or A**H X=B, using the LU factorization computed by DGBTRF. (Computational routine.)."},
  {"dgebak", "Transforms eigenvectors of a balanced matrix to those of the original matrix supplied to DGEBAL. (Computational routine.)."},
  {"dgebal", "Balances a real general matrix in order to improve the accuracy of computed eigenvalues. (Computational routine.)."},
  {"dgebrd", "Reduces a real general rectangular matrix to real bidiagonal form by an orthogonal transformation. (Computational routine.)."},
  {"dgecon", "Estimates the reciprocal of the condition number of a general matrix, in either the 1-norm or the infinity-norm, using the LU factorization computed by DGETRF. (Computational routine.)."},
  {"dgeequ", "Computes row and column scalings to equilibrate a general rectangular matrix and reduce its condition number. (Computational routine.)."},
  {"dgees", "Computes the eigenvalues and Schur factorization of a general matrix, and orders the factorization so that selected eigenvalues are at the top left of the Schur form. (Simple driver.)."},
  {"dgeesx", "Computes the eigenvalues and Schur factorization of a general matrix, orders the factorization so that selected eigenvalues are at the top left of the Schur form, and computes reciprocal condition numbers for the average of the selected eigenvalues, and for the associated right invariant subspace. (Expert driver.)."},
  {"dgeev", "Computes the eigenvalues and left and right eigenvectors of a real general matrix. (Simple driver.)."},
  {"dgeevx", "Computes the eigenvalues and left and right eigenvectors of a real general matrix, with preliminary balancing of the matrix, and computes reciprocal condition numbers for the eigenvalues and right eigenvectors. (Expert driver.)."},
  {"dgehrd", "Reduces a real general matrix to upper Hessenberg form by an orthogonal similarity transformation. (Computational routine.)."},
  {"dgelqf", "Computes an LQ factorization of a real general rectangular matrix. (Computational routine.)."},
  {"dgels", "Computes the least squares solution to an over-determined system of linear equations, A X=B or A**H X=B, or the minimum norm solution of an under-determined system, where A is a general rectangular matrix of full rank, using a QR or LQ factorization of A. (Simple driver.)."},
  {"dgelss", "Computes the minimum norm least squares solution to an over- or under-determined system of linear equations A X=B, using the singular value decomposition of A. (Simple driver.)."},
  {"dgelsx", "Computes the minimum norm least squares solution to an over- or under-determined system of linear equations A X=B, using a complete orthogonal factorization of A. (Expert driver.)."},
  {"dgeqlf", "Computes a QL factorization of a real general rectangular matrix. (Computational routine.)."},
  {"dgeqpf", "Computes a QR factorization with column pivoting of a general rectangular matrix. (Computational routine.)."},
  {"dgeqrf", "Computes a QR factorization of a real general rectangular matrix. (Computational routine.)."},
  {"dgerfs", "Improves the computed solution to a real general system of linear equations AX=B, A**T X=B or A**H X=B, and provides forward and backward error bounds for the solution. (Computational routine.)."},
  {"dgerqf", "Computes an RQ factorization of a real general rectangular matrix. (Computational routine.)."},
  {"dgesv", "Solves a real general system of linear equations AX=B. (Simple driver.)."},
  {"dgesvd", "Computes the singular value decomposition (SVD) of a general rectangular matrix. (Simple driver.)."},
  {"dgesvx", "Solves a real general system of linear equations AX=B, A**T X=B or A**H X=B, and provides an estimate of the condition number and error bounds on the solution. (Expert driver.)."},
  {"dgetrf", "Computes an LU factorization of a real general matrix, using partial pivoting with row interchanges. (Computational routine.)."},
  {"dgetri", "Computes the inverse of a real general matrix, using the LU factorization computed by DGETRF. (Computational routine.)."},
  {"dgetrs", "Solves a real general system of linear equations AX=B, A**T X=B or A**H X=B, using the LU factorization computed by DGETRF. (Computational routine.)."},
  {"dgtcon", "Estimates the reciprocal of the condition number of a general tridiagonal matrix, in either the 1-norm or the infinity-norm, using the LU factorization computed by DGTTRF. (Computational routine.)."},
  {"dgtrfs", "Improves the computed solution to a real general tridiagonal system of linear equations AX=B, A**T X=B or A**H X=B, and provides forward and backward error bounds for the solution. (Computational routine.)."},
  {"dgtsv", "Solves a real general tridiagonal system of linear equations AX=B. (Simple driver.)."},
  {"dgtsvx", "Solves a real general tridiagonal system of linear equations AX=B, A**T X=B or A**H X=B, and provides an estimate of the condition number and error bounds on the solution. (Expert driver.)."},
  {"dgttrf", "Computes an LU factorization of a real general tridiagonal matrix, using partial pivoting with row interchanges. (Computational routine.)."},
  {"dgttrs", "Solves a real general tridiagonal system of linear equations AX=B, A**T X=B or A**H X=B, using the LU factorization computed by DGTTRF. (Computational routine.)."},
  {"dhsein", "Computes specified right and/or left eigenvectors of a real upper Hessenberg matrix by inverse iteration. (Computational routine.)."},
  {"dhseqr", "Computes the eigenvalues and Schur factorization of a real upper Hessenberg matrix, using the multishift QR algorithm. (Computational routine.)."},
  {"dopgtr", "Generates the orthogonal transformation matrix from a reduction to tridiagonal form determined by SSPTRD. (Computational routine.)."},
  {"dopmtr", "Multiplies a real general matrix by the orthogonal transformation matrix from a reduction to tridiagonal form determined by DSPTRD. (Computational routine.)."},
  {"dorgbr", "Generates the orthogonal transformation matrices from a reduction to bidiagonal form determined by SGEBRD. (Computational routine.)."},
  {"dorghr", "Generates the orthogonal transformation matrix from a reduction to Hessenberg form determined by SGEHRD. (Computational routine.)."},
  {"dorglq", "Generates all or part of the orthogonal matrix Q from an LQ factorization determined by DGELQF. (Computational routine.)."},
  {"dorgql", "Generates all or part of the orthogonal matrix Q from a QL factorization determined by DGEQLF. (Computational routine.)."},
  {"dorgqr", "Generates all or part of the orthogonal matrix Q from a QR factorization determined by DGEQRF. (Computational routine.)."},
  {"dorgrq", "Generates all or part of the orthogonal matrix Q from an RQ factorization determined by DGERQF. (Computational routine.)."},
  {"dorgtr", "Generates the orthogonal transformation matrix from a reduction to tridiagonal form determined by SSYTRD. (Computational routine.)."},
  {"dormbr", "Multiplies a real general matrix by one of the orthogonal transformation matrices from a reduction to bidiagonal form determined by SGEBRD. (Computational routine.)."},
  {"dormhr", "Multiplies a real general matrix by the orthogonal transformation matrix from a reduction to Hessenberg form determined by DGEHRD. (Computational routine.)."},
  {"dormlq", "Multiplies a real general matrix by the orthogonal matrix from an LQ factorization determined by DGELQF. (Computational routine.)."},
  {"dormql", "Multiplies a real general matrix by the orthogonal matrix from a QL factorization determined by DGEQLF. (Computational routine.)."},
  {"dormqr", "Multiplies a real general matrix by the orthogonal matrix from a QR factorization determined by DGEQRF. (Computational routine.)."},
  {"dormrq", "Multiplies a real general matrix by the orthogonal matrix from an RQ factorization determined by DGERQF. (Computational routine.)."},
  {"dormtr", "Multiplies a real general matrix by the orthogonal transformation matrix from a reduction to tridiagonal form determined by DSYTRD. (Computational routine.)."},
  {"dpbcon", "Estimates the reciprocal of the condition number of a real symmetric positive definite band matrix, using the Cholesky factorization computed by DPBTRF. (Computational routine.)."},
  {"dpbequ", "Computes row and column scalings to equilibrate a real symmetric positive definite band matrix and reduce its condition number. (Computational routine.)."},
  {"dpbrfs", "Improves the computed solution to a real symmetric positive definite banded system of linear equations AX=B, and provides forward and backward error bounds for the solution. (Computational routine.)."},
  {"dpbstf", "Computes a split Cholesky factorization of a real symmetric positive definite band matrix. Designed to be used in conjunction with DSBGST. The factorization has the form A = S''S, where S is a band matrix of the same bandwidth as A, with block rows ( U 0 ) and ( M L ), where U is upper triangular of order m = (n+b)/2, L is lower triangular of order n-m, and b is the bandwidth of A."},
  {"dpbsv", "Solves a real symmetric positive definite banded system of linear equations AX=B. (Simple driver.)."},
  {"dpbsvx", "Solves a real symmetric positive definite banded system of linear equations AX=B, and provides an estimate of the condition number and error bounds on the solution. (Expert driver.)."},
  {"dpbtrf", "Computes the Cholesky factorization of a real symmetric positive definite band matrix. (Computational routine.)."},
  {"dpbtrs", "Solves a real symmetric positive definite banded system of linear equations AX=B, using the Cholesky factorization computed by DPBTRF. (Computational routine.)."},
  {"dpocon", "Estimates the reciprocal of the condition number of a real symmetric positive definite matrix, using the Cholesky factorization computed by SPOTRF/CPOTRF. by SPOTRF/CPOTRF. (Computational routine.)."},
  {"dpoequ", "Computes row and column scalings to equilibrate a real symmetric positive definite matrix and reduce its condition number. (Computational routine.)."},
  {"dporfs", "Improves the computed solution to a real symmetric positive definite system of linear equations AX=B, and provides forward and backward error bounds for the solution. (Computational routine.)."},
  {"dposv", "Solves a real symmetric positive definite system of linear equations AX=B. (Simple driver.)."},
  {"dposvx", "Solves a real symmetric positive definite system of linear equations AX=B, and provides an estimate of the condition number and error bounds on the solution. (Expert driver.)."},
  {"dpotrf", "Computes the Cholesky factorization of a real symmetric positive definite matrix. (Computational routine.)."},
  {"dpotri", "Computes the inverse of a real symmetric positive definite matrix, using the Cholesky factorization computed by DPOTRF. (Computational routine.)."},
  {"dpotrs", "Solves a real symmetric positive definite system of linear equations AX=B, using the Cholesky factorization computed by DPOTRF. (Computational routine.)."},
  {"dppcon", "Estimates the reciprocal of the condition number of a real symmetric positive definite matrix in packed storage, using the Cholesky factorization computed by DPPTRF. (Computational routine.)."},
  {"dppequ", "Computes row and column scalings to equilibrate a real symmetric positive definite matrix in packed storage and reduce its condition number. (Computational routine.)."},
  {"dpprfs", "Improves the computed solution to a real symmetric positive definite system of linear equations AX=B, where A is held in packed storage, and provides forward and backward error bounds for the solution. (Computational routine.)."},
  {"dppsv", "Solves a real symmetric positive definite system of linear equations AX=B, where A is held in packed storage. (Simple driver.)."},
  {"dppsvx", "Solves a real symmetric positive definite system of linear equations AX=B, where A is held in packed storage, and provides an estimate of the condition number and error bounds on the solution. (Expert driver.)."},
  {"dpptrf", "Computes the Cholesky factorization of a real symmetric positive definite matrix in packed storage. (Computational routine.)."},
  {"dpptri", "Computes the inverse of a real symmetric positive definite matrix in packed storage, using the Cholesky factorization computed by DPPTRF. (Computational routine.)."},
  {"dpptrs", "Solves a real symmetric positive definite system of linear equations AX=B, where A is held in packed storage, using the Cholesky factorization computed by DPPTRF. (Computational routine.)."},
  {"dptcon", "Computes the reciprocal of the condition number of a real symmetric positive definite tridiagonal matrix, using the LDL**H factorization computed by DPTTRF. (Computational routine.)."},
  {"dpteqr", "Computes all eigenvalues and eigenvectors of a real symmetric positive definite tridiagonal matrix, by computing the SVD of its bidiagonal Cholesky factor. (Computational routine.)."},
  {"dptrfs", "Improves the computed solution to a real symmetric positive definite tridiagonal system of linear equations AX=B, and provides forward and backward error bounds for the solution. (Computational routine.)."},
  {"dptsv", "Solves a real symmetric positive definite tridiagonal system of linear equations AX=B. (Simple driver.)."},
  {"dptsvx", "Solves a real symmetric positive definite tridiagonal system of linear equations AX=B, and provides an estimate of the condition number and error bounds on the solution. (Expert driver.)."},
  {"dpttrf", "Computes the LDL**H factorization of a real symmetric positive definite tridiagonal matrix. (Computational routine.)."},
  {"dpttrs", "Solves a real symmetric positive definite tridiagonal system of linear equations, using the LDL**H factorization computed by DPTTRF. (Computational routine.)."},
  {"dsbev", "Computes all eigenvalues and eigenvectors of a real symmetric band matrix. (Simple driver.)."},
  {"dsbevd", "Computes all eigenvalues and, optionally, eigenvectors of a real symmetric band matrix. If eigenvectors are desired, an divide and conquer algorithm is used."},
  {"dsbevx", "Computes selected eigenvalues and eigenvectors of a real symmetric band matrix. (Expert driver.)."},
  {"dsbgst", "Reduces a real symmetric definite banded generalized eigenproblem A * x = lambda * B * x to standard form C * y = lambda * y, such that C has the same bandwidth as A. B must have been previously factorized by DPBSTF."},
  {"dsbgv", "Computes all the eigenvalues and, optionally, the eigenvectors, of a real generalized symmetric definite banded eigenproblem, of the form A * x = lambda * B * x. A and B are assumed to be symmetric and banded, and B is also positive definite."},
  {"dsbtrd", "Reduces a real symmetric band matrix to real symmetric tridiagonal form by an orthogonal similarity transformation. (Computational routine.)."},
  {"dspcon", "Estimates the reciprocal of the condition number of a real symmetric/complex symmetric/complex Hermitian indefinite matrix in packed storage, using the factorization computed by DSPTRF. (Computational routine.)."},
  {"dspev", "Computes all eigenvalues and eigenvectors of a real symmetric matrix in packed storage. (Simple driver.)."},
  {"dspevd", "Computes all eigenvalues and, optionally, eigenvectors of a real symmetric matrix in packed storage. If eigenvectors are desired, an divide and conquer alg is used."},
  {"dspevx", "Computes selected eigenvalues and eigenvectors of a real symmetric matrix in packed storage. (Expert driver.)."},
  {"dspgst", "Reduces a real symmetric-definite generalized eigenproblem Ax= lambda Bx, ABx= lambda x, or BAx= lambda x, to standard form, where A and B are held in packed storage, and B has been factorized by SPPTRF. (Computational routine.)."},
  {"dspgv", "Computes all eigenvalues and eigenvectors of a generalized real symmetric-definite generalized eigenproblem, Ax= lambda Bx, ABx= lambda x, or BAx= lambda x, where A and B are in packed storage. (Simple driver.)."},
  {"dsprfs", "Improves the computed solution to a real symmetric/complex symmetric/complex Hermitian indefinite system of linear equations AX=B, where A is held in packed storage, and provides forward and backward error bounds for the solution. (Computational routine.)."},
  {"dspsv", "Solves a real symmetric indefinite system of linear equations AX=B, where A is held in packed storage. (Simple driver.)."},
  {"dspsvx", "Solves a real symmetric indefinite system of linear equations AX=B, where A is held in packed storage, and provides an estimate of the condition number and error bounds on the solution. (Expert driver.)."},
  {"dsptrd", "Reduces a real symmetric matrix in packed storage to real symmetric tridiagonal form by an orthogonal similarity transformation. (Computational routine.)."},
  {"dsptrf", "Computes the factorization of a real symmetric/complex symmetric/complex Hermitian-indefinite matrix in packed storage, using the diagonal pivoting method. (Computational routine.)."},
  {"dsptri", "Computes the inverse of a real symmetric/complex symmetric/complex Hermitian indefinite matrix in packed storage, using the factorization computed by DSPTRF. (Computational routine.)."},
  {"dsptrs", "Solves a real symmetric indefinite system of linear equations AX=B, where A is held in packed storage, using the factorization computed by DSPTRF. (Computational routine.)."},
  {"dstebz", "Computes selected eigenvalues of a real symmetric tridiagonal matrix by bisection. (Computational routine.)."},
  {"dstein", "Computes selected eigenvectors of a real symmetric tridiagonal matrix by inverse iteration. (Computational routine.)."},
  {"dsteqr", "Computes all eigenvalues and eigenvectors of a real symmetric tridiagonal matrix, using the implicit QL or QR algorithm. (Computational routine.)."},
  {"dsterf", "Computes all eigenvalues of a real symmetric tridiagonal matrix, using a root-free variant of the QL or QR algorithm. (Computational routine.)."},
  {"dstev", "Computes all eigenvalues and eigenvectors of a real symmetric tridiagonal matrix. (Simple driver.)."},
  {"dstevd", "Computes all eigenvalues and, optionally, eigenvectors of a real symmetric tridiagonal matrix. If eigenvectors are desired, an divide and conquer algorithm is used."},
  {"dstevx", "Computes selected eigenvalues and eigenvectors of a real symmetric tridiagonal matrix. (Expert driver.)."},
  {"dsycon", "Estimates the reciprocal of the condition number of a real symmetric/complex symmetric/complex Hermitian indefinite matrix, using the factorization computed by DSYTRF. (Computational routine.)."},
  {"dsyev", "Computes all eigenvalues and eigenvectors of a real symmetric matrix. (Simple driver.)."},
  {"dsyevd", "Computes all eigenvalues and, optionally, eigenvectors of a real symmetric matrix.. If eigenvectors are desired, an divide and conquer algorithm is used."},
  {"dsyevx", "Computes selected eigenvalues and eigenvectors of a real symmetric matrix. (Expert driver.)."},
  {"dsygst", "Reduces a real symmetric-definite generalized eigenproblem Ax= lambda Bx, ABx= lambda x, or BAx= lambda x, to standard form, where B has been factorized by DPOTRF. (Computational routine.)."},
  {"dsygv", "Computes all eigenvalues and the eigenvectors of a generalized real symmetric-definite generalized eigenproblem, Ax= lambda Bx, ABx= lambda x, or BAx= lambda x. (Simple driver.)."},
  {"dsyrfs", "Improves the computed solution to a real symmetric/complex symmetric/complex Hermitian indefinite system of linear equations AX=B, and provides forward and backward error bounds for the solution. (Computational routine.)."},
  {"dsysv", "Solves a real symmetric indefinite system of linear equations AX=B. (Simple driver.)."},
  {"dsysvx", "Solves a real symmetric indefinite system of linear equations AX=B, and provides an estimate of the condition number and error bounds on the solution. (Expert driver.)."},
  {"dsytrd", "Reduces a real symmetric matrix to real symmetric tridiagonal form by an orthogonal similarity transformation. (Computational routine.)."},
  {"dsytrf", "Computes the factorization of a real symmetric/complex symmetric/complex Hermitian-indefinite matrix, using the diagonal pivoting method. (Computational routine.)."},
  {"dsytri", "Computes the inverse of a real symmetric/complex symmetric/complex Hermitian indefinite matrix, using the factorization computed by DSYTRF. (Computational routine.)."},
  {"dsytrs", "Solves a real symmetric indefinite system of linear equations AX=B, using the factorization computed by DSPTRF. (Computational routine.)."},
  {"dtbcon", "Estimates the reciprocal of the condition number of a real triangular band matrix, in either the 1-norm or the infinity-norm. (Computational routine.)."},
  {"dtbrfs", "Provides forward and backward error bounds for the solution of a real triangular banded system of linear equations AX=B, A**T X=B or A**H X=B. (Computational routine.)."},
  {"dtbtrs", "Solves a real triangular banded system of linear equations AX=B, A**T X=B or A**H X=B. (Computational routine.)."},
  {"dtpcon", "Estimates the reciprocal of the condition number of a real triangular matrix in packed storage, in either the 1-norm or the infinity-norm. (Computational routine.)."},
  {"dtprfs", "Provides forward and backward error bounds for the solution of a real triangular system of linear equations AX=B, A**T X=B or A**H X=B, where A is held in packed storage. (Computational routine.)."},
  {"dtptri", "Computes the inverse of a real triangular matrix in packed storage. (Computational routine.)."},
  {"dtptrs", "Solves a real triangular system of linear equations AX=B, A**T X=B or A**H X=B, where A is held in packed storage. (Computational routine.)."},
  {"dtrcon", "Estimates the reciprocal of the condition number of a real triangular matrix, in either the 1-norm or the infinity-norm. (Computational routine.)."},
  {"dtrevc", "Computes left and right eigenvectors of a real upper quasi-triangular matrix. (Computational routine.)."},
  {"dtrexc", "Reorders the Schur factorization of a real matrix by a unitary similarity transformation. (Computational routine.)."},
  {"dtrrfs", "Provides forward and backward error bounds for the solution of a real triangular system of linear equations A X=B, A**T X=B or A**H X=B. (Computational routine.)."},
  {"dtrsen", "Reorders the Schur factorization of a real matrix in order to find an orthonormal basis of a right invariant subspace corresponding to selected eigenvalues, and returns reciprocal condition numbers (sensitivities) of the average of the cluster of eigenvalues and of the invariant subspace. (Computational routine.)."},
  {"dtrsna", "Estimates the reciprocal condition numbers (sensitivities) of selected eigenvalues and eigenvectors of a real upper quasi-triangular matrix. (Computational routine.)."},
  {"dtrsyl", "Solves the a real Sylvester matrix equation A X +/- X B=C where A and B are upper quasi-triangular, and may be transposed. (Computational routine.)."},
  {"dtrtri", "Computes the inverse of a real triangular matrix. (Computational routine.)."},
  {"dtrtrs", "Solves a real triangular system of linear equations AX=B, A**T X=B or A**H X=B. (Computational routine.)."},
  {"dtzrqf", "Computes an RQ factorization of a real upper trapezoidal matrix. (Computational routine.)."},
  {"sbdsqr", "Computes the singular value decomposition (SVD) of a real bidiagonal matrix, using the bidiagonal QR algorithm. (Computational routine.)."},
  {"sgbbrd", "Reduces a real general m-by-n band matrix A to upper bidiagonal form B by an orthogonal transformation: Q'' * A * P = B. The routine computes B and optionally forms Q or P'', or computes Q'' * C for a given matrix C."},
  {"sgbcon", "Estimates the reciprocal of the condition number of a general band matrix, in either the 1-norm or the infinity-norm, using the LU factorization computed by SGBTRF. (Computational routine.)."},
  {"sgbequ", "Computes row and column scalings to equilibrate a real general band matrix and reduce its condition number. (Computational routine.)."},
  {"sgbrfs", "Improves the computed solution to a real general banded system of linear equations AX=B, A**T X=B or A**H X=B, and provides forward and backward error bounds for the solution. (Computational routine.)."},
  {"sgbsv", "Solves a real general banded system of linear equations AX=B. (Simple driver.)."},
  {"sgbsvx", "Solves a real general banded system of linear equations AX=B, A**T X=B or A**H X=B, and provides an estimate of the condition number and error bounds on the solution. (Expert driver.)."},
  {"sgbtrf", "Computes an LU factorization of a real general band matrix, using partial pivoting with row interchanges. (Computational routine.)."},
  {"sgbtrs", "Solves a real general banded system of linear equations AX=B, A**T X=B or A**H X=B, using the LU factorization computed by SGBTRF. (Computational routine.)."},
  {"sgebak", "Transforms eigenvectors of a balanced matrix to those of the original matrix supplied to SGEBAL. (Computational routine.)."},
  {"sgebal", "Balances a real general matrix in order to improve the accuracy of computed eigenvalues. (Computational routine.)."},
  {"sgebrd", "Reduces a real general rectangular matrix to real bidiagonal form by an orthogonal transformation. (Computational routine.)."},
  {"sgecon", "Estimates the reciprocal of the condition number of a general matrix, in either the 1-norm or the infinity-norm, using the LU factorization computed by SGETRF. (Computational routine.)."},
  {"sgeequ", "Computes row and column scalings to equilibrate a general rectangular matrix and reduce its condition number. (Computational routine.)."},
  {"sgees", "Computes the eigenvalues and Schur factorization of a general matrix, and orders the factorization so that selected eigenvalues are at the top left of the Schur form. (Simple driver.)."},
  {"sgeesx", "Computes the eigenvalues and Schur factorization of a general matrix, orders the factorization so that selected eigenvalues are at the top left of the Schur form, and computes reciprocal condition numbers for the average of the selected eigenvalues, and for the associated right invariant subspace. (Expert driver.)."},
  {"sgeev", "Computes the eigenvalues and left and right eigenvectors of a real general matrix. (Simple driver.)."},
  {"sgeevx", "Computes the eigenvalues and left and right eigenvectors of a real general matrix, with preliminary balancing of the matrix, and computes reciprocal condition numbers for the eigenvalues and right eigenvectors. (Expert driver.)."},
  {"sgehrd", "Reduces a real general matrix to upper Hessenberg form by an orthogonal similarity transformation. (Computational routine.)."},
  {"sgelqf", "Computes an LQ factorization of a real general rectangular matrix. (Computational routine.)."},
  {"sgels", "Computes the least squares solution to an over-determined system of linear equations, A X=B or A**H X=B, or the minimum norm solution of an under-determined system, where A is a general rectangular matrix of full rank, using a QR or LQ factorization of A. (Simple driver.)."},
  {"sgelss", "Computes the minimum norm least squares solution to an over- or under-determined system of linear equations A X=B, using the singular value decomposition of A. (Simple driver.)."},
  {"sgelsx", "Computes the minimum norm least squares solution to an over- or under-determined system of linear equations A X=B, using a complete orthogonal factorization of A. (Expert driver.)."},
  {"sgeqlf", "Computes a QL factorization of a real general rectangular matrix. (Computational routine.)."},
  {"sgeqpf", "Computes a QR factorization with column pivoting of a general rectangular matrix. (Computational routine.)."},
  {"sgeqrf", "Computes a QR factorization of a real general rectangular matrix. (Computational routine.)."},
  {"sgerfs", "Improves the computed solution to a real general system of linear equations AX=B, A**T X=B or A**H X=B, and provides forward and backward error bounds for the solution. (Computational routine.)."},
  {"sgerqf", "Computes an RQ factorization of a real general rectangular matrix. (Computational routine.)."},
  {"sgesv", "Solves a real general system of linear equations AX=B. (Simple driver.)."},
  {"sgesvd", "Computes the singular value decomposition (SVD) of a general rectangular matrix. (Simple driver.)."},
  {"sgesvx", "Solves a real general system of linear equations AX=B, A**T X=B or A**H X=B, and provides an estimate of the condition number and error bounds on the solution. (Expert driver.)."},
  {"sgetrf", "Computes an LU factorization of a real general matrix, using partial pivoting with row interchanges. (Computational routine.)."},
  {"sgetri", "Computes the inverse of a real general matrix, using the LU factorization computed by SGETRF. (Computational routine.)."},
  {"sgetrs", "Solves a real general system of linear equations AX=B, A**T X=B or A**H X=B, using the LU factorization computed by SGETRF. (Computational routine.)."},
  {"sgtcon", "Estimates the reciprocal of the condition number of a general tridiagonal matrix, in either the 1-norm or the infinity-norm, using the LU factorization computed by SGTTRF. (Computational routine.)."},
  {"sgtrfs", "Improves the computed solution to a real general tridiagonal system of linear equations AX=B, A**T X=B or A**H X=B, and provides forward and backward error bounds for the solution. (Computational routine.)."},
  {"sgtsv", "Solves a real general tridiagonal system of linear equations AX=B. (Simple driver.)."},
  {"sgtsvx", "Solves a real general tridiagonal system of linear equations AX=B, A**T X=B or A**H X=B, and provides an estimate of the condition number and error bounds on the solution. (Expert driver.)."},
  {"sgttrf", "Computes an LU factorization of a real general tridiagonal matrix, using partial pivoting with row interchanges. (Computational routine.)."},
  {"sgttrs", "Solves a real general tridiagonal system of linear equations AX=B, A**T X=B or A**H X=B, using the LU factorization computed by SGTTRF. (Computational routine.)."},
  {"shsein", "Computes specified right and/or left eigenvectors of a real upper Hessenberg matrix by inverse iteration. (Computational routine.)."},
  {"shseqr", "Computes the eigenvalues and Schur factorization of a real upper Hessenberg matrix, using the multishift QR algorithm. (Computational routine.)."},
  {"sopgtr", "Generates the orthogonal transformation matrix from a reduction to tridiagonal form determined by SSPTRD. (Computational routine.)."},
  {"sopmtr", "Multiplies a real general matrix by the orthogonal transformation matrix from a reduction to tridiagonal form determined by SSPTRD. (Computational routine.)."},
  {"sorgbr", "Generates the orthogonal transformation matrices from a reduction to bidiagonal form determined by SGEBRD. (Computational routine.)."},
  {"sorghr", "Generates the orthogonal transformation matrix from a reduction to Hessenberg form determined by SGEHRD. (Computational routine.)."},
  {"sorglq", "Generates all or part of the orthogonal matrix Q from an LQ factorization determined by SGELQF. (Computational routine.)."},
  {"sorgql", "Generates all or part of the orthogonal matrix Q from a QL factorization determined by SGEQLF. (Computational routine.)."},
  {"sorgqr", "Generates all or part of the orthogonal matrix Q from a QR factorization determined by SGEQRF. (Computational routine.)."},
  {"sorgrq", "Generates all or part of the orthogonal matrix Q from an RQ factorization determined by SGERQF. (Computational routine.)."},
  {"sorgtr", "Generates the orthogonal transformation matrix from a reduction to tridiagonal form determined by SSYTRD. (Computational routine.)."},
  {"sormbr", "Multiplies a real general matrix by one of the orthogonal transformation matrices from a reduction to bidiagonal form determined by SGEBRD. (Computational routine.)."},
  {"sormhr", "Multiplies a real general matrix by the orthogonal transformation matrix from a reduction to Hessenberg form determined by SGEHRD. (Computational routine.)."},
  {"sormlq", "Multiplies a real general matrix by the orthogonal matrix from an LQ factorization determined by SGELQF. (Computational routine.)."},
  {"sormql", "Multiplies a real general matrix by the orthogonal matrix from a QL factorization determined by SGEQLF. (Computational routine.)."},
  {"sormqr", "Multiplies a real general matrix by the orthogonal matrix from a QR factorization determined by SGEQRF. (Computational routine.)."},
  {"sormrq", "Multiplies a real general matrix by the orthogonal matrix from an RQ factorization determined by SGERQF. (Computational routine.)."},
  {"sormtr", "Multiplies a real general matrix by the orthogonal transformation matrix from a reduction to tridiagonal form determined by SSYTRD. (Computational routine.)."},
  {"spbcon", "Estimates the reciprocal of the condition number of a real symmetric positive definite band matrix, using the Cholesky factorization computed by SPBTRF. (Computational routine.)."},
  {"spbequ", "Computes row and column scalings to equilibrate a real symmetric positive definite band matrix and reduce its condition number. (Computational routine.)."},
  {"spbrfs", "Improves the computed solution to a real symmetric positive definite banded system of linear equations AX=B, and provides forward and backward error bounds for the solution. (Computational routine.)."},
  {"spbstf", "Computes a split Cholesky factorization of a real symmetric positive definite band matrix. Designed to be used in conjunction with SSBGST. The factorization has the form A = S''S, where S is a band matrix of the same bandwidth as A, with block rows ( U 0 ) and ( M L ), where U is upper triangular of order m = (n+b)/2, L is lower triangular of order n-m, and b is the bandwidth of A."},
  {"spbsv", "Solves a real symmetric positive definite banded system of linear equations AX=B. (Simple driver.)."},
  {"spbsvx", "Solves a real symmetric positive definite banded system of linear equations AX=B, and provides an estimate of the condition number and error bounds on the solution. (Expert driver.)."},
  {"spbtrf", "Computes the Cholesky factorization of a real symmetric positive definite band matrix. (Computational routine.)."},
  {"spbtrs", "Solves a real symmetric positive definite banded system of linear equations AX=B, using the Cholesky factorization computed by SPBTRF. (Computational routine.)."},
  {"spocon", "Estimates the reciprocal of the condition number of a real symmetric positive definite matrix, using the Cholesky factorization computed by SPOTRF/CPOTRF. by SPOTRF/CPOTRF. (Computational routine.)."},
  {"spoequ", "Computes row and column scalings to equilibrate a real symmetric positive definite matrix and reduce its condition number. (Computational routine.)."},
  {"sporfs", "Improves the computed solution to a real symmetric positive definite system of linear equations AX=B, and provides forward and backward error bounds for the solution. (Computational routine.)."},
  {"sposv", "Solves a real symmetric positive definite system of linear equations AX=B. (Simple driver.)."},
  {"sposvx", "Solves a real symmetric positive definite system of linear equations AX=B, and provides an estimate of the condition number and error bounds on the solution. (Expert driver.)."},
  {"spotrf", "Computes the Cholesky factorization of a real symmetric positive definite matrix. (Computational routine.)."},
  {"spotri", "Computes the inverse of a real symmetric positive definite matrix, using the Cholesky factorization computed by SPOTRF. (Computational routine.)."},
  {"spotrs", "Solves a real symmetric positive definite system of linear equations AX=B, using the Cholesky factorization computed by SPOTRF. (Computational routine.)."},
  {"sppcon", "Estimates the reciprocal of the condition number of a real symmetric positive definite matrix in packed storage, using the Cholesky factorization computed by SPPTRF. (Computational routine.)."},
  {"sppequ", "Computes row and column scalings to equilibrate a real symmetric positive definite matrix in packed storage and reduce its condition number. (Computational routine.)."},
  {"spprfs", "Improves the computed solution to a real symmetric positive definite system of linear equations AX=B, where A is held in packed storage, and provides forward and backward error bounds for the solution. (Computational routine.)."},
  {"sppsv", "Solves a real symmetric positive definite system of linear equations AX=B, where A is held in packed storage. (Simple driver.)."},
  {"sppsvx", "Solves a real symmetric positive definite system of linear equations AX=B, where A is held in packed storage, and provides an estimate of the condition number and error bounds on the solution. (Expert driver.)."},
  {"spptrf", "Computes the Cholesky factorization of a real symmetric positive definite matrix in packed storage. (Computational routine.)."},
  {"spptri", "Computes the inverse of a real symmetric positive definite matrix in packed storage, using the Cholesky factorization computed by SPPTRF. (Computational routine.)."},
  {"spptrs", "Solves a real symmetric positive definite system of linear equations AX=B, where A is held in packed storage, using the Cholesky factorization computed by SPPTRF. (Computational routine.)."},
  {"sptcon", "Computes the reciprocal of the condition number of a real symmetric positive definite tridiagonal matrix, using the LDL**H factorization computed by SPTTRF. (Computational routine.)."},
  {"spteqr", "Computes all eigenvalues and eigenvectors of a real symmetric positive definite tridiagonal matrix, by computing the SVD of its bidiagonal Cholesky factor. (Computational routine.)."},
  {"sptrfs", "Improves the computed solution to a real symmetric positive definite tridiagonal system of linear equations AX=B, and provides forward and backward error bounds for the solution. (Computational routine.)."},
  {"sptsv", "Solves a real symmetric positive definite tridiagonal system of linear equations AX=B. (Simple driver.)."},
  {"sptsvx", "Solves a real symmetric positive definite tridiagonal system of linear equations AX=B, and provides an estimate of the condition number and error bounds on the solution. (Expert driver.)."},
  {"spttrf", "Computes the LDL**H factorization of a real symmetric positive definite tridiagonal matrix. (Computational routine.)."},
  {"spttrs", "Solves a real symmetric positive definite tridiagonal system of linear equations, using the LDL**H factorization computed by SPTTRF. (Computational routine.)."},
  {"ssbev", "Computes all eigenvalues and eigenvectors of a real symmetric band matrix. (Simple driver.)."},
  {"ssbevd", "Computes all eigenvalues and, optionally, eigenvectors of a real symmetric band matrix. If eigenvectors are desired, an divide and conquer algorithm is used."},
  {"ssbevx", "Computes selected eigenvalues and eigenvectors of a real symmetric band matrix. (Expert driver.)."},
  {"ssbgst", "Reduces a real symmetric definite banded generalized eigenproblem A * x = lambda * B * x to standard form C * y = lambda * y, such that C has the same bandwidth as A. B must have been previously factorized by SPBSTF."},
  {"ssbgv", "Computes all the eigenvalues and, optionally, the eigenvectors, of a real generalized symmetric definite banded eigenproblem, of the form A * x = lambda * B * x. A and B are assumed to be symmetric and banded, and B is also positive definite."},
  {"ssbtrd", "Reduces a real symmetric band matrix to real symmetric tridiagonal form by an orthogonal similarity transformation. (Computational routine.)."},
  {"sspcon", "Estimates the reciprocal of the condition number of a real symmetric/complex symmetric/complex Hermitian indefinite matrix in packed storage, using the factorization computed by SSPTRF. (Computational routine.)."},
  {"sspev", "Computes all eigenvalues and eigenvectors of a real symmetric matrix in packed storage. (Simple driver.)."},
  {"sspevd", "Computes all eigenvalues and, optionally, eigenvectors of a real symmetric matrix in packed storage. If eigenvectors are desired, an divide and conquer algorithm is used."},
  {"sspevx", "Computes selected eigenvalues and eigenvectors of a real symmetric matrix in packed storage. (Expert driver.)."},
  {"sspgst", "Reduces a real symmetric-definite generalized eigenproblem Ax= lambda Bx, ABx= lambda x, or BAx= lambda x, to standard form, where A and B are held in packed storage, and B has been factorized by SPPTRF. (Computational routine.)."},
  {"sspgv", "Computes all eigenvalues and eigenvectors of a generalized real symmetric-definite generalized eigenproblem, Ax= lambda Bx, ABx= lambda x, or BAx= lambda x, where A and B are in packed storage. (Simple driver.)."},
  {"ssprfs", "Improves the computed solution to a real symmetric/complex symmetric/complex Hermitian indefinite system of linear equations AX=B, where A is held in packed storage, and provides forward and backward error bounds for the solution. (Computational routine.)."},
  {"sspsv", "Solves a real symmetric indefinite system of linear equations AX=B, where A is held in packed storage. (Simple driver.)."},
  {"sspsvx", "Solves a real symmetric indefinite system of linear equations AX=B, where A is held in packed storage, and provides an estimate of the condition number and error bounds on the solution. (Expert driver.)."},
  {"ssptrd", "Reduces a real symmetric matrix in packed storage to real symmetric tridiagonal form by an orthogonal similarity transformation. (Computational routine.)."},
  {"ssptrf", "Computes the factorization of a real symmetric/complex symmetric/complex Hermitian-indefinite matrix in packed storage, using the diagonal pivoting method. (Computational routine.)."},
  {"ssptri", "Computes the inverse of a real symmetric/complex symmetric/complex Hermitian indefinite matrix in packed storage, using the factorization computed by SSPTRF. (Computational routine.)."},
  {"ssptrs", "Solves a real symmetric indefinite system of linear equations AX=B, where A is held in packed storage, using the factorization computed by SSPTRF. (Computational routine.)."},
  {"sstebz", "Computes selected eigenvalues of a real symmetric tridiagonal matrix by bisection. (Computational routine.)."},
  {"sstein", "Computes selected eigenvectors of a real symmetric tridiagonal matrix by inverse iteration. (Computational routine.)."},
  {"ssteqr", "Computes all eigenvalues and eigenvectors of a real symmetric tridiagonal matrix, using the implicit QL or QR algorithm. (Computational routine.)."},
  {"ssterf", "Computes all eigenvalues of a real symmetric tridiagonal matrix, using a root-free variant of the QL or QR algorithm. (Computational routine.)."},
  {"sstev", "Computes all eigenvalues and eigenvectors of a real symmetric tridiagonal matrix. (Simple driver.)."},
  {"sstevd", "Computes all eigenvalues and, optionally, eigenvectors of a real symmetric tridiagonal matrix. If eigenvectors are desired, an divide and conquer algorithm is used."},
  {"sstevx", "Computes selected eigenvalues and eigenvectors of a real symmetric tridiagonal matrix. (Expert driver.)."},
  {"ssycon", "Estimates the reciprocal of the condition number of a real symmetric/complex symmetric/complex Hermitian indefinite matrix, using the factorization computed by SSYTRF. (Computational routine.)."},
  {"ssyev", "Computes all eigenvalues and eigenvectors of a real symmetric matrix. (Simple driver.)."},
  {"ssyevd", "Computes all eigenvalues and, optionally, eigenvectors of a real symmetric matrix. If eigenvectors are desired, an divide and conquer algorithm is used."},
  {"ssyevx", "Computes selected eigenvalues and eigenvectors of a real symmetric matrix. (Expert driver.)."},
  {"ssygst", "Reduces a real symmetric-definite generalized eigenproblem Ax= lambda Bx, ABx= lambda x, or BAx= lambda x, to standard form, where B has been factorized by SPOTRF. (Computational routine.)."},
  {"ssygv", "Computes all eigenvalues and the eigenvectors of a generalized real symmetric-definite generalized eigenproblem, Ax= lambda Bx, ABx= lambda x, or BAx= lambda x. (Simple driver.)."},
  {"ssyrfs", "Improves the computed solution to a real symmetric/complex symmetric/complex Hermitian indefinite system of linear equations AX=B, and provides forward and backward error bounds for the solution. (Computational routine.)."},
  {"ssysv", "Solves a real symmetric indefinite system of linear equations AX=B. (Simple driver.)."},
  {"ssysvx", "Solves a real symmetric indefinite system of linear equations AX=B, and provides an estimate of the condition number and error bounds on the solution. (Expert driver.)."},
  {"ssytrd", "Reduces a real symmetric matrix to real symmetric tridiagonal form by an orthogonal similarity transformation. (Computational routine.)."},
  {"ssytrf", "Computes the factorization of a real symmetric/complex symmetric/complex Hermitian-indefinite matrix, using the diagonal pivoting method. (Computational routine.)."},
  {"ssytri", "Computes the inverse of a real symmetric/complex symmetric/complex Hermitian indefinite matrix, using the factorization computed by SSYTRF. (Computational routine.)."},
  {"ssytrs", "Solves a real symmetric indefinite system of linear equations AX=B, using the factorization computed by SSPTRF. (Computational routine.)."},
  {"stbcon", "Estimates the reciprocal of the condition number of a real triangular band matrix, in either the 1-norm or the infinity-norm. (Computational routine.)."},
  {"stbrfs", "Provides forward and backward error bounds for the solution of a real triangular banded system of linear equations AX=B, A**T X=B or A**H X=B. (Computational routine.)."},
  {"stbtrs", "Solves a real triangular banded system of linear equations AX=B, A**T X=B or A**H X=B. (Computational routine.)."},
  {"stpcon", "Estimates the reciprocal of the condition number of a real triangular matrix in packed storage, in either the 1-norm or the infinity-norm. (Computational routine.)."},
  {"stprfs", "Provides forward and backward error bounds for the solution of a real triangular system of linear equations AX=B, A**T X=B or A**H X=B, where A is held in packed storage. (Computational routine.)."},
  {"stptri", "Computes the inverse of a real triangular matrix in packed storage. (Computational routine.)."},
  {"stptrs", "Solves a real triangular system of linear equations AX=B, A**T X=B or A**H X=B, where A is held in packed storage. (Computational routine.)."},
  {"strcon", "Estimates the reciprocal of the condition number of a real triangular matrix, in either the 1-norm or the infinity-norm. (Computational routine.)."},
  {"strevc", "Computes left and right eigenvectors of a real upper quasi-triangular matrix. (Computational routine.)."},
  {"strexc", "Reorders the Schur factorization of a real matrix by a unitary similarity transformation. (Computational routine.)."},
  {"strrfs", "Provides forward and backward error bounds for the solution of a real triangular system of linear equations A X=B, A**T X=B or A**H X=B. (Computational routine.)."},
  {"strsen", "Reorders the Schur factorization of a real matrix in order to find an orthonormal basis of a right invariant subspace corresponding to selected eigenvalues, and returns reciprocal condition numbers (sensitivities) of the average of the cluster of eigenvalues and of the invariant subspace. (Computational routine.)."},
  {"strsna", "Estimates the reciprocal condition numbers (sensitivities) of selected eigenvalues and eigenvectors of a real upper quasi-triangular matrix. (Computational routine.)."},
  {"strsyl", "Solves the a real Sylvester matrix equation A X +/- X B=C where A and B are upper quasi-triangular, and may be transposed. (Computational routine.)."},
  {"strtri", "Computes the inverse of a real triangular matrix. (Computational routine.)."},
  {"strtrs", "Solves a real triangular system of linear equations AX=B, A**T X=B or A**H X=B. (Computational routine.)."},
  {"stzrqf", "Computes an RQ factorization of a real upper trapezoidal matrix. (Computational routine.)."},
  {"zbdsqr", "Computes the singular value decomposition (SVD) of a real bidiagonal matrix, using the bidiagonal QR algorithm. (Computational routine.)."},
  {"zgbbrd", "Reduces a complex general m-by-n band matrix A to real upper bidiagonal form B by a unitary transformation: Q'' * A * P = B. The routine computes B and optionally forms Q or P'', or computes Q'' * C for a given matrix C."},
  {"zgbcon", "Estimates the reciprocal of the condition number of a general band matrix, in either the 1-norm or the infinity-norm, using the LU factorization computed by ZGBTRF. (Computational routine.)."},
  {"zgbequ", "Computes row and column scalings to equilibrate a complex general band matrix and reduce its condition number. (Computational routine.)."},
  {"zgbrfs", "Improves the computed solution to a complex general banded system of linear equations AX=B, A**T X=B or A**H X=B, and provides forward and backward error bounds for the solution. (Computational routine.)."},
  {"zgbsv", "Solves a complex general banded system of linear equations AX=B. (Simple driver.)."},
  {"zgbsvx", "Solves a complex general banded system of linear equations AX=B, A**T X=B or A**H X=B, and provides an estimate of the condition number and error bounds on the solution. (Expert driver.)."},
  {"zgbtrf", "Computes an LU factorization of a complex general band matrix, using partial pivoting with row interchanges. (Computational routine.)."},
  {"zgbtrs", "Solves a complex general banded system of linear equations AX=B, A**T X=B or A**H X=B, using the LU factorization computed by ZGBTRF. (Computational routine.)."},
  {"zgebak", "Transforms eigenvectors of a balanced matrix to those of the original matrix supplied to ZGEBAL. (Computational routine.)."},
  {"zgebal", "Balances a complex general matrix in order to improve the accuracy of computed eigenvalues. (Computational routine.)."},
  {"zgebrd", "Reduces a complex general rectangular matrix to real bidiagonal form by an unitary transformation. (Computational routine.)."},
  {"zgecon", "Estimates the reciprocal of the condition number of a general matrix, in either the 1-norm or the infinity-norm, using the LU factorization computed by ZGETRF. (Computational routine.)."},
  {"zgeequ", "Computes row and column scalings to equilibrate a general rectangular matrix and reduce its condition number. (Computational routine.)."},
  {"zgees", "Computes the eigenvalues and Schur factorization of a general matrix, and orders the factorization so that selected eigenvalues are at the top left of the Schur form. (Simple driver.)."},
  {"zgeesx", "Computes the eigenvalues and Schur factorization of a general matrix, orders the factorization so that selected eigenvalues are at the top left of the Schur form, and computes reciprocal condition numbers for the average of the selected eigenvalues, and for the associated right invariant subspace. (Expert driver.)."},
  {"zgeev", "Computes the eigenvalues and left and right eigenvectors of a complex general matrix. (Simple driver.)."},
  {"zgeevx", "Computes the eigenvalues and left and right eigenvectors of a complex general matrix, with preliminary balancing of the matrix, and computes reciprocal condition numbers for the eigenvalues and right eigenvectors. (Expert driver.)."},
  {"zgehrd", "Reduces a complex general matrix to upper Hessenberg form by an unitary similarity transformation. (Computational routine.)."},
  {"zgelqf", "Computes an LQ factorization of a complex general rectangular matrix. (Computational routine.)."},
  {"zgels", "Computes the least squares solution to an over-determined system of linear equations, A X=B or A**H X=B, or the minimum norm solution of an under-determined system, where A is a general rectangular matrix of full rank, using a QR or LQ factorization of A. (Simple driver.)."},
  {"zgelss", "Computes the minimum norm least squares solution to an over- or under-determined system of linear equations A X=B, using the singular value decomposition of A. (Simple driver.)."},
  {"zgelsx", "Computes the minimum norm least squares solution to an over- or under-determined system of linear equations A X=B, using a complete orthogonal factorization of A. (Expert driver.)."},
  {"zgeqlf", "Computes a QL factorization of a complex general rectangular matrix. (Computational routine.)."},
  {"zgeqpf", "Computes a QR factorization with column pivoting of a general rectangular matrix. (Computational routine.)."},
  {"zgeqrf", "Computes a QR factorization of a complex general rectangular matrix. (Computational routine.)."},
  {"zgerfs", "Improves the computed solution to a complex general system of linear equations AX=B, A**T X=B or A**H X=B, and provides forward and backward error bounds for the solution. (Computational routine.)."},
  {"zgerqf", "Computes an RQ factorization of a complex general rectangular matrix. (Computational routine.)."},
  {"zgesv", "Solves a complex general system of linear equations AX=B. (Simple driver.)."},
  {"zgesvd", "Computes the singular value decomposition (SVD) of a general rectangular matrix. (Simple driver.)."},
  {"zgesvx", "Solves a complex general system of linear equations AX=B, A**T X=B or A**H X=B, and provides an estimate of the condition number and error bounds on the solution. (Expert driver.)."},
  {"zgetrf", "Computes an LU factorization of a complex general matrix, using partial pivoting with row interchanges. (Computational routine.)."},
  {"zgetri", "Computes the inverse of a complex general matrix, using the LU factorization computed by ZGETRF. (Computational routine.)."},
  {"zgetrs", "Solves a complex general system of linear equations AX=B, A**T X=B or A**H X=B, using the LU factorization computed by ZGETRF. (Computational routine.)."},
  {"zgtcon", "Estimates the reciprocal of the condition number of a general tridiagonal matrix, in either the 1-norm or the infinity-norm, using the LU factorization computed by ZGTTRF. (Computational routine.)."},
  {"zgtrfs", "Improves the computed solution to a complex general tridiagonal system of linear equations AX=B, A**T X=B or A**H X=B, and provides forward and backward error bounds for the solution. (Computational routine.)."},
  {"zgtsv", "Solves a complex general tridiagonal system of linear equations AX=B. (Simple driver.)."},
  {"zgtsvx", "Solves a complex general tridiagonal system of linear equations AX=B, A**T X=B or A**H X=B, and provides an estimate of the condition number and error bounds on the solution. (Expert driver.)."},
  {"zgttrf", "Computes an LU factorization of a complex general tridiagonal matrix, using partial pivoting with row interchanges. (Computational routine.)."},
  {"zgttrs", "Solves a complex general tridiagonal system of linear equations AX=B, A**T X=B or A**H X=B, using the LU factorization computed by ZGTTRF. (Computational routine.)."},
  {"zhbev", "Computes all eigenvalues and eigenvectors of a complex Hermitian band matrix.  (Simple driver.)"},
  {"zhbevd", "Computes all eigenvalues and, optionally, eigenvectors of a complex Hermitian band matrix."},
  {"zhbevx", "Computes selected eigenvalues and eigenvectors of a complex Hermitian band matrix.  (Expert driver.)"},
  {"zhbgst", "Reduces a complex Hermitian definite banded generalized eigenproblem A * x = lambda * B * x to standard form C * y = lambda * y, such that C has the same bandwidth as A. B must have been previously factorized by ZPBSTF."},
  {"zhbgv", "Computes all the eigenvalues and, optionally, the eigenvectors, of a complex generalized Hermitian definite banded eigenproblem, of the form A * x = lambda * B * x. A and B are assumed to be Hermitian and banded, and B is also positive definite."},
  {"zhbtrd", "Reduces a complex Hermitian band matrix to real symmetric tridiagonal form by an unitary similarity transformation.  (Computational routine.)"},
  {"zhecon", "Estimates the reciprocal of the condition number of a real symmetric/complex symmetric/complex Hermitian indefinite matrix, using the factorization computed by ZHETRF. (Computational routine.)."},
  {"zheev", "Computes all eigenvalues and eigenvectors of a complex Hermitian matrix. (Simple driver.)"},
  {"zheevd", "Computes all eigenvalues and, optionally, eigenvectors of a complex Hermitian matrix."},
  {"zheevx", "Computes selected eigenvalues and eigenvectors of a complex Hermitian matrix.  (Expert driver.)"},
  {"zhegst", "Reduces a complex Hermitian-definite generalized eigenproblem Ax= lambda Bx, ABx= lambda x, or BAx= lambda x, to standard form, where B has been factorized by ZPOTRF.  (Computational routine.)"},
  {"zhegv", "Computes all eigenvalues and the eigenvectors of a generalized complex Hermitian-definite generalized eigenproblem, Ax= lambda Bx, ABx= lambda x, or BAx= lambda x.  (Simple driver.)"},
  {"zherfs", "Improves the computed solution to a real symmetric/complex symmetric/complex Hermitian indefinite system of linear equations AX=B, and provides forward and backward error bounds for the solution. (Computational routine.)."},
  {"zhesv", "Solves a complex Hermitian indefinite system of linear equations AX=B. (Simple driver.)."},
  {"zhesvx", "Solves a complex Hermitian indefinite system of linear equations AX=B, and provides an estimate of the condition number and error bounds on the solution. (Expert driver.)."},
  {"zhetrd", "Reduces a complex Hermitian matrix to real symmetric tridiagonal form by an unitary similarity transformation.  (Computational routine.)"},
  {"zhetrf", "Computes the factorization of a real symmetric/complex symmetric/complex Hermitian-indefinite matrix, using the diagonal pivoting method. (Computational routine.)."},
  {"zhetri", "Computes the inverse of a real symmetric/complex symmetric/complex Hermitian indefinite matrix, using the factorization computed by ZHETRF. (Computational routine.)."},
  {"zhetrs", "Solves a complex Hermitian indefinite system of linear equations AX=B, using the factorization computed by ZHPTRF. (Computational routine.)."},
  {"zhpcon", "Estimates the reciprocal of the condition number of a real symmetric/complex symmetric/complex Hermitian indefinite matrix in packed storage, using the factorization computed by ZHPTRF. (Computational routine.)."},
  {"zhpev", "Computes all eigenvalues and eigenvectors of a complex Hermitian matrix in packed storage.  (Simple driver.)"},
  {"zhpevd", "Computes all eigenvalues and, optionally, eigenvectors of a complex Hermitian matrix in packed storage."},
  {"zhpevx", "Computes selected eigenvalues and eigenvectors of a complex Hermitian matrix in packed storage.  (Expert driver.)"},
  {"zhpgst", "Reduces a complex Hermitian-definite generalized eigenproblem Ax= lambda Bx, ABx= lambda x, or BAx= lambda x, to standard form, where A and B are held in packed storage, and B has been factorized by CPPTRF. (Computational routine.)"},
  {"zhpgv", "Computes all eigenvalues and eigenvectors of a generalized complex Hermitian-definite generalized eigenproblem, Ax= lambda Bx, ABx= lambda x, or BAx= lambda x, where A and B are in packed storage.  (Simple driver.)"},
  {"zhprfs", "Improves the computed solution to a real symmetric/complex symmetric/complex Hermitian indefinite system of linear equations AX=B, where A is held in packed storage, and provides forward and backward error bounds for the solution. (Computational routine.)."},
  {"zhpsv", "Solves a complex Hermitian indefinite system of linear equations AX=B, where A is held in packed storage. (Simple driver.)."},
  {"zhpsvx", "Solves a complex Hermitian indefinite system of linear equations AX=B, where A is held in packed storage, and provides an estimate of the condition number and error bounds on the solution. (Expert driver.)."},
  {"zhptrd", "Reduces a complex Hermitian matrix in packed storage to real symmetric tridiagonal form by an unitary similarity transformation.  (Computational routine.)"},
  {"zhptrf", "Computes the factorization of a real symmetric/complex symmetric/complex Hermitian-indefinite matrix in packed storage, using the diagonal pivoting method. (Computational routine.)."},
  {"zhptri", "Computes the inverse of a real symmetric/complex symmetric/complex Hermitian indefinite matrix in packed storage, using the factorization computed by ZHPTRF. (Computational routine.)."},
  {"zhptrs", "Solves a complex Hermitian indefinite system of linear equations AX=B, where A is held in packed storage, using the factorization computed by ZHPTRF. (Computational routine.)."},
  {"zhsein", "Computes specified right and/or left eigenvectors of an upper Hessenberg matrix by inverse iteration. (Computational routine.)."},
  {"zhseqr", "Computes the eigenvalues and Schur factorization of an upper Hessenberg matrix, using the multishift QR algorithm. (Computational routine.)."},
  {"zpbcon", "Estimates the reciprocal of the condition number of a complex Hermitian positive definite band matrix, using the Cholesky factorization computed by CPBTRF. (Computational routine.)."},
  {"zpbequ", "Computes row and column scalings to equilibrate a complex Hermitian positive definite band matrix and reduce its condition number. (Computational routine.)."},
  {"zpbrfs", "Improves the computed solution to a complex Hermitian positive definite banded system of linear equations AX=B, and provides forward and backward error bounds for the solution. (Computational routine.)."},
  {"zpbstf", "Computes a split Cholesky factorization of a complex Hermitian positive definite band matrix. This routine is designed to be used in conjunction with ZHBGST."},
  {"zpbsv", "Solves a complex Hermitian positive definite banded system of linear equations AX=B. (Simple driver.)."},
  {"zpbsvx", "Solves a complex Hermitian positive definite banded system of linear equations AX=B, and provides an estimate of the condition number and error bounds on the solution. (Expert driver.)."},
  {"zpbtrf", "Computes the Cholesky factorization of a complex Hermitian positive definite band matrix. (Computational routine.)."},
  {"zpbtrs", "Solves a complex Hermitian positive definite banded system of linear equations AX=B, using the Cholesky factorization computed by ZPBTRF. (Computational routine.)."},
  {"zpocon", "Estimates the reciprocal of the condition number of a complex Hermitian positive definite matrix, using the Cholesky factorization computed by SPOTRF/CPOTRF. by SPOTRF/CPOTRF. (Computational routine.)."},
  {"zpoequ", "Computes row and column scalings to equilibrate a complex Hermitian positive definite matrix and reduce its condition number. (Computational routine.)."},
  {"zporfs", "Improves the computed solution to a complex Hermitian positive definite system of linear equations AX=B, and provides forward and backward error bounds for the solution. (Computational routine.)."},
  {"zposv", "Solves a complex Hermitian positive definite system of linear equations AX=B. (Simple driver.)."},
  {"zposvx", "Solves a complex Hermitian positive definite system of linear equations AX=B, and provides an estimate of the condition number and error bounds on the solution. (Expert driver.)."},
  {"zpotrf", "Computes the Cholesky factorization of a complex Hermitian positive definite matrix. (Computational routine.)."},
  {"zpotri", "Computes the inverse of a complex Hermitian positive definite matrix, using the Cholesky factorization computed by ZPOTRF. (Computational routine.)."},
  {"zpotrs", "Solves a complex Hermitian positive definite system of linear equations AX=B, using the Cholesky factorization computed by ZPOTRF. (Computational routine.)."},
  {"zppcon", "Estimates the reciprocal of the condition number of a complex Hermitian positive definite matrix in packed storage, using the Cholesky factorization computed by ZPPTRF. (Computational routine.)."},
  {"zppequ", "Computes row and column scalings to equilibrate a complex Hermitian positive definite matrix in packed storage and reduce its condition number. (Computational routine.)."},
  {"zpprfs", "Improves the computed solution to a complex Hermitian positive definite system of linear equations AX=B, where A is held in packed storage, and provides forward and backward error bounds for the solution. (Computational routine.)."},
  {"zppsv", "Solves a complex Hermitian positive definite system of linear equations AX=B, where A is held in packed storage. (Simple driver.)."},
  {"zppsvx", "Solves a complex Hermitian positive definite system of linear equations AX=B, where A is held in packed storage, and provides an estimate of the condition number and error bounds on the solution. (Expert driver.)."},
  {"zpptrf", "Computes the Cholesky factorization of a complex Hermitian positive definite matrix in packed storage. (Computational routine.)."},
  {"zpptri", "Computes the inverse of a complex Hermitian positive definite matrix in packed storage, using the Cholesky factorization computed by ZPPTRF. (Computational routine.)."},
  {"zpptrs", "Solves a complex Hermitian positive definite system of linear equations AX=B, where A is held in packed storage, using the Cholesky factorization computed by ZPPTRF. (Computational routine.)."},
  {"zptcon", "Computes the reciprocal of the condition number of a complex Hermitian positive definite tridiagonal matrix, using the LDL**H factorization computed by ZPTTRF. (Computational routine.)."},
  {"zpteqr", "Computes all eigenvalues and eigenvectors of a real symmetric positive definite tridiagonal matrix, by computing the SVD of its bidiagonal Cholesky factor. (Computational routine.)."},
  {"zptrfs", "Improves the computed solution to a complex Hermitian positive definite tridiagonal system of linear equations AX=B, and provides forward and backward error bounds for the solution. (Computational routine.)."},
  {"zptsv", "Solves a complex Hermitian positive definite tridiagonal system of linear equations AX=B. (Simple driver.)."},
  {"zptsvx", "Solves a complex Hermitian positive definite tridiagonal system of linear equations AX=B, and provides an estimate of the condition number and error bounds on the solution. (Expert driver.)."},
  {"zpttrf", "Computes the LDL**H factorization of a complex Hermitian positive definite tridiagonal matrix. (Computational routine.)."},
  {"zpttrs", "Solves a complex Hermitian positive definite tridiagonal system of linear equations, using the LDL**H factorization computed by ZPTTRF. (Computational routine.)."},
  {"zspcon", "Estimates the reciprocal of the condition number of a real symmetric/complex symmetric/complex Hermitian indefinite matrix in packed storage, using the factorization computed by ZSPTRF. (Computational routine.)."},
  {"zsprfs", "Improves the computed solution to a real symmetric/complex symmetric/complex Hermitian indefinite system of linear equations AX=B, where A is held in packed storage, and provides forward and backward error bounds for the solution. (Computational routine.)."},
  {"zspsv", "Solves a complex symmetric indefinite system of linear equations AX=B, where A is held in packed storage. (Simple driver.)."},
  {"zspsvx", "Solves a complex symmetric indefinite system of linear equations AX=B, where A is held in packed storage, and provides an estimate of the condition number and error bounds on the solution. (Expert driver.)."},
  {"zsptrf", "Computes the factorization of a real symmetric/complex symmetric/complex Hermitian-indefinite matrix in packed storage, using the diagonal pivoting method. (Computational routine.)."},
  {"zsptri", "Computes the inverse of a real symmetric/complex symmetric/complex Hermitian indefinite matrix in packed storage, using the factorization computed by ZSPTRF. (Computational routine.)."},
  {"zsptrs", "Solves a complex symmetric indefinite system of linear equations AX=B, where A is held in packed storage, using the factorization computed by ZSPTRF. (Computational routine.)."},
  {"zstein", "Computes selected eigenvectors of a real symmetric tridiagonal matrix by inverse iteration. (Computational routine.)."},
  {"zsteqr", "Computes all eigenvalues and eigenvectors of a real symmetric tridiagonal matrix, using the implicit QL or QR algorithm. (Computational routine.)."},
  {"zsycon", "Estimates the reciprocal of the condition number of a real symmetric/complex symmetric/complex Hermitian indefinite matrix, using the factorization computed by ZSYTRF. (Computational routine.)."},
  {"zsyrfs", "Improves the computed solution to a real symmetric/complex symmetric/complex Hermitian indefinite system of linear equations AX=B, and provides forward and backward error bounds for the solution. (Computational routine.)."},
  {"zsysv", "Solves a complex symmetric indefinite system of linear equations AX=B. (Simple driver.)."},
  {"zsysvx", "Solves a complex symmetric indefinite system of linear equations AX=B, and provides an estimate of the condition number and error bounds on the solution. (Expert driver.)."},
  {"zsytrf", "Computes the factorization of a real symmetric/complex symmetric/complex Hermitian-indefinite matrix, using the diagonal pivoting method. (Computational routine.)."},
  {"zsytri", "Computes the inverse of a real symmetric/complex symmetric/complex Hermitian indefinite matrix, using the factorization computed by ZSYTRF. (Computational routine.)."},
  {"zsytrs", "Solves a complex symmetric indefinite system of linear equations AX=B, using the factorization computed by ZSPTRF. (Computational routine.)."},
  {"ztbcon", "Estimates the reciprocal of the condition number of a complex triangular band matrix, in either the 1-norm or the infinity-norm. (Computational routine.)."},
  {"ztbrfs", "Provides forward and backward error bounds for the solution of a complex triangular banded system of linear equations AX=B, A**T X=B or A**H X=B. (Computational routine.)."},
  {"ztbtrs", "Solves a complex triangular banded system of linear equations AX=B, A**T X=B or A**H X=B. (Computational routine.)."},
  {"ztpcon", "Estimates the reciprocal of the condition number of a complex triangular matrix in packed storage, in either the 1-norm or the infinity-norm. (Computational routine.)."},
  {"ztprfs", "Provides forward and backward error bounds for the solution of a complex triangular system of linear equations AX=B, A**T X=B or A**H X=B, where A is held in packed storage. (Computational routine.)."},
  {"ztptri", "Computes the inverse of a complex triangular matrix in packed storage. (Computational routine.)."},
  {"ztptrs", "Solves a complex triangular system of linear equations AX=B, A**T X=B or A**H X=B, where A is held in packed storage. (Computational routine.)."},
  {"ztrcon", "Estimates the reciprocal of the condition number of a complex triangular matrix, in either the 1-norm or the infinity-norm. (Computational routine.)."},
  {"ztrevc", "Computes left and right eigenvectors of a complex upper triangular matrix. (Computational routine.)."},
  {"ztrexc", "Reorders the Schur factorization of a complex matrix by a unitary similarity transformation. (Computational routine.)."},
  {"ztrrfs", "Provides forward and backward error bounds for the solution of a complex triangular system of linear equations A X=B, A**T X=B or A**H X=B. (Computational routine.)."},
  {"ztrsen", "Reorders the Schur factorization of a complex matrix in order to find an orthonormal basis of a right invariant subspace corresponding to selected eigenvalues, and returns reciprocal condition numbers (sensitivities) of the average of the cluster of eigenvalues and of the invariant subspace. (Computational routine.)."},
  {"ztrsna", "Estimates the reciprocal condition numbers (sensitivities) of selected eigenvalues and eigenvectors of a complex upper triangular matrix. (Computational routine.)."},
  {"ztrsyl", "Solves the a complex Sylvester matrix equation A X +/- X B=C where A and B are upper triangular, and may be transposed. (Computational routine.)."},
  {"ztrtri", "Computes the inverse of a complex triangular matrix. (Computational routine.)."},
  {"ztrtrs", "Solves a complex triangular system of linear equations AX=B, A**T X=B or A**H X=B. (Computational routine.)."},
  {"ztzrqf", "Computes an RQ factorization of a complex upper trapezoidal matrix. (Computational routine.)."},
  {"zungbr", "Generates the unitary transformation matrices from a reduction to bidiagonal form determined by CGEBRD. (Computational routine.)."},
  {"zunghr", "Generates the unitary transformation matrix from a reduction to Hessenberg form determined by CGEHRD. (Computational routine.)."},
  {"zunglq", "Generates all or part of the unitary matrix Q from an LQ factorization determined by ZGELQF. (Computational routine.)."},
  {"zungql", "Generates all or part of the unitary matrix Q from a QL factorization determined by ZGEQLF. (Computational routine.)."},
  {"zungqr", "Generates all or part of the unitary matrix Q from a QR factorization determined by ZGEQRF. (Computational routine.)."},
  {"zungrq", "Generates all or part of the unitary matrix Q from an RQ factorization determined by ZGERQF. (Computational routine.)."},
  {"zungtr", "Generates the unitary transformation matrix from a reduction to tridiagonal form determined by CHETRD. (Computational routine.)."},
  {"zunmbr", "Multiplies a complex general matrix by one of the unitary transformation matrices from a reduction to bidiagonal form determined by CGEBRD. (Computational routine.)."},
  {"zunmhr", "Multiplies a complex general matrix by the unitary transformation matrix from a reduction to Hessenberg form determined by ZGEHRD. (Computational routine.)."},
  {"zunmlq", "Multiplies a complex general matrix by the unitary matrix from an LQ factorization determined by ZGELQF. (Computational routine.)."},
  {"zunmql", "Multiplies a complex general matrix by the unitary matrix from a QL factorization determined by ZGEQLF. (Computational routine.)."},
  {"zunmqr", "Multiplies a complex general matrix by the unitary matrix from a QR factorization determined by ZGEQRF. (Computational routine.)."},
  {"zunmrq", "Multiplies a complex general matrix by the unitary matrix from an RQ factorization determined by ZGERQF. (Computational routine.)."},
  {"zunmtr", "Multiplies a complex general matrix by the unitary transformation matrix from a reduction to tridiagonal form determined by ZHETRD. (Computational routine.)."},
  {"zupgtr", "Generates the unitary transformation matrix from a reduction to tridiagonal form determined by CHPTRD. (Computational routine.)."},
  {"zupmtr", "Multiplies a complex general matrix by the unitary transformation matrix from a reduction to tridiagonal form determined by ZHPTRD. (Computational routine.)."},
};

/// List of subroutines from IMSL
map_ss sublib_IMSL {};

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
  {"IBM_SSP", sublib_IBM_SSP},
  {"PDP11_SYSTEM", sublib_PDP11_SYSTEM},
  {"UNIVAC_MATH_PACK",  sublib_UNIVAC_MATH_PACK},
  {"UNIVAC_STAT_PACK",  sublib_UNIVAC_STAT_PACK},
  {"BLAS1",  sublib_BLAS1},
  {"BLAS2",  sublib_BLAS2},
  {"BLAS3",  sublib_BLAS3},
  {"LINPACK",  sublib_LINPACK},
  {"EISPACK",  sublib_EISPACK},
  {"LAPACK",  sublib_LAPACK},
  // {"IMSL",  sublib_IMSL},
  {"ORNL_PH",  sublib_ORNL_PH},
};

// Note: Derive set_default_sublibs from keys of map_default_sublibs
/// Default libraries and routine groups containing target routine names
set_str set_default_sublibs {
  // "NR1",
  "NR2",
  "IBM_SSP",
  "PDP11_SYSTEM",
  "UNIVAC_MATH_PACK",
  "UNIVAC_STAT_PACK",
  "BLAS1",
  "BLAS2",
  "BLAS3",
  "LINPACK",
  "EISPACK",
  "LAPACK",
  // "IMSL",
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

// /**
//  *  \brief Return the lower-case value of a given character
//  *  \param[in] in Character
//  */
// char chartolower(char in) {
//   return std::tolower(in);
// }

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
        // std::transform(exsub.begin(), exsub.end(), exsub.begin(), chartolower);
        FLPR::tolower(exsub);

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

      std::cerr << std::endl << "# Analysis of local routines";
      if (!sscfg.dry_run) {
        if (LocalSubDb.size() > 0) {
          std::cerr << ":" << std::endl;
          std::cerr << "\"Routine\", \"File\", \"Line\", \"Library\"" << std::endl;
          for (const auto& dsub : LocalSubDb) {
            std::cerr << "\"" << dsub.name << "\", "   
                      << "\"" << dsub.filename << "\", "   
                      << dsub.lineno << ", "   
                      << "\"" << dsub.origin
                      << "\"" << std::endl;
          }
        } else {
          std::cerr << " - no locally-defined routines matched external libraries" << std::endl;        
        }
      } else {
        std::cerr << " - disabled for dry-run" << std::endl;        
      }

      std::cerr << std::endl << "# Analysis of called routines";
      if (!sscfg.dry_run) {
        if (CalledSubDb.size() > 0) {
          std::cerr << ":" << std::endl;
          std::cerr << "\"Routine\", \"Parent\", \"File\", \"Line\", \"Library\"" << std::endl;
          for (const auto& csub : CalledSubDb) {
            std::cerr << "\"" << csub.name << "\", "   
                      << "\"" << csub.parent_routine << "\", "
                      << "\"" << csub.filename << "\", "   
                      << csub.lineno << ", "   
                      << "\"" << csub.origin
                      << "\"" << std::endl;
          }
        } else {
          std::cerr << " - no called routines matched external libraries" << std::endl;        
        }
      } else {
        std::cerr << " - disabled for dry-run" << std::endl;        
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
  // std::cerr << "  sf: Scanning " << filename << std::endl;

  File file(filename);
  if (!file)
    return false;

  sscfg.current_file = filename;

  FLPR::Procedure_Visitor puv(file, subtender_procedure);
  bool const scanned = puv.visit();
  // if (changed) {
  //   write_file(std::cout, file);
  // }

  sscfg.current_file = "";

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
  FLPR::tolower(lname);
  // std::transform(lname.begin(), lname.end(), lname.begin(), chartolower);

  // for (auto const& sublib: sscfg.excluded_subs) {
  //   std::cerr << "Excluding: " << sublib << std::endl;
  // }
  int plineno {404};
  
  // There has to be a way to directly get the line number of the first line
  // of the procedure. This works but it's not direct.
  auto proc_top_maybe {proc.crange(Procedure::PROC_BEGIN)};
  for (auto const &stmt : proc_top_maybe) {
    plineno = stmt.stmt_tree().cursor()->token_range.front().start_line;
    break;
  }

  // There has to be a way to directly get the line number of the first line
  // of the procedure but this isn't it...
  // plineno = proc_top_maybe->stmt_tree().cursor()->token_range.front().start_line;

  // std::cerr << "Checking " << lname << " against exclusion list." << std::endl;
  if (sscfg.excluded_subs.count(lname) > 0) {
    // std::cerr << "  sp: Skipping " << proc.name() << " as " << lname << std::endl;
  } else {
    // std::cerr << "  sp: Scanning " << proc.name() << " as " << lname << std::endl;

    for (auto const& sublib: sscfg.scan_libs) {
      if (detect_subroutine(lname, sublib)) {
          // Hit -- Write to Db
          LocalSubDb.emplace_back(LocalSub(lname, sscfg.current_file, plineno, sublib));        
          // std::cerr << "      sp: *** Found " << lname << " on line " << plineno
          //           << " of " << sscfg.current_file << " in library " << sublib << std::endl;
//          std::cerr << "      sp: " << lname << " -> " << lib_it->second << std::endl;
      }
    }
  }

  /* Detect defined and called subroutines, check them against all selected
   * libraries and record matches in the appopriate stats objects
   */

  // Q1: Does proc.name() match any sublib routine names?

  // Set range of proc corresponding to Procedure::EXECUTION_PART
  auto execution_part {proc.crange(Procedure::EXECUTION_PART)};
  
  for (auto const &stmt : execution_part) {
    // Detect called procs and get their names, lcased for scanning
    // Q2: Does called proc name match any sublib routine names? 
    bool found_call {false}; 
    int const stmt_tag = stmt.syntax_tag();
    found_call = (   TAG(SG_CALL_STMT) == stmt_tag
                  || TAG(KW_CALL) == stmt_tag);
    if (found_call) {
      // Get the name of the called subroutine
      auto punkt = stmt.stmt_tree().cursor().down();
      assert(TAG(SG_CALL_STMT) == punkt->syntag);
      punkt.down();
      assert(TAG(KW_CALL) == punkt->syntag);
      punkt.next();
      assert(TAG(SG_PROCEDURE_DESIGNATOR) == punkt->syntag);
      // std::cerr << "Punkt teile: " << punkt->token_range.size() << std::endl;
      assert(punkt->token_range.size() == 1);
      // std::cerr << "Punkt nimt: " << punkt->token_range.front().text() << std::endl;
      // std::cerr << "Was ist punkt: " << punkt->syntag << std::endl;
      std::string lcname {punkt->token_range.front().text()};
      // std::cerr << "      sp: Found subroutine call to " << lcname << std::endl;

      // Drop its case
      FLPR::tolower(lcname);

      // Get line number
      int lineno {punkt->token_range.front().start_line};

//      std::cerr << "Line number? " << stmt.stmt_tree().cursor().down().down().next()->token_range.front().start_line << std::endl;

      for (auto const& sublib: sscfg.scan_libs) {
      // Use detect_subroutine
        if (detect_subroutine(lcname, sublib)) {
            // Hit -- Write to Db
            CalledSubDb.emplace_back(CalledSub(lcname, sscfg.current_file, lname, lineno, sublib));        
            // std::cerr << "      sp: *** Found " << lcname << " in library " << sublib 
            //           << ", called from " << lname << std::endl;
//          std::cerr << "      sp: " << lname << " -> " << lib_it->second << std::endl;
        }
      }
    }
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
