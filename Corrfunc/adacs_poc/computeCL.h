// Header
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/opencl.h>
#include <string.h>

// Computation of Correlation using OpenCL
int CorrelateOpenCL(double *xpos, double *ypos, double *zpos, double *rupp, double *weights, double **rpavg, long **npairs, 
					double rmin, double rmax, int Npart, int nbins, int Nweights, size_t Nsize, double precision, int debug);
