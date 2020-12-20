// Function declarations

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>

int64_t getnumlines(const char *fname,const char comment);
int64_t read_ascii_file(const char *filename, double **xpos, double **ypos, double **zpos);
int setup_bins_double(const char *fname,double *rmin,double *rmax,int *nbin,double **rupp);
int setup_weights(double **weights, int Npart, int Nweights);