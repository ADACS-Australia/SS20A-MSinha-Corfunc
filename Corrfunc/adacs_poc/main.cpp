#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>
#include <sys/time.h>
#include "tools.h"
#include "computeCL.h"

int main(int argc, char **argv) {

	// Position information
	double *xpos, *ypos, *zpos;
	// Bin data
	double rmin, rmax;
	double *rupp;
	int nbins;
	// A timer
	struct timeval start, end;
	// Error
	int error = 0;

	// Check input
	if (argc != 3) {
		printf("Wrong number of arguments. Exiting\n");
		return 1;
	}

	// Read file information
	long Npart = read_ascii_file(argv[1], &xpos, &ypos, &zpos);

	// Load bucket data. Ignore its returned status.
	setup_bins_double(argv[2], &rmin, &rmax, &nbins, &rupp);

	// Generate weights
	double *weights;
	int Nweights = 4;
	setup_weights(&weights, (int)Npart, Nweights); // Fake weights tool

	// Start timer
	gettimeofday(&start, NULL);

	// ==============================================================================================
	// ===========================   Main demonstration section =====================================
	double precision = 1000.0;	// Set to -1.0 for exact, otherwise 100 for two decimals, 1000 for three decimals etc.
	int debug = 1;
	// Computed Results
	double *rpavg;
	long *npairs;
	size_t Nsize = 128;
	error = CorrelateOpenCL(xpos, ypos, zpos, rupp, weights, &rpavg, &npairs, rmin, rmax, (int)Npart, (int)nbins, Nweights, Nsize, precision, debug);
	printf("Executed Entire Command Queue using OpenCL: Error Code = %d\n", error);
	// ==============================================================================================
	// ==============================================================================================

	// Check the histogram
	double rlow = rupp[0];
	printf("--Results--\n");
	for (int i = 1; i < nbins; i++) {
		printf("%.4g\t%.4g\t rpavg[%d] = %g, npairs[%d] = %ld\n", rlow, rupp[i], i, rpavg[i], i, npairs[i]);
		rlow=rupp[i];
	}

	// Check the weights
	double sum = 0.0;
	for (int i = 0; i < Npart; i++) {
		double w = 0.0;
		int index = i*Nweights;
		for (int j = 0; j < Nweights; j++) {
			w += weights[index + j]*weights[index + j];
		}
		sum += w;
	}
	printf("Sum = %g\n", sum);

	// End timer
	gettimeofday(&end, NULL);

	// Report the time
	int elapsed = ((end.tv_sec - start.tv_sec)*1000000) + (end.tv_usec - start.tv_usec);
	double elapsed_seconds = (double)elapsed/1000000.0;
	printf("Time = %g seconds\n", elapsed_seconds);

	free(xpos);	free(ypos);	free(zpos);
	free(rupp);	free(npairs); free(rpavg);
	free(weights);

	return 0;
}


