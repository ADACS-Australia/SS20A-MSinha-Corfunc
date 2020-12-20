#include "computeCL.h"
#include "tools.h"

// For convenience of logging
int DEBUG;

// Logging tools 
int LOGINFO(const char* message);
int LOGERROR(const char* message);
// BAIL must return void, since exit returns void.
void BAIL(const char *message, const int error_code);

// These functions are internally declared and are not visible to the external users.
int SetClDevice();
int FinalizeCl();
int InitCl(size_t bytes, size_t binbytes, size_t pairsbytes, size_t weightsbytes);
int RunCl(int n, size_t localSize);
int CollectClResult(double **rpavg, int64_t **npairs, int64_t **rpavg_avg, size_t bytes, size_t binbytes, size_t pairsbytes);
int BuildCl(double **xpos, double **ypos, double **zpos, double **rupp, double **weights, size_t bytes, size_t binbytes, double precision, double rmin, double rmax, int n, int nbins, int nweights);

// OpenCL kernel.
// Notes:
//    This implementation is common, but not guaranteed to work.
//    This is because OpenCL does not enforce global/local memory 
//    consistency across all work items and groups.
//    Rather than getting the actual *val from global, it might
//    be a local copy in cache. 
const char *kernelSource =                                      "\n" \
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable                   \n" \
"#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable     \n" \
"#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable\n" \
"void atomic_add_double(__global double *val, double delta) {    \n" \
"    union { double f; ulong i } new, old;                       \n" \
"    // Manually check the atomic exchange                       \n" \
"    do {                                                        \n" \
"        old.f = *val;                                           \n" \
"        new.f = old.f + delta;                                  \n" \
"    } while (atom_cmpxchg((volatile __global ulong*)val, old.i, new.i) != old.i); \n" \
"}                                                               \n" \
"__kernel void vecAdd(  __global double *a,                      \n" \
"                       __global double *b,                      \n" \
"                       __global double *c,                      \n" \
"                       __global double *d,                      \n" \
"                       __global double *w,                      \n" \
"                       __global double *rpavg,                  \n" \
"                       __global long *npairs,                   \n" \
"                       __global long *rpavg_approx,             \n" \
"                       const double precision,                  \n" \
"                       const double rmin,                       \n" \
"                       const double rmax,                       \n" \
"                       const unsigned int n,                    \n" \
"                       const unsigned int nbins,                \n" \
"                       const unsigned int nweights)             \n" \
"{                                                               \n" \
"    //Get our global thread ID                                  \n" \
"    int id = get_global_id(0);                                  \n" \
"    double dx, dy, dz;                                          \n" \
"    double x, y, z, r2, r;                                      \n" \
"    double weight;                                              \n" \
"    int index_A, index_B;                                       \n" \
"    int i;                                                      \n" \
"    long r_approx;                                              \n" \
"    const double sqr_rmin = rmin*rmin;                          \n" \
"    const double sqr_rmax = rmax*rmax;                          \n" \
"    //Make sure we do not go out of bounds                      \n" \
"    if (id < n) {                                               \n" \
"        x = a[id]; y = b[id]; z = c[id];                        \n" \
"        index_A = id*nweights;                                  \n" \
"        for (i = 0; i < n; i++) {                               \n" \
"            dx = x - a[i];                                      \n" \
"            dy = y - b[i];                                      \n" \
"            dz = z - c[i];                                      \n" \
"            r2 = dx*dx + dy*dy + dz*dz;                         \n" \
"            if(r2 < sqr_rmin || r2 >= sqr_rmax) continue;       \n" \
"            r = sqrt(r2);                                       \n" \
"            // Calculate weighted R                             \n" \
"            index_B = i*nweights;                               \n" \
"            weight = 0.0;                                       \n" \
"            for (int j = 0; j < nweights; j++) {                \n" \
"                weight += w[index_A+j]*w[index_B+j];            \n" \
"            }                                                   \n" \
"            // Apply weight                                     \n" \
"            r = weight*r;                                       \n" \
"            if (precision == -1.0) {                            \n" \
"                for (int kbin=nbins-1; kbin>=1; kbin--) {       \n" \
"                    if (r >= d[kbin-1])  {                      \n" \
"                        atom_inc(&npairs[kbin]);                \n" \
"                        atomic_add_double(&rpavg[kbin], r);     \n" \
"                        break;                                  \n" \
"                    }                                           \n" \
"                }                                               \n" \
"            } else {                                            \n" \
"                r_approx = (long)(r*1000.0);                    \n" \
"                for (int kbin=nbins-1; kbin>=1; kbin--) {       \n" \
"                    if (r >= d[kbin-1])  {                      \n" \
"                        atom_inc(&npairs[kbin]);                \n" \
"                        atom_add(&rpavg_approx[kbin], r_approx);\n" \
"                        break;                                  \n" \
"                    }                                           \n" \
"                }                                               \n" \
"            }                                                   \n" \
"        }                                                       \n" \
"    }                                                           \n" \
"}                                                               \n" \
                                                                "\n" ;

// We are using a single device here on a single thread - this means there is no 
// reason we can't use global memory safely to store OpenCL context details.
cl_platform_id cpPlatform;	// Platform
cl_context context; 		// context
cl_command_queue queue; 	// Command queue
cl_program program;			// Program
cl_kernel kernel;			// Kernel
cl_device_id device_id;		// Device ID

// Create global cl_mem's 
cl_mem d_xpos, d_ypos, d_zpos;
cl_mem d_rupp;
cl_mem d_npairs, d_rpavg;
cl_mem d_rpavg_approx;
cl_mem d_weights;

int CorrelateOpenCL(double *xpos, double *ypos, double *zpos, double *rupp, double *weights, double **rpavg,
					long **npairs, double rmin, double rmax, int Npart, int nbins, int Nweights, size_t Nsize, double precision, int debug) {

	int error = 0;
	size_t size = (int)Npart*sizeof(double);
	size_t sizebins = (int)nbins*sizeof(double);
	size_t sizenpairs = nbins*sizeof(long);
	size_t sizeweights = (int)Npart*Nweights*sizeof(double);
	// Allocate some variables for internal use
	double *rpavg_i = (double*)malloc(sizebins);
	long *npairs_i = (long*)malloc(sizenpairs);
	long *rpavg_approx_i = (long*)malloc(sizenpairs);        // Our approx version
	double *rpavg_approx_d_i = (double*)malloc(sizenpairs);  // Double version of approx version

	// Set the level of debugging
	DEBUG = debug;

	LOGINFO(" Running OpenCL wrapper function");

	// Set the OpenCL device. Currently hard coded to use a GPU device.
	error += SetClDevice();

	// Init CL
	InitCl(size, sizebins, sizenpairs, sizeweights);

	// Build the instruction pipeline
	// This involves copying data across
	error += BuildCl(&xpos, &ypos, &zpos, &rupp, &weights, size, sizebins, precision, rmin, rmax, (int)Npart, (int)nbins, Nweights);

	// Launch
	error += RunCl(Npart, Nsize);

	// Collect the result
	error += CollectClResult(&rpavg_i, &npairs_i, &rpavg_approx_i, size, sizebins, sizenpairs);

	// Wrap up CL
	error += FinalizeCl();

	// Normalize as required
	for (int i = 1; i < nbins; i++) {
		double avg = 0.0; double approx = 0.0;
		if (npairs_i[i] > 0) {
			avg = (double)rpavg_i[i]/npairs_i[i];
			rpavg_i[i] = avg;
			approx = (double)rpavg_approx_i[i]/(precision*npairs_i[i]);
			rpavg_approx_d_i[i] = approx;
		}
	}

	// Return the relevant result
	*npairs = npairs_i;
	if (precision == -1.0) {
		*rpavg = rpavg_i;
		free(rpavg_approx_d_i);
	} else {
		*rpavg = rpavg_approx_d_i;
		free(rpavg_i);
	}

	// Need to free the long form of the approximate version
	free(rpavg_approx_i);

	if (error == 0) {
		LOGINFO(" OpenCL wrapper function finished");
	} else {
		LOGERROR(" OpenCL wrapper function failure. View logs for more information");
	}

	return error;
}


int InitCl(size_t bytes, size_t binbytes, size_t pairsbytes, size_t weightsbytes) {
	cl_int err;
	int error = 0;
	char buffer[1024];
	// Set the context
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	if (err != 0) {
		sprintf(buffer, "Error encountered while creating OpenCL context: OpenCL error code = %d", err);
		LOGERROR(buffer);
		error = 1;
	} else {
		LOGINFO(" (SUCCESS) Created the OpenCL context");
	}
	// Create the command queue
	queue = clCreateCommandQueue(context, device_id, 0, &err);
	if (err != 0) {
		sprintf(buffer, "Error encountered while creating OpenCL command queue: OpenCL error code = %d", err);
		LOGERROR(buffer);
		error = 1;
	} else {
		LOGINFO(" (SUCCESS) Created the OpenCL command queue");
	}
	// Allocate memory for positions
	d_xpos = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
	d_ypos = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
	d_zpos = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
	// Allocate memory for bin data
	d_rupp = clCreateBuffer(context, CL_MEM_READ_ONLY, binbytes, NULL, NULL);
	// Allocate memory for histgram data
	d_rpavg = clCreateBuffer(context, CL_MEM_READ_ONLY, binbytes, NULL, NULL);
	d_npairs = clCreateBuffer(context, CL_MEM_READ_ONLY, pairsbytes, NULL, NULL);
	d_rpavg_approx = clCreateBuffer(context, CL_MEM_READ_ONLY, pairsbytes, NULL, NULL);
	// Allocate memory for weights
	d_weights = clCreateBuffer(context, CL_MEM_READ_ONLY, weightsbytes, NULL, NULL);

	LOGINFO(" (SUCCESS) Allocated OpenCL memory");

	return error;
}

int BuildCl(double **xpos, double **ypos, double **zpos, double **rupp, double **weights, size_t bytes, 
			size_t binbytes, double precision, double rmin, double rmax, int n, int nbins, int nweights) {

	cl_int err;
	int error = 0;
	char buffer[1024];
	size_t weightbytes = n*nweights*sizeof(double);

	// Create the compute program from the source buffer
	program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, NULL, &err);
	if (err != 0) {
		sprintf(buffer, "Error encountered while compiling kernel from source: OpenCL error code = %d", err);
		LOGERROR(buffer);
		error = 1;
	} else {
		LOGINFO(" (SUCCESS) Compiled kernel from source");
	}

	// Build the program executable
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != 0) {
		sprintf(buffer, "Error encountered while building program: OpenCL error code = %d", err);
		LOGERROR(buffer);
		error = 1;
	} else {
		LOGINFO(" (SUCCESS) Built OpenGL Program");
	}

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "vecAdd", &err);
	if (err != 0) {
		sprintf(buffer, "Error encountered while creating kernel: OpenCL error code = %d", err);
		LOGERROR(buffer);
		error = 1;
	} else {
		LOGINFO(" (SUCCESS) Created OpenCL kernel");
	}

	// Copy the positions across across
	err = clEnqueueWriteBuffer(queue, d_xpos, CL_TRUE, 0, bytes, *xpos, 0, NULL, NULL);
	err += clEnqueueWriteBuffer(queue, d_ypos, CL_TRUE, 0, bytes, *ypos, 0, NULL, NULL);
	err += clEnqueueWriteBuffer(queue, d_zpos, CL_TRUE, 0, bytes, *zpos, 0, NULL, NULL);
	// Copy the bin data across
	err += clEnqueueWriteBuffer(queue, d_rupp, CL_TRUE, 0, binbytes, *rupp, 0, NULL, NULL);
	// Copy the weight information across
	err += clEnqueueWriteBuffer(queue, d_weights, CL_TRUE, 0, weightbytes, *weights, 0, NULL, NULL);

	if (err != 0) {
		LOGERROR("Error while copying data to GPU");
		error = 1;
	} else {
		LOGINFO(" (SUCCESS) Copied data to the GPU");
	}

	// Register the arguments for the vecAdd kernel
	int index = 0;
	err = clSetKernelArg(kernel, index, sizeof(cl_mem), &d_xpos); index++;
	err += clSetKernelArg(kernel, index, sizeof(cl_mem), &d_ypos); index++;
	err += clSetKernelArg(kernel, index, sizeof(cl_mem), &d_zpos); index++;
	err += clSetKernelArg(kernel, index, sizeof(cl_mem), &d_rupp); index++;
	err += clSetKernelArg(kernel, index, sizeof(cl_mem), &d_weights); index++;
	err += clSetKernelArg(kernel, index, sizeof(cl_mem), &d_rpavg); index++;
	err += clSetKernelArg(kernel, index, sizeof(cl_mem), &d_npairs); index++;
	err += clSetKernelArg(kernel, index, sizeof(cl_mem), &d_rpavg_approx); index++;
	err += clSetKernelArg(kernel, index, sizeof(double), &precision); index++;
	err += clSetKernelArg(kernel, index, sizeof(double), &rmin); index++;
	err += clSetKernelArg(kernel, index, sizeof(double), &rmax); index++;
	err += clSetKernelArg(kernel, index, sizeof(unsigned int), &n); index++;
	err += clSetKernelArg(kernel, index, sizeof(unsigned int), &nbins); index++;
	err += clSetKernelArg(kernel, index, sizeof(unsigned int), &nweights); index++;

	if (err != 0) {
		LOGERROR("Error found while processing kernel arguments");
		error = 1;
	} else {
		LOGINFO(" (SUCCESS) Processed kernel arguments");
	}

	return error;
}

int RunCl(int n, size_t localSize) {
	size_t globalSize;
	char buffer[1024];
	cl_int err;
	int error = 0;

	// Number of work items in each local work group
	// Number of total work items - localSize must be devisor
	globalSize = ceil(n/(double)localSize)*localSize;
	
	if ((int)globalSize <= 0) {
		// This is a serious error. This is normally due to a mistake with n, our problem size, in this case
		// the number of particles to correlate.
		sprintf(buffer, "Error preparing OpenCL work items: Number of work items = %d", (int)globalSize);
		LOGERROR(buffer);
		error = 1;
	} else {
		sprintf(buffer, " (SUCCESS) Prepared kernel with %d threads and %d work items", (int)localSize, (int)globalSize);
		LOGINFO(buffer);
	}
	// Execute the kernel over the entire range of the data set 
	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
	if (err != 0) {
		sprintf(buffer, "Error encountered while executing command queue: OpenCL error code = %d", err);
		LOGERROR(buffer);
		error = 1;
	} else {
		LOGINFO(" (SUCCESS) Executed OpenCL command queue");
	}

	// Let it complete before attempting to rescue some data
	err = clFinish(queue);
	if (err != 0) {
		sprintf(buffer, "Error encountered while finishing work queue: OpenCL error code = %d", err);
		LOGERROR(buffer);
		error = 1;
	} else {
		LOGINFO(" (SUCCESS) Completed OpenCL command queue");
	}

	return error;
}

int CollectClResult(double **rpavg, long **npairs, long **rpavg_approx, size_t bytes, size_t binbytes, size_t pairsbytes) {
	cl_int err;
	int error = 0;
	// Copy the result 
	err = clEnqueueReadBuffer(queue, d_rpavg, CL_TRUE, 0, binbytes, *rpavg, 0, NULL, NULL );
	err += clEnqueueReadBuffer(queue, d_npairs, CL_TRUE, 0, pairsbytes, *npairs, 0, NULL, NULL );
	err += clEnqueueReadBuffer(queue, d_rpavg_approx, CL_TRUE, 0, pairsbytes, *rpavg_approx, 0, NULL, NULL );
	if (err != 0) {
		LOGERROR("Error found while copying results from the GPU.");
		error = 1;
	} else {
		LOGINFO(" (SUCCESS) Copied data from the GPU");
	}
	return error;
}

int FinalizeCl() {
	// Free device memory
	cl_int err;
	int error = 0;
	char buffer[1024];

	err = clReleaseMemObject(d_xpos);
	err += clReleaseMemObject(d_ypos);
	err += clReleaseMemObject(d_zpos);
	err += clReleaseMemObject(d_rupp);
	err += clReleaseMemObject(d_rpavg);
	err += clReleaseMemObject(d_npairs);
	err += clReleaseMemObject(d_rpavg_approx);
	err += clReleaseMemObject(d_weights);

	if (err != 0) {
		sprintf(buffer, "Error encountered while freeing GPU memory on one or more items");
		LOGERROR(buffer);
		error = 1;
	} else {
		LOGINFO(" (SUCCESS) Freed OpenCL memory");
	}
	err = clReleaseKernel(kernel);
	err += clReleaseProgram(program);
	// Release the command queue
	err += clReleaseCommandQueue(queue);
	err += clReleaseContext(context);

	if (err != 0) {
		sprintf(buffer, "Error encountered while shutting down OpenCL command queue or context");
		LOGERROR(buffer);
		error = 1;
	} else {
		LOGINFO(" (SUCCESS) Freed OpenCL queue and context");
	}

	return error;
}

int SetClDevice() {
	// Investigate opencl platforms present
	cl_platform_id* PlatformIDs;
	cl_uint NumPlatforms;
	cl_uint ChosenDevice;
	char cBuffer[1024];
	char cInfo[1024];
	cl_uint NvPlatform;
	clGetPlatformIDs(0, NULL, &NumPlatforms);
	PlatformIDs = new cl_platform_id[NumPlatforms];
	bool GPUFound = false;
	char buffer[64];
	size_t param_value_size = 1024*sizeof(char);
	cl_int err;

	err = clGetPlatformIDs(NumPlatforms, PlatformIDs, NULL);
	sprintf(buffer, " (STATUS) Detected %d OpenCL platforms", NumPlatforms); 
	LOGINFO(buffer);

	for (cl_uint i = 0; i < NumPlatforms; ++i) {
		err += clGetPlatformInfo(PlatformIDs[i], CL_PLATFORM_NAME, 1024, cBuffer, NULL);
		if (strstr(cBuffer, "NVIDIA") != NULL) {
				// Found a suitable GPU
				GPUFound = true;
				ChosenDevice = i;
				sprintf(buffer, " (STATUS) Detected on platform %d: %s", i, cBuffer);
				LOGINFO(buffer);
		}
	}

	if (GPUFound) {
		// Finalize the chosen device
		err += clGetDeviceIDs(PlatformIDs[ChosenDevice], CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
		// Get the device name for the user
		err += clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(cBuffer), cBuffer, NULL);
	} else {
		BAIL("Failed to find suitable device. Exiting.", 1);
		return 1; // Most likely unused, given that BAIL will kill the program
	}
	
	// One last check on OpenCL errors
	if (err != 0) {
		LOGERROR("Found GPU device; However several OpenCL errors were encountered");
		return 1;
	} else {
		LOGINFO(" (SUCCESS) Selected OpenCL GPU device");
		sprintf(buffer, " (SUCCESS) Selected device: %s", cBuffer);
		LOGINFO(buffer);
	}
	// No errors
	return 0;
}

int LOGINFO(const char* message) {
	if (DEBUG == 1) return printf("INFO:\t%s\n", message);
}

int LOGERROR(const char* message) {
	return printf("ERROR:\t%s\n",message);
}

void BAIL(const char* message, const int error_code) {
	LOGERROR(message);
	exit(error_code);
}

