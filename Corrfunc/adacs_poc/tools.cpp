// Tools.cpp
#include "tools.h"

int64_t getnumlines(const char *fname,const char comment)
{
    FILE *fp= NULL;
    const int MAXLINESIZE = 10000;
    int64_t nlines=0;
    char str_line[MAXLINESIZE];

    fp = fopen(fname,"rt");
    if(fp == NULL) {
        fprintf(stderr,"Error: Could not open file `%s'\n", fname);
        perror(NULL);
        return -1;
    }

    while(1){
    if(fgets(str_line, MAXLINESIZE,fp)!=NULL) {
            //WARNING: this does not remove white-space. You might
            //want to implement that (was never an issue for me)
            if(str_line[0] !=comment)
                nlines++;
        } else
            break;
    }
    fclose(fp);
    return nlines;
}

int64_t read_ascii_file(const char *filename, double **xpos, double **ypos, double **zpos)
{
    int64_t numlines = getnumlines(filename, '#');
    if(numlines <= 0) return numlines;

    double *x = (double*)calloc(numlines, sizeof(*x));
    double *y = (double*)calloc(numlines, sizeof(*y));
    double *z = (double*)calloc(numlines, sizeof(*z));
    if(x == NULL || y == NULL || z == NULL) {
        free(x);free(y);free(z);
        fprintf(stderr,"Error: Could not allocate memory for %" PRId64 " elements for the (x/y/z) arrays\n", numlines);
        perror(NULL);
        return -1;
    }

    FILE *fp = fopen(filename, "rt");
    if(fp == NULL) {
        fprintf(stderr,"Error:Could not open file `%s' in function %s\n", filename, __FUNCTION__);
        fprintf(stderr,"This is strange because the function `getnumlines' successfully counted the number of lines in that file\n");
        fprintf(stderr,"Did that file (`%s') just get deleted?\n", filename);
        perror(NULL);
        return -1;
    }

    int64_t index=0;
    const int nitems = 3;
    const int MAXLINESIZE = 10000;
    char buf[MAXLINESIZE];
    while(1) {
    if(fgets(buf,MAXLINESIZE,fp)!=NULL) {
            int nread=sscanf(buf,"%lf %lf %lf",&x[index], &y[index], &z[index]);
            if(nread==nitems) {
                index++;
            }
    } else {
            break;
        }
    }
    fclose(fp);
    if(index != numlines) {
        fprintf(stderr,"Error: There are supposed to be `%'" PRId64 " lines of data in the file\n", numlines);
        fprintf(stderr,"But could only parse `%'" PRId64 " lines containing (x y z) data\n", index);
        fprintf(stderr,"exiting...\n");
        return -1;
    }
    *xpos = x;
    *ypos = y;
    *zpos = z;
    return numlines;
}

int setup_weights(double **weights, int Npart, int Nweights)
{
    int error = 0;
    double *w = (double*)calloc(Npart*Nweights, sizeof(*w));
    int index;
    if(w == NULL) {
        free(w);
        printf("Error: Could not allocate memory for %d particles and %d weights", Npart, Nweights);
        error = 1;
    } else {
        // Set weights to make values equal to the current benchmarks
        double fake_weight = sqrt(1.0/Nweights);
        for (int i = 0; i < Npart; i++) {
            for (int j = 0; j < Nweights; j++) {
				index = i*Nweights + j;
                w[index] = fake_weight;
            }
        }
    }
    *weights = w;
    return error;
}



int setup_bins_double(const char *fname,double *rmin,double *rmax,int *nbin,double **rupp)
{
    //set up the bins according to the binned data file
    //the form of the data file should be <rlow  rhigh ....>
    const int MAXBUFSIZE=1000;
    char buf[MAXBUFSIZE];
    FILE *fp=NULL;
    double low,hi;
    const char comment='#';
    const int nitems=2;
    int nread=0;
    *nbin = ((int) getnumlines(fname,comment))+1;
    *rupp = (double*)calloc(*nbin+1, sizeof(double));
    if(rupp == NULL) {
        fprintf(stderr,"Error: Could not allocate memory for %d bins to store the histogram limits\n", *nbin+1);
        perror(NULL);
        return EXIT_FAILURE;
    }
    fp = fopen(fname,"rt");
    if(fp == NULL) {
        free(*rupp);
        fprintf(stderr,"Error: Could not open file `%s'..exiting\n",fname);
        perror(NULL);
        return EXIT_FAILURE;
    }

    int index=1;
    while(1) {
        if(fgets(buf,MAXBUFSIZE,fp)!=NULL) {
            nread=sscanf(buf,"%lf %lf",&low,&hi);
            if(nread==nitems) {

                if(index==1) {
                    *rmin=low;
                    (*rupp)[0]=low;
                }

                (*rupp)[index] = hi;
                index++;
            }
        } else {
            break;
        }
    }
    *rmax = (*rupp)[index-1];
    fclose(fp);

    (*rupp)[*nbin]=*rmax ;
    (*rupp)[*nbin-1]=*rmax ;

    return EXIT_SUCCESS;
}


