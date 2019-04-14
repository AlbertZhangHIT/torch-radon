#include <TH/TH.h>
#define MAXX(x,y) ((x) > (y) ? (x) : (y))  

void incrementRadon(float *pr, float pixel, float r)   
{   
    int r1;   
    float delta;   
   
    r1 = (int) r;   
    delta = r - r1;   
    pr[r1] += pixel * (1.0 - delta);   
    pr[r1+1] += pixel * delta;   
}   

static void    
radonckernel(float *pPtr, float *iPtr, float *thetaPtr, int M, int N,    
      int xOrigin, int yOrigin, int numAngles, int rFirst, int rSize)   
{   
    int k, m, n;              /* loop counters */   
    float angle;             /* radian angle value */   
    float cosine, sine;      /* cosine and sine of current angle */   
    float *pr;               /* points inside output array */   
    float *pixelPtr;         /* points inside input array */   
    float pixel;             /* current pixel value */   
    float *ySinTable, *xCosTable;   
    /* tables for x*cos(angle) and y*sin(angle) */   
    float x,y;   
    float r, delta;   
    int r1;   
   
    /* Allocate space for the lookup tables */   
    xCosTable = new float[2*N];
    ySinTable = new float[2*M]; 
   
    for (k = 0; k < numAngles; k++) {   
        angle = thetaPtr[k];   
        pr = pPtr + k*rSize;  /* pointer to the top of the output column */   
        cosine = cos(angle);    
        sine = sin(angle);      
   
        /* Radon impulse response locus:  R = X*cos(angle) + Y*sin(angle) */   
        /* Fill the X*cos table and the Y*sin table.                      */   
        /* x- and y-coordinates are offset from pixel locations by 0.25 */   
        /* spaced by intervals of 0.5. */   
        for (n = 0; n < N; n++)   
        {   
            x = n - xOrigin;   
            xCosTable[2*n]   = (x - 0.25)*cosine;   
            xCosTable[2*n+1] = (x + 0.25)*cosine;   
        }   
        for (m = 0; m < M; m++)   
        {   
            y = yOrigin - m;   
            ySinTable[2*m] = (y - 0.25)*sine;   
            ySinTable[2*m+1] = (y + 0.25)*sine;   
        }   
   
        pixelPtr = iPtr;   
        for (n = 0; n < N; n++)   
        {   
            for (m = 0; m < M; m++)   
            {   
                pixel = *pixelPtr++;   
                if (pixel != 0.0)   
                {   
                    pixel *= 0.25;   
   
                    r = xCosTable[2*n] + ySinTable[2*m] - rFirst;   
                    incrementRadon(pr, pixel, r);   
   
                    r = xCosTable[2*n+1] + ySinTable[2*m] - rFirst;   
                    incrementRadon(pr, pixel, r);   
   
                    r = xCosTable[2*n] + ySinTable[2*m+1] - rFirst;   
                    incrementRadon(pr, pixel, r);   
   
                    r = xCosTable[2*n+1] + ySinTable[2*m+1] - rFirst;   
                    incrementRadon(pr, pixel, r);   
                }   
            }   
        }   
    }   
    
    delete [] xCosTable;
    delete [] ySinTable;             
}  

void radonc(float* Img, float* theta, int M, int N, int m, int n, float* P, float* R)
{
    int numAngles;          /* number of theta values */   
    float *thetaPtr;       /* pointer to theta values in radians */   
    float *pr1, *pr2;      /* float pointers used in loop */   
    float deg2rad;         /* conversion factor */   
    float temp;            /* temporary theta-value holder */   
    int k;                  /* loop counter */    
    int xOrigin, yOrigin;   /* center of image */   
    int temp1, temp2;       /* used in output size computation */   
    int rFirst, rLast;      /* r-values for first and last row of output */   
    int rSize;              /* number of rows in output */  

    /* Get THETA values */   
    deg2rad = 3.14159265358979 / 180.0;  
    numAngles = m * n;
    thetaPtr = new float[numAngles];
    for (k = 0; k < numAngles; k++)
        *(thetaPtr++) = *(theta++) * deg2rad;

    /* Where is the coordinate system's origin? */   
    xOrigin = MAXX(0, (N-1)/2);   
    yOrigin = MAXX(0, (M-1)/2);   
    /* How big will the output be? */   
    temp1 = M - 1 - yOrigin;   
    temp2 = N - 1 - xOrigin;   
    rLast = (int) ceil(sqrt((float) (temp1*temp1+temp2*temp2))) + 1;   
    rFirst = -rLast;   
    rSize = rLast - rFirst + 1;      

    for (k = rFirst; k <= rLast; k++)
        *(R++) = (float) k;

    /* Invoke main computation routines */   
    radonckernel(P, Img, thetaPtr, M, N, xOrigin, yOrigin, numAngles, rFirst, rSize);

}

int cpu_radon(THFloatTensor * P, THFloatTensor * R, THFloatTensor * img, THFloatTensor * theta)
{
    // Image size
    int M = THFloatTensor_size(img, 0);
    int N = THFloatTensor_size(img, 1);
    int m = THFloatTensor_size(theta, 0);
    int n = THFloatTensor_size(theta, 1);

    float * P_flat = THFloatTensor_data(P);
    float * R_flat = THFloatTensor_data(R);
    float * img_flat = THFloatTensor_data(img);
    float * theta_flat = THFloatTensor_data(theta);

    radonc(img_flat, theta_flat, M, N, m, n, P_flat, R_flat);
}