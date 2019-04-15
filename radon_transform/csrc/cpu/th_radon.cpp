
#include "cpu/th_radon.h"

#define MAXX(x,y) ((x) > (y) ? (x) : (y))  

template <typename T>
void incrementRadon(T *pr, T pixel, T r)   
{   
    int r1;   
    T delta;   
   
    r1 = (int) r;   
    delta = r - r1;   
    pr[r1] += pixel * (1.0 - delta);   
    pr[r1+1] += pixel * delta;   
}   

template <typename T>
void radonckernel(const T *pPtr, const T *iPtr, const T *thetaPtr, const int M, const int N,    
      const int xOrigin, const int yOrigin, const int numAngles, const int rFirst, const int rSize)   
{   
    int k, m, n;              /* loop counters */   
    T angle;             /* radian angle value */   
    T cosine, sine;      /* cosine and sine of current angle */   
    const T *pixelPtr;         /* points inside input array */   
    T pixel;             /* current pixel value */   
    T *ySinTable, *xCosTable;   
    /* tables for x*cos(angle) and y*sin(angle) */   
    T x,y;   
    T r;    
   
    /* Allocate space for the lookup tables */   
    xCosTable = new T[2*N];
    ySinTable = new T[2*M]; 
   
    for (k = 0; k < numAngles; k++) {   
        angle = thetaPtr[k];   
        const T *pr = pPtr + k*rSize;  /* pointer to the top of the output column */   
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

void radonc(float* Img, float* theta, int M, int N, int m, int n, float* P)
{
    int numAngles;          /* number of theta values */   
    float *thetaPtr;       /* pointer to theta values in radians */   
    float deg2rad;         /* conversion factor */   
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

    /* Invoke main computation routines */   
    radonckernel(P, Img, thetaPtr, M, N, xOrigin, yOrigin, numAngles, rFirst, rSize);

}

at::Tensor radon_cpu(const at::Tensor& input,
                    const at::Tensor& theta) {
    AT_ASSERTM(!input.type().is_cuda(), "input must be a CPU tensor");
    AT_ASSERTM(!theta.type().is_cuda(), "input must be a CPU tensor");

    auto M = input.size(0);
    auto N = input.size(1);

    auto xOrigin = MAXX(0, (N-1)/2);   
    auto yOrigin = MAXX(0, (M-1)/2);      
    auto temp1 = M - 1 - yOrigin;   
    auto temp2 = N - 1 - xOrigin;  
    auto rLast = std::ceil(std::sqrt((float) (temp1*temp1+temp2*temp2))) + 1;
    auto rFirst = -rLast;
    auto rSize = rLast - rFirst + 1;   
    auto numAngles = theta.numel();

    auto radon_img = at::zeros({rSize, numAngles}, input.options());

    if (radon_img.numel() == 0) {
        return radon_img;
    }

    AT_DISPATCH_FLOATING_TYPES(input.type(), "radon", [&] {
      radonckernel<scalar_t>(
        radon_img.data<scalar_t>(),
        input.data<scalar_t>(),
        theta.data<scalar_t>(),
        M, N, xOrigin, yOrigin, numAngles, rFirst, rSize
        );
    });
    return radon_img;
}
