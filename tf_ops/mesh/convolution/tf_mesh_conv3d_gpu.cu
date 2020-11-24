// numInterior: (NfIn)
// intrplWgts:  (NfIn*maxK, 3)
// input:       (NfIn*maxK, C)
// filter:      (3, C, r)
// output:      (NfIn, C*r)
__global__ void facet2facet_conv3d_forward(int NfIn, int C, int r, const int* numInterior,
                                           const float* intrplWgts, const float* input,
                                           const float* filter, float* output)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x; // thread index
    int fcIdx = idx/(C*r);  // global face index in the batch
    int cout = idx%(C*r);  // output channel ID
    int cin = cout/r;     // input channel ID

    if(fcIdx<NfIn)
    {
        int kStart = 0;
        int kEnd = numInterior[fcIdx];
        if (fcIdx>0) kStart = numInterior[fcIdx-1];

        // convolution
        int K = kEnd - kStart;
        for(int k=kStart;k<kEnd;k++)
        {
            // get interpolation weights (w1,w2,w3) related to (v1->v2->v3) of the face
            float w1=intrplWgts[k*3], w2=intrplWgts[k*3+1], w3=intrplWgts[k*3+2];

            float weight = w1*filter[cout] + w2*filter[cout+C*r] + w3*filter[cout+2*C*r];
            output[fcIdx*C*r+cout] += weight*input[k*C+cin];
        }
         output[fcIdx*C*r+cout] /= K;
    }
}


// numInterior: (NfIn)
// intrplWgts:  (NfIn*maxK, 3)
// filter:      (3, C, r)
// gradOutput:  (NfIn, C*r)
// gradInput:   (NfIn*maxK, C)
__global__ void facet2facet_input_backward(int NfIn, int C, int r, const int* numInterior,
                                           const float* intrplWgts, const float* filter,
                                           const float* gradOutput, float* gradInput)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x; // thread index
    int fcIdx = idx/(C*r);  // global face index in the batch
    int cout = idx%(C*r);  // output channel ID
    int cin = cout/r;     // input channel ID

    if(fcIdx<NfIn)
    {
        int kStart = 0;
        int kEnd = numInterior[fcIdx];
        if (fcIdx>0) kStart = numInterior[fcIdx-1];

        // convolution
        int K = kEnd - kStart;
        for(int k=kStart;k<kEnd;k++)
        {
            // get interpolation weights (w1,w2,w3) related to (v1->v2->v3) of the face
            float w1=intrplWgts[k*3], w2=intrplWgts[k*3+1], w3=intrplWgts[k*3+2];

            float weight = w1*filter[cout] + w2*filter[cout+C*r] + w3*filter[cout+2*C*r];
            float derIn = gradOutput[fcIdx*C*r+cout]*weight/K;
            atomicAdd(&gradInput[k*C+cin], derIn);
        }
    }
}


// numInterior: (NfIn)
// intrplWgts:  (NfIn*maxK, 3)
// input:       (NfIn*maxK, C)
// gradOutput:  (NfIn, C*r)
// gradFilter:  (3, C, r)
__global__ void facet2facet_filter_backward(int NfIn, int C, int r, const int* numInterior,
                                            const float* intrplWgts, const float* input, const float* gradOutput,
                                            float* gradFilter, int sharedMemSize, int startIdx)
{
    extern __shared__ float gradPerBlock[]; // the gradient on each block
    for (int i=threadIdx.x;i<sharedMemSize;i+=blockDim.x)
    {
        gradPerBlock[i] = 0; // for 1D block
    }
    __syncthreads();

    int idx = blockIdx.x*blockDim.x + threadIdx.x; // thread index
    int fcIdx = idx/(C*r);  // global face index in the batch
    int cout = idx%(C*r);  // output channel ID
    int cin = cout/r;     // input channel ID

    int endIdx = sharedMemSize+startIdx;
    if(fcIdx<NfIn)
    {
        int kStart = 0;
        int kEnd = numInterior[fcIdx];
        if (fcIdx>0) kStart = numInterior[fcIdx-1];

        // convolution
        int K = kEnd - kStart;
        float derFilt[3] = {0,0,0};
        for(int k=kStart;k<kEnd;k++)
        {
            // get interpolation weights (w1,w2,w3) related to (v1->v2->v3) of the face
            float w1=intrplWgts[k*3], w2=intrplWgts[k*3+1], w3=intrplWgts[k*3+2];

            float temp = gradOutput[fcIdx*C*r+cout]*input[k*C+cin]/K;
            derFilt[0] += temp*w1; derFilt[1] += temp*w2; derFilt[2] += temp*w3;
        }
        for(int m=0;m<3;m++)
        {
            int currIdx = m*C*r+cout;
            if((currIdx>=startIdx) && (currIdx<endIdx)) // within the shared memory
            {
               atomicAdd(&gradPerBlock[currIdx-startIdx],derFilt[m]);
            }
        }
    }
    __syncthreads();

    for (int i=threadIdx.x;i<sharedMemSize;i+=blockDim.x)
    {
        atomicAdd(&gradFilter[i+startIdx],gradPerBlock[i]); // for 1D block
    }
}
void facet2facetConv3dLauncher(int NfIn, int C, int r, const int* numInterior, const float* intrplWgts,
                               const float* input, const float* filter, float* output)
{
    int numGrid = NfIn*C*r/1024 + 1;
    facet2facet_conv3d_forward<<<numGrid,1024>>>(NfIn, C, r, numInterior, intrplWgts,
                                                 input, filter, output);

}
void facet2facetConv3dGradLauncher(int NfIn, int C, int r, const int* numInterior, const float* intrplWgts,
                                   const float* input, const float* filter, const float* gradOutput,
                                   float* gradInput, float* gradFilter)
{
    int numGrid = NfIn*C*r/1024 + 1;
    facet2facet_input_backward<<<numGrid,1024>>>(NfIn, C, r, numInterior, intrplWgts, filter,
                                                 gradOutput, gradInput);


    // titan xp has shared memory of 49152 bytes, each float value takes 4 bytes in the memory
    int maxSharedMemSize = int(49152/sizeof(float));
    int maxIter = (3*C*r)/maxSharedMemSize;
    int remainder = (3*C*r)%maxSharedMemSize;
    for(int iter=0;iter<maxIter;iter++)
    {
        facet2facet_filter_backward<<<numGrid,1024,sizeof(float)*maxSharedMemSize>>>(NfIn, C, r, numInterior,
                                                                            intrplWgts, input, gradOutput, gradFilter,
                                                                            maxSharedMemSize, maxSharedMemSize*iter);
    }
    if(remainder>0) // fill the remainder
    {
        facet2facet_filter_backward<<<numGrid,1024,sizeof(float)*remainder>>>(NfIn, C, r, numInterior,
                                                                        intrplWgts, input, gradOutput, gradFilter,
                                                                        remainder, maxSharedMemSize*maxIter);
    }
}


// numInterval: (NfIn)
// face:    (NfIn, 3)
// input:   (NvIn, C)
// filter:  (3, C, r)
// output:  (NfIn, C*r)
__global__ void vertex2facet_conv3d_forward(int NfIn, int C, int r, const int* numInterval, const int* face,
                                      const float* input, const float* filter, float* output)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x; // thread index
    int fcIdx = idx/(C*r);  // global face index in the batch
    int cout = idx%(C*r);   // output channel ID
    int cin = cout/r;       // input channel ID

    if(fcIdx<NfIn)
    {
        int v1=face[3*fcIdx], v2=face[3*fcIdx+1], v3=face[3*fcIdx+2];

        // convolution, ensure that output has been all initialized to zeros
        int intervalSize = numInterval[fcIdx];
        float step   = 1.0/float(intervalSize);
        int numInterior = (intervalSize+1)*(intervalSize+2)/2; // number of interior points to interpolate
        for(int k1=0;k1<=intervalSize;k1++)
        {
            for(int k2=0;k2<=intervalSize-k1;k2++)
            {
                float w1 = k1*step;
                float w2 = k2*step;
                float w3 = 1 - w1 - w2;

                // use vertex features, (x,y,z), (nx,ny,nz) can be already concatenated into
                float weight = w1*filter[cout]    + w2*filter[cout+C*r] + w3*filter[cout+2*C*r];
                float feat   = w1*input[v1*C+cin] + w2*input[v2*C+cin]  + w3*input[v3*C+cin];
                output[fcIdx*C*r+cout] += weight*feat;
            }
        }
        output[fcIdx*C*r+cout] /= numInterior;
    }
}

// numInterval:    (NfIn)
// face:       (NfIn, 3)
// filter:     (3, C, r)
// gradOutput: (NfIn, C*r)
// gradInput:  (NvIn, C)
__global__ void vertex2facet_input_backward(int NfIn, int C, int r, const int* numInterval, const int* face,
                                      const float* filter, const float* gradOutput, float* gradInput)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x; // thread index
    int fcIdx = idx/(C*r);  // global face index in the batch
    int cout = idx%(C*r);  // output channel ID
    int cin = cout/r;     // input channel ID

    if(fcIdx<NfIn)
    {
        int v1=face[3*fcIdx], v2=face[3*fcIdx+1], v3=face[3*fcIdx+2];
        float derIn[3] = {0,0,0};

        int intervalSize = numInterval[fcIdx];
        float step   = 1.0/float(intervalSize);
        int numInterior = (intervalSize+1)*(intervalSize+2)/2; // number of interior points to interpolate
        for(int k1=0;k1<=intervalSize;k1++)
        {
            for(int k2=0;k2<=intervalSize-k1;k2++)
            {
                float w1=k1*step, w2=k2*step, w3=1-w1-w2;

                // use vertex features, (x,y,z), (nx,ny,nz) can be already concatenated into
                float weight = w1*filter[cout] + w2*filter[cout+C*r] + w3*filter[cout+2*C*r];
                derIn[0] += weight*w1; derIn[1] += weight*w2; derIn[2] += weight*w3;
            }
        }
        // gradient accumulation from all adjacent faces
        atomicAdd(&gradInput[v1*C+cin], gradOutput[fcIdx*C*r+cout]*derIn[0]/numInterior);
        atomicAdd(&gradInput[v2*C+cin], gradOutput[fcIdx*C*r+cout]*derIn[1]/numInterior);
        atomicAdd(&gradInput[v3*C+cin], gradOutput[fcIdx*C*r+cout]*derIn[2]/numInterior);
    }
}

// numInterval:    (NfIn)
// face:       (NfIn, 3)
// input:      (NvIn, C)
// gradOutput: (NfIn, C*r)
// gradFilter: (3, C, r)
__global__ void vertex2facet_filter_backward(int NfIn, int C, int r, const int* numInterval, const int* face,
                                       const float* input, const float* gradOutput, float* gradFilter,
                                       int sharedMemSize, int startIdx)
{
    extern __shared__ float gradPerBlock[]; // the gradient on each block
    for (int i=threadIdx.x;i<sharedMemSize;i+=blockDim.x)
    {
        gradPerBlock[i] = 0; // for 1D block
    }
    __syncthreads();

    int idx   = blockIdx.x*blockDim.x + threadIdx.x; // thread index
    int fcIdx = idx/(C*r);  // global face index in the batch
    int cout  = idx%(C*r);  // output channel ID
    int cin   = cout/r;     // input channel ID

    int endIdx = sharedMemSize+startIdx;
    if(fcIdx<NfIn)
    {
        int v1=face[3*fcIdx], v2=face[3*fcIdx+1], v3=face[3*fcIdx+2];

        // convolution, ensure that output has been all initialized to zeros
        int intervalSize = numInterval[fcIdx];    // number of interior points to sample
        float step   = 1.0/float(intervalSize);
        int numInterior = (intervalSize+1)*(intervalSize+2)/2;
        float derFilt[3] = {0,0,0};
        for(int k1=0;k1<=intervalSize;k1++)
        {
            for(int k2=0;k2<=intervalSize-k1;k2++)
            {
                float w1 = k1*step;
                float w2 = k2*step;
                float w3 = 1 - w1 - w2;

                // use vertex features, (x,y,z), (nx,ny,nz) can be already concatenated into
                float feat = w1*input[v1*C+cin] + w2*input[v2*C+cin] + w3*input[v3*C+cin];
                derFilt[0] += w1*feat; derFilt[1] += w2*feat; derFilt[2] += w3*feat;
            }
        }
        for(int m=0;m<3;m++)
        {
            int currIdx = m*C*r+cout;
            if((currIdx>=startIdx) && (currIdx<endIdx)) // within the shared memory
            {
               atomicAdd(&gradPerBlock[currIdx-startIdx],gradOutput[fcIdx*C*r+cout]*derFilt[m]/numInterior);
            }
        }
    }
    __syncthreads();

    for (int i=threadIdx.x;i<sharedMemSize;i+=blockDim.x)
    {
        atomicAdd(&gradFilter[i+startIdx],gradPerBlock[i]); // for 1D block
    }
}
void vertex2facetConv3dLauncher(int NfIn, int C, int r, const int* numInterval, const int* face,
                          const float* input, const float* filter, float* output)
{
    int numGrid = NfIn*C*r/1024 + 1;
    vertex2facet_conv3d_forward<<<numGrid,1024>>>(NfIn, C, r, numInterval, face, input, filter, output);
}
void vertex2facetConv3dGradLauncher(int NfIn, int C, int r, const int* numInterval, const int* face,
                              const float* input, const float* filter, const float* gradOutput,
                              float* gradInput, float* gradFilter)
{
    int numGrid = NfIn*C*r/1024 + 1;
    vertex2facet_input_backward<<<numGrid,1024>>>(NfIn, C, r, numInterval, face, filter, gradOutput, gradInput);

    // titan xp has shared memory of 49152 bytes, each float value takes 4 bytes in the memory
    int maxSharedMemSize = int(49152/sizeof(float));
    int maxIter = (3*C*r)/maxSharedMemSize;
    int remainder = (3*C*r)%maxSharedMemSize;
    for(int iter=0;iter<maxIter;iter++)
    {
        vertex2facet_filter_backward<<<numGrid,1024,sizeof(float)*maxSharedMemSize>>>(NfIn, C, r, numInterval, face,
                                                                                input, gradOutput, gradFilter,
                                                                                maxSharedMemSize, maxSharedMemSize*iter);
    }
    if(remainder>0) // fill the remainder
    {
        vertex2facet_filter_backward<<<numGrid,1024,sizeof(float)*remainder>>>(NfIn, C, r, numInterval, face,
                                                                         input, gradOutput, gradFilter,
                                                                         remainder, maxSharedMemSize*maxIter);
    }
}

// vtMap:   NvIn; // only non-negative mapid got output features
// nfCount: NvIn;
// face:    NfIn*3;
// coeff:   NfIn*K;
// input:   NfIn*C;
// filter:  K*C*r;
// output:  NvOut*(C*r);
__global__ void facet2vertex_conv3d_forward(int NfIn, int C, int r, const int K, const int* vtMap,
                                    const int* nfCount, const int* face, const float* coeff,
                                    const float* input, const float* filter, float* output)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x; // thread index
    int fcIdx = idx/(C*r);  // global face index in the batch
    int cout = idx%(C*r);   // output channel ID
    int cin = cout/r;       // input channel ID

    if (fcIdx<NfIn) // index must be in the legal range
    {
        // a fuzzy combined weights
        float weight = 0;
        for(int k=0;k<K;k++)
        {
            float xi_k = coeff[fcIdx*K+k];
            weight += xi_k*filter[k*C*r+cout];
        }
        float out_feat = weight*input[fcIdx*C+cin];

        int v[3] = {face[3*fcIdx], face[3*fcIdx+1], face[3*fcIdx+2]};
        for(int k=0;k<3;k++)          // aggregate context of vertex from adjacent faces
        {
            int vi = v[k];
            int vo = vtMap[vi];       // for non-strided convolution, we have vtMap[vi]=vi.
            int nfSize = nfCount[vo]; //nfSize is the number of adjacent faces to vi, try nfSize=1 for no averaging
            if (vo>=0)
                atomicAdd(&output[vo*C*r+cout], out_feat/nfSize);
        }
    }
}

// vtMap:      NvIn;   // only non-negative mapid got output features
// nfCount:    NvIn;
// face:       NfIn*3;
// coeff:      NfIn*K;
// filter:     K*C*r;
// gradOutput: NvOut*(C*r)
// gradInput:  NfIn*C;
__global__ void facet2vertex_input_backward(int NfIn, int C, int r, const int K, const int* vtMap,
                                    const int* nfCount, const int* face, const float* coeff,
                                    const float* filter, const float* gradOutput, float* gradInput)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x; // thread index
    int fcIdx = idx/(C*r);  // global face index in the batch
    int cout = idx%(C*r);  // output channel ID
    int cin = cout/r;     // input channel ID

    if (fcIdx<NfIn)     // index must be in the legal range
    {
        // a fuzzy combined weights
        float weight = 0;
        for(int k=0;k<K;k++)
        {
            float xi_k = coeff[fcIdx*K+k];
            weight += xi_k*filter[k*C*r+cout];
        }

        // gradInput is on faces, each face collect gradients from three vertices
        // better no atomic addition
        int v[3] = {face[3*fcIdx], face[3*fcIdx+1], face[3*fcIdx+2]};
        for(int k=0;k<3;k++)    // aggregate context of vertex from adjacent faces
        {
            int vi = v[k];
            int vo = vtMap[vi];
            int nfSize = nfCount[vo];
            if (vo>=0)
            {
                float derIn = gradOutput[vo*C*r+cout]*weight/nfSize;
                atomicAdd(&gradInput[fcIdx*C+cin], derIn);
            }
        }
    }
}

// vtMap:      NvIn;   // only non-negative mapid got output features
// nfCount:    NvIn;
// face:       NfIn*3;
// coeff:      NfIn*K;
// input:      NfIn*C;
// gradOutput: NvOut*(C*r)
// gradFilter: K*C*r;
__global__ void facet2vertex_filter_backward(int NfIn, int C, int r, const int K, const int* vtMap,
                                     const int* nfCount, const int* face, const float* coeff,
                                     const float* input, const float* gradOutput,
                                     float* gradFilter, int sharedMemSize, int startIdx)
{
    extern __shared__ float gradPerBlock[]; // the gradient on each block
    for (int i=threadIdx.x;i<sharedMemSize;i+=blockDim.x)
    {
        gradPerBlock[i] = 0; // for 1D block
    }
    __syncthreads();

    int idx = blockIdx.x*blockDim.x + threadIdx.x; // thread index
    int fcIdx = idx/(C*r);  // global face index in the batch
    int cout = idx%(C*r);  // output channel ID
    int cin = cout/r;     // input channel ID

    int endIdx = sharedMemSize+startIdx;
    if (fcIdx<NfIn)     // index must be in the legal range
    {
        int v[3] = {face[3*fcIdx], face[3*fcIdx+1], face[3*fcIdx+2]};

        for(int k=0;k<K;k++)
        {
            int currIdx = k*C*r+cout;
            if((currIdx>=startIdx) && (currIdx<endIdx)) // within the shared memory
            {
                float derFilt = coeff[fcIdx*K+k]*input[fcIdx*C+cin];
                for(int m=0;m<3;m++)
                {
                    int vi = v[m];
                    int vo = vtMap[vi];
                    int nfSize = nfCount[vo];
                    if (vo>=0)
                        atomicAdd(&gradPerBlock[currIdx-startIdx], gradOutput[vo*C*r+cout]*derFilt/nfSize);
                }
            }
        }
    }
    __syncthreads();

    for (int i=threadIdx.x;i<sharedMemSize;i+=blockDim.x)
    {
        atomicAdd(&gradFilter[i+startIdx],gradPerBlock[i]); // for 1D block
    }
}

// vtMap:      NvIn;
// nfCount:    NvIn;
// face:       NfIn*3;
// input:      NfIn*C;
// filter:     K*C*r;
// gradOutput: NvOut*(C*r)
// gradCoeff:  NfIn*K;
__global__ void facet2vertex_coeff_backward(int NfIn, int C, int r, const int K, const int* vtMap,
                                    const int* nfCount, const int* face, const float* input,
                                    const float* filter, const float* gradOutput, float* gradCoeff)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x; // thread index
    int fcIdx = idx/(C*r);  // global face index in the batch
    int cout = idx%(C*r);  // output channel ID
    int cin = cout/r;     // input channel ID

    if (fcIdx<NfIn) // index must be in the legal range
    {
        int v[3] = {face[3*fcIdx], face[3*fcIdx+1], face[3*fcIdx+2]};

        for(int k=0;k<K;k++)
        {
            float derCoeff = filter[k*C*r+cout]*input[fcIdx*C+cin];
            for(int m=0;m<3;m++)
            {
                int vi = v[m];
                int vo = vtMap[vi];
                int nfSize = nfCount[vo];
                if (vo>=0)
                    atomicAdd(&gradCoeff[fcIdx*K+k], gradOutput[vo*C*r+cout]*derCoeff/nfSize);
            }
        }
    }
}
void facet2vertexConv3dLauncher(int NfIn, int C, int r, int K, const int* vtMap, const int* nfCount, const int* face,
                        const float* coeff, const float* input, const float* filter, float* output)
{
    int numGrid = NfIn*C*r/1024 + 1;
    facet2vertex_conv3d_forward<<<numGrid,1024>>>(NfIn, C, r, K, vtMap, nfCount, face, coeff, input, filter, output);
}
void facet2vertexConv3dGradLauncher(int NfIn, int C, int r, int K, const int* vtMap, const int* nfCount, const int* face,
                            const float* coeff, const float* input, const float* filter, const float* gradOutput,
                            float* gradInput, float* gradFilter, float* gradCoeff)
{
    int numGrid = NfIn*C*r/1024 + 1;
    facet2vertex_input_backward<<<numGrid,1024>>>(NfIn, C, r, K, vtMap, nfCount, face, coeff, filter, gradOutput, gradInput);

    // titan xp has shared memory of 49152 bytes, each float value takes 4 bytes in the memory
    int maxSharedMemSize = int(49152/sizeof(float));

    int maxIter = (K*C*r)/maxSharedMemSize;
    int remainder = (K*C*r)%maxSharedMemSize;
    for(int iter=0;iter<maxIter;iter++)
    {
        facet2vertex_filter_backward<<<numGrid,1024,sizeof(float)*maxSharedMemSize>>>(NfIn, C, r, K, vtMap, nfCount, face,
                                                                              coeff, input, gradOutput, gradFilter,
                                                                              maxSharedMemSize, maxSharedMemSize*iter);
    }
    if(remainder>0) // fill the remainder
    {
        facet2vertex_filter_backward<<<numGrid,1024,sizeof(float)*remainder>>>(NfIn, C, r, K, vtMap, nfCount, face,
                                                                       coeff, input, gradOutput, gradFilter,
                                                                       remainder, maxSharedMemSize*maxIter);
    }

    facet2vertex_coeff_backward<<<numGrid,1024>>>(NfIn, C, r, K, vtMap, nfCount, face, input, filter, gradOutput, gradCoeff);
}


