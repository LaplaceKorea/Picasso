__global__ void sum_weight(const int NvIn, const float* wgtIn, const int* vtReplace,
                           const int* vtMap, float* wgtSum)
{
    int vi = blockIdx.x*blockDim.x + threadIdx.x;

    if (vi<NvIn)
    {
        int vc = vi;
        if (vtReplace[vi]<0)
            vc = -vtReplace[vi];

        int vo = vtMap[vc];
        if(vo>=0) // exist in output
        {
            atomicAdd(&wgtSum[vo], wgtIn[vi]);
        }
    }
}
__global__ void normalize_weight(const int NvIn, const float* wgtIn, const float* wgtSum,
                                 const int* vtReplace, const int* vtMap, float* wgtOut)
{
    int vi = blockIdx.x*blockDim.x + threadIdx.x;

    if (vi<NvIn)
    {
        int vc = vi;
        if (vtReplace[vi]<0)
            vc = -vtReplace[vi];

        int vo = vtMap[vc];
        if(vo>=0) // exist in output
        {
            wgtOut[vi] = wgtIn[vi]/(wgtSum[vo]+2e-16);
        }
    }
}
void normalizeWeightLauncher(const int NvIn, const float* wgtIn, const int* vtReplace,
                             const int* vtMap, float* wgtOut)
{
    int numGrid = int(NvIn/1024) + 1;

    float* wgtSum;
    cudaMalloc(&wgtSum, NvIn*sizeof(float));
    cudaMemset(wgtSum, 0, NvIn*sizeof(float));
    sum_weight<<<numGrid,1024>>>(NvIn, wgtIn, vtReplace, vtMap, wgtSum);
    normalize_weight<<<numGrid,1024>>>(NvIn, wgtIn, wgtSum, vtReplace, vtMap, wgtOut);
    cudaFree(wgtSum);
}