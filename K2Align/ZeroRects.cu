#include "Functions.cuh"

__global__ void ZeroRectsKernel(tfloat* d_input, char* d_output, int3 dims, int3 rectdims);

bool d_HasZeroRects(tfloat* d_input, int3 dims, int3 rectdims)
{
	char* d_result = (char*)CudaMallocValueFilled(Elements(dims), (char)0);

	int TpB = NextMultipleOf(min(192, dims.x - rectdims.x), 32);
	dim3 grid = dim3((dims.x + TpB - 1) / TpB, dims.y - rectdims.y);
	ZeroRectsKernel <<<grid, TpB>>> (d_input, d_result, dims, rectdims);

	char* d_max;
	cudaMalloc((void**)&d_max, sizeof(char));
	d_Max(d_result, d_max, Elements(dims));

	char h_max = 0;
	cudaMemcpy(&h_max, d_max, sizeof(char), cudaMemcpyDeviceToHost);

	cudaFree(d_result);
	cudaFree(d_max);

	return h_max > 0;
}

__global__ void ZeroRectsKernel(tfloat* d_input, char* d_output, int3 dims, int3 rectdims)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= dims.x - rectdims.x)
		return;

	for(int y = blockIdx.y; y < min(blockIdx.y + rectdims.y, dims.y); y++)
		for(int x = idx; x < min(idx + rectdims.x, dims.x); x++)
			if(d_input[getOffset(x, y, dims.x)] != (tfloat)0)
			{
				d_output[getOffset(idx, blockIdx.y, dims.x)] = 0;
				return;
			}

	d_output[getOffset(idx, blockIdx.y, dims.x)] = 1;
}