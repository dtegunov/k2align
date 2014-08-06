#include "../../GTOM/Prerequisites.cuh"
#include "../../GTOM/Functions.cuh"
#include "Functions.cuh"

void ReadEM(string path, void** data, EM_DATATYPE &datatype, int nframe)
{
	char* header = (char*)malloc(512 * sizeof(char));

	ifstream inputfile(path, ios::in|ios::binary|ios::ate);
	inputfile.seekg(0, ios::beg);
	inputfile.read(header, 512 * sizeof(char));
	
	int3 dims;
	dims.x = ((int*)header)[1];
	dims.y = ((int*)header)[2];
	dims.z = 1;
	datatype = (EM_DATATYPE)header[3];

	free(header);

	size_t bytesperfield = 1;
	if(datatype == EM_DATATYPE::EM_SHORT)
		bytesperfield = 2;
	else if(datatype == EM_DATATYPE::EM_LONG || datatype == EM_DATATYPE::EM_SINGLE)
		bytesperfield = 4;
	else if(datatype == EM_DATATYPE::EM_SINGLECOMPLEX || datatype == EM_DATATYPE::EM_DOUBLE)
		bytesperfield = 8;
	else if(datatype == EM_DATATYPE::EM_DOUBLECOMPLEX)
		bytesperfield = 16;

	size_t datasize = Elements(dims) * bytesperfield;
	cudaMallocHost(data, datasize);

	inputfile.seekg(datasize * nframe, ios::cur);

	inputfile.read((char*)*data, datasize);

	inputfile.close();
}

void ReadMRC(string path, void** data, EM_DATATYPE &datatype, int nframe, bool flipx)
{
	char* header = (char*)malloc(1024 * sizeof(char));

	ifstream inputfile(path, ios::in|ios::binary);
	inputfile.seekg(0, ios::beg);
	//#pragma omp critical
	inputfile.read(header, 1024 * sizeof(char));

	int mode = ((int*)header)[3];
	if(mode == 0)
		datatype = EM_DATATYPE::EM_BYTE;
	else if(mode == 1)
		datatype = EM_DATATYPE::EM_SHORT;
	else if(mode == 2)
		datatype = EM_DATATYPE::EM_SINGLE;
	else if(mode == 3)
		datatype = EM_DATATYPE::EM_SHORTCOMPLEX;
	else if(mode == 4)
		datatype = EM_DATATYPE::EM_SINGLECOMPLEX;

	size_t bytesperfield = 1;
	if(datatype == EM_DATATYPE::EM_SHORT)
		bytesperfield = 2;
	else if(datatype == EM_DATATYPE::EM_SHORTCOMPLEX || datatype == EM_DATATYPE::EM_LONG || datatype == EM_DATATYPE::EM_SINGLE)
		bytesperfield = 4;
	else if(datatype == EM_DATATYPE::EM_SINGLECOMPLEX || datatype == EM_DATATYPE::EM_DOUBLE)
		bytesperfield = 8;
	else if(datatype == EM_DATATYPE::EM_DOUBLECOMPLEX)
		bytesperfield = 16;

	int extendedsize = header[23];
	void* extendedheader = malloc(extendedsize * sizeof(char));
	inputfile.read((char*)extendedheader, extendedsize * sizeof(char));
	
	int3 dims;
	dims.x = ((int*)header)[0];
	dims.y = ((int*)header)[1];
	dims.z = 1;
	size_t datasize = Elements(dims) * bytesperfield;
	cudaMallocHost(data, datasize);

	inputfile.seekg(datasize * nframe, ios::cur);

	inputfile.read((char*)*data, datasize);

	inputfile.close();
	free(header);
	free(extendedheader);

	if(!flipx)
		return;

	size_t layersize = dims.x * dims.y;
	size_t linewidth = dims.x;
	void* flipbuffer = malloc(linewidth * bytesperfield);
	size_t offsetlayer, offsetrow;
	int dimsxminusone = dims.x - 1;

	for(int z = 0; z < dims.z; z++)
	{
		offsetlayer = z * layersize;
		if(datatype == EM_DATATYPE::EM_BYTE)
			for (int y = 0; y < dims.y; y++)
			{
				offsetrow = offsetlayer + y * linewidth;
				memcpy(flipbuffer, ((char*)*data) + offsetrow, linewidth);
				for (int x = 0; x < dims.x; x++)
					((char*)*data)[offsetrow + x] = ((char*)flipbuffer)[dimsxminusone - x];
			}		
		else if(datatype == EM_DATATYPE::EM_SHORT)
			for (int y = 0; y < dims.y; y++)
			{
				offsetrow = offsetlayer + y * linewidth;
				memcpy(flipbuffer, ((short*)*data) + offsetrow, linewidth);
				for (int x = 0; x < dims.x; x++)
					((short*)*data)[offsetrow + x] = ((short*)flipbuffer)[dimsxminusone - x];
			}		
		else if(datatype == EM_DATATYPE::EM_SHORTCOMPLEX || datatype == EM_DATATYPE::EM_LONG || datatype == EM_DATATYPE::EM_SINGLE)
			for (int y = 0; y < dims.y; y++)
			{
				offsetrow = offsetlayer + y * linewidth;
				memcpy(flipbuffer, ((float*)*data) + offsetrow, linewidth);
				for (int x = 0; x < dims.x; x++)
					((float*)*data)[offsetrow + x] = ((float*)flipbuffer)[dimsxminusone - x];
			}		
		else if(datatype == EM_DATATYPE::EM_SINGLECOMPLEX || datatype == EM_DATATYPE::EM_DOUBLE)
			for (int y = 0; y < dims.y; y++)
			{
				offsetrow = offsetlayer + y * linewidth;
				memcpy(flipbuffer, ((double*)*data) + offsetrow, linewidth);
				for (int x = 0; x < dims.x; x++)
					((double*)*data)[offsetrow + x] = ((double*)flipbuffer)[dimsxminusone - x];
			}		
	}

	free(flipbuffer);
}

void ReadRAW(string path, void** data, EM_DATATYPE datatype, int3 dims, int nframe, size_t headerbytes)
{
	ifstream inputfile(path, ios::in|ios::binary|ios::ate);
	inputfile.seekg(0, ios::beg);

	size_t bytesperfield = 1;
	if(datatype == EM_DATATYPE::EM_SHORT)
		bytesperfield = 2;
	else if(datatype == EM_DATATYPE::EM_LONG || datatype == EM_DATATYPE::EM_SINGLE)
		bytesperfield = 4;
	else if(datatype == EM_DATATYPE::EM_SINGLECOMPLEX || datatype == EM_DATATYPE::EM_DOUBLE)
		bytesperfield = 8;
	else if(datatype == EM_DATATYPE::EM_DOUBLECOMPLEX)
		bytesperfield = 16;

	size_t datasize = Elements(toInt3(dims.x, dims.y, 1)) * bytesperfield;
	cudaMallocHost(data, datasize);

	inputfile.seekg(headerbytes + datasize * nframe, ios::cur);

	inputfile.read((char*)*data, datasize);

	inputfile.close();
}

void EMToDeviceTfloat(void* h_input, tfloat* d_output, EM_DATATYPE datatype, size_t elements)
{
	if(datatype == EM_DATATYPE::EM_BYTE)
		CudaMemcpyFromHostArrayConverted<char, tfloat>((char*)h_input, d_output, elements);
	else if(datatype == EM_DATATYPE::EM_SHORT)
		CudaMemcpyFromHostArrayConverted<short, tfloat>((short*)h_input, d_output, elements);
	else if(datatype == EM_DATATYPE::EM_LONG)
		CudaMemcpyFromHostArrayConverted<int, tfloat>((int*)h_input, d_output, elements);
	else if(datatype == EM_DATATYPE::EM_SINGLE)
		cudaMemcpy(d_output, h_input, elements * sizeof(tfloat), cudaMemcpyHostToDevice);
	else if(datatype == EM_DATATYPE::EM_DOUBLE)
		CudaMemcpyFromHostArrayConverted<double, tfloat>((double*)h_input, d_output, elements);
	else
		throw;
}

tfloat* EMToDeviceTfloat(void* h_input, EM_DATATYPE datatype, size_t elements)
{
	tfloat* d_output;

	if(datatype == EM_DATATYPE::EM_BYTE)
		CudaMallocFromHostArrayConverted<char, tfloat>((char*)h_input, &d_output, elements);
	else if(datatype == EM_DATATYPE::EM_SHORT)
		CudaMallocFromHostArrayConverted<short, tfloat>((short*)h_input, &d_output, elements);
	else if(datatype == EM_DATATYPE::EM_LONG)
		CudaMallocFromHostArrayConverted<int, tfloat>((int*)h_input, &d_output, elements);
	else if(datatype == EM_DATATYPE::EM_SINGLE)
	{
		cudaMalloc((void**)&d_output, elements * sizeof(tfloat));
		cudaMemcpy(d_output, h_input, elements * sizeof(tfloat), cudaMemcpyHostToDevice);
	}
	else if(datatype == EM_DATATYPE::EM_DOUBLE)
		CudaMallocFromHostArrayConverted<double, tfloat>((double*)h_input, &d_output, elements);
	else
		throw;

	return d_output;
}

tfloat* EMToHostTfloat(void* h_input, EM_DATATYPE datatype, size_t elements)
{
	tfloat* h_output;
	cudaMallocHost((void**)&h_output, elements * sizeof(tfloat));

	if(datatype == EM_DATATYPE::EM_BYTE)
		#pragma omp parallel for schedule(dynamic, 1024)
		for(intptr_t i = 0; i < elements; i++)
			h_output[i] = (tfloat)((char*)h_input)[i];
	else if(datatype == EM_DATATYPE::EM_SHORT)
		#pragma omp parallel for schedule(dynamic, 1024)
		for(intptr_t i = 0; i < elements; i++)
			h_output[i] = (tfloat)((short*)h_input)[i];
	else if(datatype == EM_DATATYPE::EM_LONG)
		#pragma omp parallel for schedule(dynamic, 1024)
		for(intptr_t i = 0; i < elements; i++)
			h_output[i] = (tfloat)((int*)h_input)[i];
	else if(datatype == EM_DATATYPE::EM_SINGLE)
		#pragma omp parallel for schedule(dynamic, 1024)
		for(intptr_t i = 0; i < elements; i++)
			h_output[i] = (tfloat)((float*)h_input)[i];
	else if(datatype == EM_DATATYPE::EM_DOUBLE)
		#pragma omp parallel for schedule(dynamic, 1024)
		for(intptr_t i = 0; i < elements; i++)
			h_output[i] = (tfloat)((double*)h_input)[i];
	else
		throw;
	
	return h_output;
}

void ReadMRCDims(string path, int3 &dims)
{
	char* header = (char*)malloc(1024 * sizeof(char));

	ifstream inputfile(path, ios::in|ios::binary);
	inputfile.seekg(0, ios::beg);
	inputfile.read(header, 1024 * sizeof(char));

	dims.x = ((int*)header)[0];
	dims.y = ((int*)header)[1];
	dims.z = ((int*)header)[2];

	free(header);
}

void ReadEMDims(string path, int3 &dims)
{
	char* header = (char*)malloc(1024 * sizeof(char));

	ifstream inputfile(path, ios::in|ios::binary);
	inputfile.seekg(0, ios::beg);
	inputfile.read(header, 1024 * sizeof(char));

	dims.x = ((int*)header)[1];
	dims.y = ((int*)header)[2];
	dims.z = ((int*)header)[3];

	free(header);
}