#include "Functions.cuh"
#include "tinydir.h"
#include <omp.h>
#include "boost/thread.hpp"
#include "boost/program_options.hpp"

namespace po = boost::program_options;

__declspec(dllexport) void __stdcall h_FrameAlign(char* c_imagepath, tfloat* h_outputwhole, tfloat* h_outputquads, 
												 bool correctgain, tfloat* h_gainfactor, int3 gainfactordims,
												 bool correctxray, 
												 float bandpasslow, float bandpasshigh,
												 char* c_subframeformat, int rawdatatype,
												 int3 rawdims, 
												 int averageextent,
												 int adjacentgap,
												 int firstframe, int lastframe,
												 int* h_outputranges, int numberoutputranges,
												 tfloat maxdrift, int minvalidframes,
												 int outputdownsamplefactor,
												 int3 quaddims,
												 int3 quadnum,
												 bool* iscorrupt,
												 char* c_logpath)
{
	int cudadevice;
	ofstream logfile;

	string imagepath(c_imagepath);
	string subframeformat(c_subframeformat);
	string logpath(c_logpath);

	tfloat downsamplefactor = (tfloat)4;

	bool doquads = true;
	bool dowholeframe = true;

	int outputfirstframe = 99999, outputlastframe = 0;
	for (int i = 0; i < numberoutputranges * 2; i++)
	{
		outputfirstframe = min(h_outputranges[i], outputfirstframe);
		outputlastframe = max(h_outputranges[i], outputlastframe);
	}

	adjacentgap = max(adjacentgap, averageextent);
	outputfirstframe = max(outputfirstframe, firstframe);
	outputlastframe = min(outputlastframe, lastframe);
	minvalidframes = min(minvalidframes, outputlastframe - outputfirstframe + 1);

	int3 outputframedims;

	tfloat* d_gainfactor;
	if(correctgain)
	{
		EM_DATATYPE gainfactor_datatype = EM_DATATYPE::EM_BYTE;
		d_gainfactor = (tfloat*)CudaMallocFromHostArray(h_gainfactor, Elements(gainfactordims) * sizeof(tfloat));
	}

	boost::thread_group writethreads;
	boost::mutex writethreadsadd;

	std::vector<string> subframepaths;
	subframepaths.push_back(imagepath);
	int n_subframes = 0;
	
	int3 framedims;

	//Read dimensions of the current stack, fit the gain mask if needed
	if(subframeformat == "mrc")
		ReadMRCDims(subframepaths[0], framedims);
	else if(subframeformat == "em")
		ReadEMDims(subframepaths[0], framedims);
	else if(subframeformat == "dat")
		framedims = toInt3(rawdims.x, rawdims.y, rawdims.z);
	else
		throw;

	lastframe = min(lastframe, framedims.z - 1);
	outputlastframe = min(framedims.z - 1, outputlastframe);
	n_subframes = min(framedims.z, lastframe + 1);
	framedims.z = 1;

	adjacentgap = min(7, max(n_subframes - 5, 1));

	outputframedims = toInt3(framedims.x / outputdownsamplefactor, framedims.y / outputdownsamplefactor, 1);

	quaddims.x = min(quaddims.x, framedims.x);
	quaddims.y = min(quaddims.y, framedims.y);
	int3 outputquaddims = toInt3(quaddims.x / outputdownsamplefactor, quaddims.y / outputdownsamplefactor, 1);

	//Allocate array of pointers to pre-processed data used for cross-correlation
	void** h_subframes = (void**)malloc(n_subframes * sizeof(void*));
	EM_DATATYPE subframetype;
	if(subframeformat == "dat")
		subframetype = (EM_DATATYPE)rawdatatype;
	tcomplex** h_subframeFFTs = (tcomplex**)malloc(n_subframes * (1 + Elements(quadnum)) * sizeof(tcomplex*));

	int3 downsampleddims;
	int3 downsampledquaddims;
	tfloat* d_gainfactorcut;
	bool useoriginalgain = true;

	//Read data, downsample and downfilter it, store FFT of each frame in host memory
	std::cout << "\tProcessing subframes\n";

	downsampleddims = toInt3(framedims.x / downsamplefactor, framedims.y / downsamplefactor, 1);
	downsampledquaddims = toInt3(quaddims.x / downsamplefactor, quaddims.y / downsamplefactor, 1);

	//In case frame dimensions are smaller than gain mask, extract central portion
	if(correctgain)
	{
		if(framedims.x > gainfactordims.x || framedims.y > gainfactordims.y)
			throw;

		if(framedims.x != gainfactordims.x || framedims.y != gainfactordims.y)
		{
			cudaMalloc((void**)&d_gainfactorcut, Elements(framedims) * sizeof(tfloat));
			d_Extract(d_gainfactor, d_gainfactorcut, gainfactordims, framedims, toInt3(gainfactordims.x / 2, gainfactordims.y / 2, 0));
			useoriginalgain = false;
		}
	}

	tfloat3* h_translations = (tfloat3*)malloc(n_subframes * n_subframes * (1 + Elements(quadnum)) * sizeof(tfloat3));
	for (int j = 0; j < n_subframes * n_subframes * (1 + Elements(quadnum)); j++)
		h_translations[j] = tfloat3(0, 0, 0);

	set<int> exclude;

	boost::mutex preloadmutex;
	int itemspreloaded = 0;

	//Asynchronous image preloading loop
	boost::thread asyncpreload([&]
	{
		for(int s = firstframe; s < n_subframes; s++)
		{
			char* h_subframe;

			if(subframeformat == "mrc")
				ReadMRC(subframepaths[0], (void**)&h_subframe, subframetype, s);			
			else if(subframeformat == "em")
				ReadEM(subframepaths[0], (void**)&h_subframe, subframetype, s);
			else if(subframeformat == "dat")
				ReadRAW(subframepaths[0], (void**)&h_subframe, (EM_DATATYPE)rawdatatype, framedims, s);

			h_subframes[s] = h_subframe;

			preloadmutex.lock();
			itemspreloaded++;
			preloadmutex.unlock();
		}
	});
			
	//Preprocess new image and cross-correlate it with already loaded data
	{
		//Allocate resources for initial frame FFT and downsampling
		tfloat* d_subframe;
		cudaMalloc((void**)&d_subframe, Elements(framedims) * sizeof(tfloat));
		tfloat* d_subframetemp;
		cudaMalloc((void**)&d_subframetemp, Elements(framedims) * sizeof(tfloat));
		tfloat* d_subframedownsampled;
		cudaMalloc((void**)&d_subframedownsampled, Elements(downsampleddims) * sizeof(tfloat));
		tcomplex* d_subframeFFT;
		cudaMalloc((void**)&d_subframeFFT, ElementsFFT(downsampleddims) * sizeof(tcomplex));
		tcomplex** d_quadFFTs = (tcomplex**)malloc(Elements(quadnum) * sizeof(tcomplex*));
		for (int q = 0; q < Elements(quadnum); q++)
			cudaMalloc((void**)(d_quadFFTs + q), ElementsFFT(downsampledquaddims) * sizeof(tcomplex));

		cufftHandle scaleplanforw = d_FFTR2CGetPlan(2, framedims);
		cufftHandle scaleplanback = d_IFFTC2CGetPlan(2, downsampleddims);
		cufftHandle downsampledforw = d_FFTR2CGetPlan(2, downsampleddims);
		cufftHandle quadforw = d_FFTR2CGetPlan(2, downsampledquaddims);
							
		//Allocate resources for frame cross-correlation
		tfloat maskradius = (tfloat)10;
		tcomplex* d_inputFFT1;
		cudaMalloc((void**)&d_inputFFT1, ElementsFFT(downsampleddims) * sizeof(tcomplex));
		tfloat* d_correlation;
		cudaMalloc((void**)&d_correlation, Elements(downsampleddims) * sizeof(tfloat));
		tfloat* d_correlationmask = CudaMallocValueFilled(Elements(downsampleddims), (tfloat)1);
		d_SphereMask(d_correlationmask, d_correlationmask, downsampleddims, &maskradius, (tfloat)0, (tfloat3*)NULL);
		tfloat* d_correlationquadmask = CudaMallocValueFilled(Elements(downsampleddims), (tfloat)1);
		d_SphereMask(d_correlationquadmask, d_correlationquadmask, downsampledquaddims, &maskradius, (tfloat)0, (tfloat3*)NULL);
		tfloat* d_quadcorrelationmask = CudaMallocValueFilled(Elements(downsampledquaddims), (tfloat)1);
		d_SphereMask(d_quadcorrelationmask, d_quadcorrelationmask, downsampledquaddims, &maskradius, (tfloat)0, (tfloat3*)NULL);
		tfloat* d_correlationshifted;
		cudaMalloc((void**)&d_correlationshifted, Elements(downsampleddims) * sizeof(tfloat));
		tfloat3* d_translation;
		cudaMalloc((void**)&d_translation, sizeof(tfloat3));
		tfloat3* h_translation = (tfloat3*)malloc(sizeof(tfloat3));
		tfloat* d_peakvalue;
		cudaMalloc((void**)&d_peakvalue, sizeof(tfloat));
				
		cufftHandle corrplanback = d_IFFTC2RGetPlan(2, downsampleddims);
		cufftHandle corrquadplanback = d_IFFTC2RGetPlan(2, downsampledquaddims);

		int3 perquadshift = toInt3(quadnum.x > 0 ? downsampledquaddims.x - (downsampledquaddims.x * quadnum.x - downsampleddims.x) / (quadnum.x - 1) : 0,
									quadnum.y > 0 ? downsampledquaddims.y - (downsampledquaddims.y * quadnum.y - downsampleddims.y) / (quadnum.y - 1) : 0,
									0);

		for(int s = firstframe; s < n_subframes; s++)
		{
			//Wait for next frame to be read from disk
			while(itemspreloaded <= min(s + averageextent - firstframe, n_subframes - 1))
				boost::this_thread::sleep(boost::posix_time::milliseconds(1));
					
			//Moving window average
			d_ValueFill(d_subframe, Elements(framedims), (tfloat)0);
			for(int ss = max(s - averageextent, 0); ss <= min(s + averageextent, n_subframes - 1); ss++)
			{
				MixedToDeviceTfloat(h_subframes[ss], d_subframetemp, subframetype, Elements(framedims));
				if(d_HasZeroRects(d_subframetemp, framedims, toInt3(100, 20, 1)) || d_HasZeroRects(d_subframetemp, framedims, toInt3(20, 100, 1)))
				{
					*iscorrupt = true;
					if(exclude.count(ss) == 0)
						exclude.insert(ss);
				}
				d_AddVector(d_subframe, d_subframetemp, d_subframe, Elements(framedims));
				cudaStreamQuery(0);
			}

			//Multiply by gain mask
			if(correctgain)
				d_MultiplyByVector(d_subframe, d_gainfactor, d_subframe, Elements(framedims));

			//Smooth out abnormal pixel values
			if(correctxray)
				d_Xray(d_subframe, d_subframe, framedims);

			//Scale, filter and save frame's FFT to host memory
			d_Scale(d_subframe, d_subframedownsampled, framedims, downsampleddims, T_INTERP_MODE::T_INTERP_FOURIER, NULL, &scaleplanback);
			cudaStreamQuery(0);

			//Neat version to avoid artifacts at the edges, since this involves highpass as well
			d_BandpassNeat(d_subframedownsampled, 
							d_subframedownsampled, 
							downsampleddims, 
							(tfloat)min(framedims.x, framedims.y) * bandpasslow, 
							(tfloat)min(framedims.x, framedims.y) * bandpasshigh,
							0);
			d_HammingMask(d_subframedownsampled, (tfloat*)d_subframeFFT, downsampleddims, NULL, NULL);

			d_FFTR2C((tfloat*)d_subframeFFT, d_subframeFFT, &downsampledforw);
			cudaStreamQuery(0);
			
			tcomplex* h_subframeFFT = (tcomplex*)MallocPinnedFromDeviceArray(d_subframeFFT, ElementsFFT(downsampleddims) * sizeof(tcomplex));
			cudaStreamQuery(0);

			//Store transformed, downsampled and filtered subframe for later use
			h_subframeFFTs[s] = h_subframeFFT;

			//Extract and save quad FFTs
			if(doquads)
				for (int y = 0; y < quadnum.y; y++)
					for (int x = 0; x < quadnum.x; x++)
					{
						tfloat* d_quaddownsampled = d_subframe;
						tcomplex* d_quadFFT = d_quadFFTs[y * quadnum.x + x];

						d_Extract(d_subframedownsampled, 
									d_quaddownsampled, 
									downsampleddims, 
									downsampledquaddims,
									toInt3(x * perquadshift.x + downsampleddims.x / 2,
											y * perquadshift.y + downsampleddims.y / 2,
											0));
						d_HammingMask(d_quaddownsampled, d_quaddownsampled, downsampledquaddims, NULL, NULL);
						d_FFTR2C(d_quaddownsampled, d_quadFFT, &quadforw);

						tcomplex* h_quadFFT = (tcomplex*)MallocPinnedFromDeviceArray(d_quadFFT, ElementsFFT(downsampledquaddims) * sizeof(tcomplex));
						cudaStreamQuery(0);

						h_subframeFFTs[n_subframes + (y * quadnum.x + x) * n_subframes + s] = h_quadFFT;
					}

			//Build new column for full frame and quad correlation matrices
			int s2 = s;
			for(int s1 = firstframe; s1 < s2 - adjacentgap; s1++)
			{
				tfloat3 relativetranslation((tfloat)0);

				if(dowholeframe)
				{
					cudaMemcpy(d_inputFFT1, h_subframeFFTs[s1], ElementsFFT(downsampleddims) * sizeof(tcomplex), cudaMemcpyHostToDevice);

					d_ComplexMultiplyByConjVector(d_subframeFFT, d_inputFFT1, d_inputFFT1, ElementsFFT(downsampleddims));
					d_IFFTC2R(d_inputFFT1, d_correlation, &corrplanback, downsampleddims);
					d_RemapFullFFT2Full(d_correlation, d_correlationshifted, downsampleddims);
					d_MultiplyByVector(d_correlationshifted, d_correlationmask, d_correlationshifted, Elements(downsampleddims));
					d_Peak(d_correlationshifted, d_translation, d_peakvalue, downsampleddims, T_PEAK_MODE::T_PEAK_SUBFINE, NULL, NULL);

					cudaMemcpy(h_translation, d_translation, sizeof(tfloat3), cudaMemcpyDeviceToHost);

					//d_Peak return a result where center equals zero shift, thus subtract center coords
					relativetranslation = tfloat3(h_translation[0].x - (tfloat)(downsampleddims.x / 2),
															h_translation[0].y - (tfloat)(downsampleddims.y / 2),
															(tfloat)0);
					h_translations[s1 * n_subframes + s2] = relativetranslation;
					h_translations[s2 * n_subframes + s1] = tfloat3(-relativetranslation.x, -relativetranslation.y, (tfloat)0);
				}
				//printf("%d -> %d: %f, %f\n", s1, s2, relativetranslation.x * downsamplefactor, relativetranslation.y * downsamplefactor);

				if(doquads)
					for (int y = 0; y < quadnum.y; y++)
						for (int x = 0; x < quadnum.x; x++)
						{
							int globalsubframeoffset = (1 + y * quadnum.x + x) * n_subframes;

							cudaMemcpy(d_inputFFT1, h_subframeFFTs[globalsubframeoffset + s1], ElementsFFT(downsampledquaddims) * sizeof(tcomplex), cudaMemcpyHostToDevice);

							d_ComplexMultiplyByConjVector(d_quadFFTs[y * quadnum.x + x], d_inputFFT1, d_inputFFT1, ElementsFFT(downsampledquaddims));
							d_IFFTC2R(d_inputFFT1, d_correlation, &corrquadplanback, downsampledquaddims);
							d_RemapFullFFT2Full(d_correlation, d_correlationshifted, downsampledquaddims);
							d_MultiplyByVector(d_correlationshifted, d_correlationquadmask, d_correlationshifted, Elements(downsampledquaddims));
							d_Peak(d_correlationshifted, d_translation, d_peakvalue, downsampledquaddims, T_PEAK_MODE::T_PEAK_SUBFINE, NULL, NULL);

							cudaMemcpy(h_translation, d_translation, sizeof(tfloat3), cudaMemcpyDeviceToHost);

							relativetranslation = tfloat3(h_translation[0].x - (tfloat)(downsampledquaddims.x / 2),
															h_translation[0].y - (tfloat)(downsampledquaddims.y / 2),
															(tfloat)0);
							h_translations[(globalsubframeoffset + s1) * n_subframes + s2] = relativetranslation;
							h_translations[(globalsubframeoffset + s2) * n_subframes + s1] = tfloat3(-relativetranslation.x, -relativetranslation.y, (tfloat)0);
						}
			}
		}

		//Clean up cross-correlation resources
		cufftDestroy(corrquadplanback);
		cufftDestroy(corrplanback);
		cudaFree(d_inputFFT1);
		cudaFree(d_quadcorrelationmask);
		cudaFree(d_correlationquadmask);
		cudaFree(d_correlationmask);
		cudaFree(d_correlation);
		cudaFree(d_correlationshifted);
		cudaFree(d_translation);
		cudaFree(d_peakvalue);
		free(h_translation);

		cudaFree(d_subframetemp);
		cudaFree(d_subframe);
		cudaFree(d_subframedownsampled);
		cudaFree(d_subframeFFT);
		for (int q = 0; q < Elements(quadnum); q++)
			cudaFree(d_quadFFTs[q]);
		free(d_quadFFTs);

		cufftDestroy(scaleplanforw);
		cufftDestroy(scaleplanback);
		cufftDestroy(downsampledforw);
		cufftDestroy(quadforw);

		for(int s = (firstframe); s < n_subframes; s++)
			for (int q = 0; q < 1 + Elements(quadnum); q++)
				cudaFreeHost(h_subframeFFTs[q * n_subframes + s]);
		free(h_subframeFFTs);
	}

	cudaDeviceSynchronize();

	//Find optimal translation values performing a least-squares fit
	printf("\tOptimizing translation vector\n");

	tfloat2* subframetranslations = (tfloat2*)MallocValueFilled(n_subframes * (1 + Elements(quadnum)) * 2, (tfloat)0);
	tfloat* errorstddev = (tfloat*)MallocValueFilled((1 + Elements(quadnum)), (tfloat)0);
	if(dowholeframe)
		OptimizeTranslation(h_translations, subframetranslations, adjacentgap, firstframe, n_subframes, averageextent, 60, (tfloat)2, errorstddev[0], exclude);

	if(doquads)
		for (int q = 0; q < Elements(quadnum); q++)
			OptimizeTranslation(h_translations + (q + 1) * (n_subframes * n_subframes),
								subframetranslations + (q + 1) * n_subframes,
								adjacentgap,
								firstframe,
								n_subframes,
								averageextent,
								60,
								(tfloat)2,
								errorstddev[1 + q],
								exclude);

	free(h_translations);

	//In console version, print output to screen
	if(dowholeframe)
	{
		tfloat2* motionvector = (tfloat2*)malloc(n_subframes * sizeof(tfloat2));
		tfloat2 translationsum = tfloat2(0, 0);
		#pragma omp critical
		{
			printf("\tMotion vector\n");
			for(int n = firstframe; n < n_subframes; n++)
			{
				translationsum.x += subframetranslations[n].x;
				translationsum.y += subframetranslations[n].y;
				motionvector[n] = translationsum;
				printf("\t%d -> %d: %f, %f\n", n - 1, n, translationsum.x * (tfloat)downsamplefactor, translationsum.y * (tfloat)downsamplefactor);
			}
			printf("\tError: %f\n", errorstddev[0] * downsamplefactor);
		}
		free(motionvector);
	}

	//Shift subframes, compute average
	#pragma omp critical
	printf("\tShifting subframes\n");

	int3 perquadshift = toInt3(quadnum.x > 0 ? quaddims.x - (quaddims.x * quadnum.x - framedims.x) / (quadnum.x - 1) : 0,
								quadnum.y > 0 ? quaddims.y - (quaddims.y * quadnum.y - framedims.y) / (quadnum.y - 1) : 0,
								0);

	//Allocate resources for shifting frames and summing them up
	tfloat* d_average = (tfloat*)CudaMallocValueFilled(Elements(framedims) * numberoutputranges, (tfloat)0);
	tfloat* d_quadaverages = (tfloat*)CudaMallocValueFilled(max(Elements(quaddims) * Elements(quadnum), 1) * numberoutputranges, (tfloat)0);
	tcomplex* d_shiftintermediate;
	cudaMalloc((void**)&d_shiftintermediate, ElementsFFT(framedims) * sizeof(tcomplex));
	cufftHandle shiftplanforw = d_FFTR2CGetPlan(2, framedims);
	cufftHandle shiftplanback = d_IFFTC2RGetPlan(2, framedims);
	cufftHandle shiftquadplanforw = d_FFTR2CGetPlan(2, quaddims);
	cufftHandle shiftquadplanback = d_IFFTC2RGetPlan(2, quaddims);
	tfloat* d_subframe;
	cudaMalloc((void**)&d_subframe, Elements(framedims) * sizeof(tfloat));
	tfloat* d_quad;
	cudaMalloc((void**)&d_quad, Elements(quaddims) * sizeof(tfloat));
			
	int framesvalid = 0;
	int alignto = outputfirstframe + (min(outputlastframe, n_subframes - 1) - outputfirstframe) / 2;

	//Shift frames by previously computed values and add them to the final output
	for(int n = max(firstframe, outputfirstframe); n <= min(n_subframes - 1, min(outputlastframe, n_subframes - 1)); n++)
	{
		if(exclude.count(n) > 0)
			continue;

		tfloat drift = max(abs(subframetranslations[n].x), abs(subframetranslations[n].y));
		if(n < n_subframes - 1)
			drift = max(drift, max(abs(subframetranslations[n + 1].x), abs(subframetranslations[n + 1].y)));
		drift *= (tfloat)downsamplefactor;

		if(drift > maxdrift)
			continue;

		framesvalid++;

		tfloat3 translation = tfloat3(0, 0, 0);
		if(n < alignto)
			for(int s = n + 1; s <= alignto; s++)
			{
				translation.x += subframetranslations[s].x;
				translation.y += subframetranslations[s].y;
			}
		else if(n > alignto)
			for(int s = alignto + 1; s <= n; s++)					
			{
				translation.x -= subframetranslations[s].x;
				translation.y -= subframetranslations[s].y;
			}
			
		//Remember that computations were performed on downsampled data
		translation.x *= (tfloat)downsamplefactor;
		translation.y *= (tfloat)downsamplefactor;

		tfloat3* translationquads = (tfloat3*)MallocValueFilled<tfloat>(Elements(quadnum) * 3, (tfloat)0);
		for (int q = 0; q < Elements(quadnum); q++)
		{
			if(n < alignto)
				for(int s = n + 1; s <= alignto; s++)
				{
					translationquads[q].x += subframetranslations[(1 + q) * n_subframes + s].x;
					translationquads[q].y += subframetranslations[(1 + q) * n_subframes + s].y;
				}
			else if(n > alignto)
				for(int s = alignto + 1; s <= n; s++)					
				{
					translationquads[q].x -= subframetranslations[(1 + q) * n_subframes + s].x;
					translationquads[q].y -= subframetranslations[(1 + q) * n_subframes + s].y;
				}
			
			translationquads[q].x *= (tfloat)downsamplefactor;
			translationquads[q].y *= (tfloat)downsamplefactor;
		}

		MixedToDeviceTfloat(h_subframes[n], d_subframe, subframetype, Elements(framedims));
		if(correctgain)
			d_MultiplyByVector(d_subframe, d_gainfactor, d_subframe, Elements(framedims));
		if(correctxray)
			d_Xray(d_subframe, d_subframe, framedims);

		//Extract and add quads before whole subframe is shifted
		if(doquads)
			for (int y = 0; y < quadnum.y; y++)
				for (int x = 0; x < quadnum.x; x++)
				{
					tfloat3 globalshift = tfloat3((tfloat)(x * perquadshift.x + quaddims.x / 2),// + translationquads[y * quadnum.x + x].x,
													(tfloat)(y * perquadshift.y + quaddims.y / 2),// + translationquads[y * quadnum.x + x].y,
													(tfloat)0);
					tfloat3 localshift = tfloat3(translationquads[y * quadnum.x + x].x,
													translationquads[y * quadnum.x + x].y,
													(tfloat)0);

					d_Extract(d_subframe, 
								d_quad, 
								framedims, 
								quaddims, 
								toInt3(x * perquadshift.x + quaddims.x / 2,
										y * perquadshift.y + quaddims.y / 2,
										0));
					d_Shift(d_quad, 
							d_quad, 
							quaddims, 
							&localshift,
							&shiftquadplanforw,
							&shiftquadplanback,
							d_shiftintermediate);
					
					for (int r = 0; r < numberoutputranges; r++)
					{
						if(n >= h_outputranges[r * 2] && n <= h_outputranges[r * 2 + 1])
							d_AddVector(d_quadaverages + Elements(quaddims) * Elements(quadnum) * r + (y * quadnum.x + x) * Elements(quaddims),
										d_quad,
										d_quadaverages + Elements(quaddims) * Elements(quadnum) * r + (y * quadnum.x + x) * Elements(quaddims),
										Elements(quaddims));
					}
				}

		//Shift whole subframe and add to average
		if(dowholeframe)
		{
			if(n != alignto)
				d_Shift(d_subframe, d_subframe, framedims, &translation, &shiftplanforw, &shiftplanback, d_shiftintermediate);

			for (int r = 0; r < numberoutputranges; r++)
			{
				if(n >= h_outputranges[r * 2] && n <= h_outputranges[r * 2 + 1])
					d_AddVector(d_average + Elements(framedims) * r, d_subframe, d_average + Elements(framedims) * r, Elements(framedims));
			}
		}

		free(translationquads);
	}

	//Clean up resources for shift & sum
	cudaFree(d_quad);
	cudaFree(d_subframe);
	cufftDestroy(shiftplanforw);
	cufftDestroy(shiftplanback);
	cufftDestroy(shiftquadplanforw);
	cufftDestroy(shiftquadplanback);
	cudaFree(d_shiftintermediate);

	//Make final corrections and write log file
	{
		imgstats5* d_stats;
		cudaMalloc((void**)&d_stats, sizeof(imgstats5));
		imgstats5 h_stats;
				
		//In case outliers have become more prominent after summation, get rid of them
		for (int r = 0; r < numberoutputranges; r++)
			if(correctxray)
			{
				d_Xray(d_average + Elements(framedims) * r, d_average + Elements(framedims) * r, framedims);
			}

		//Get image statistics
		d_Dev(d_average, d_stats, Elements(framedims), (char*)NULL);
		cudaMemcpy(&h_stats, d_stats, sizeof(imgstats5), cudaMemcpyDeviceToHost);

		logfile.open(logpath.c_str(), ios::out);

		logfile << "Aligned image stats:\n";
		logfile << "Subframes:\t" << framesvalid << "\n";
		logfile << "Average:\t" << h_stats.mean << "\n";
		logfile << "StdDev:\t" << h_stats.stddev << "\n";
		logfile << "Mean squared alignment error:\t" << errorstddev[0] * downsamplefactor << "\n";

		if(dowholeframe)
		{
			logfile << "Relative translation:\n";
			for(int n = (outputfirstframe); n <= min(n_subframes - 1, min(outputlastframe, n_subframes - 1)); n++)
				logfile << subframetranslations[n].x * (tfloat)downsamplefactor << "\t" << subframetranslations[n].y * (tfloat)downsamplefactor << "\n";
		}

		if(doquads)
		{
			tfloat* h_quadshifts = (tfloat*)malloc(Elements(quadnum) * sizeof(tfloat2));

			logfile << "\n" << Elements(quadnum) << " segments:\n";

			for (int y = 0; y < quadnum.y; y++)
				for (int x = 0; x < quadnum.x; x++)
				{
					logfile << "\n";

					int nquad = y * quadnum.x + x;
					logfile << (x + 1)  << "-" << (y + 1) << ":\n";
					for(int n = (outputfirstframe); n <= min(n_subframes - 1, min(outputlastframe, n_subframes - 1)); n++)
						logfile << subframetranslations[(1 + nquad) * n_subframes + n].x * (tfloat)downsamplefactor << "\t" << subframetranslations[(1 + nquad) * n_subframes + n].y * (tfloat)downsamplefactor << "\n";
				}

			logfile << "\nInter-segment error:\n";
			for(int n = max(outputfirstframe, firstframe); n <= min(n_subframes - 1, min(outputlastframe, n_subframes - 1)); n++)
			{
				for (int q = 0; q < Elements(quadnum); q++)
				{
					h_quadshifts[q] = subframetranslations[(1 + q) * n_subframes + n].x;
					h_quadshifts[Elements(quadnum) + q] = subframetranslations[(1 + q) * n_subframes + n].y;
				}

				tfloat2 quadstddev = tfloat2(StdDev(h_quadshifts, Elements(quadnum)), StdDev(h_quadshifts + Elements(quadnum), Elements(quadnum)));
				logfile << quadstddev.x * downsamplefactor << "\t" << quadstddev.y * downsamplefactor << "\n";
			}

			free(h_quadshifts);
		}

		logfile.close();
	}

	free(subframetranslations);
	free(errorstddev);

	//Downsample output
	if(outputdownsamplefactor > (tfloat)1)
	{
		for (int r = 0; r < numberoutputranges; r++)
		{
			if(dowholeframe)
				d_Scale(d_average + Elements(framedims) * r, d_average + Elements(outputframedims) * r, framedims, outputframedims, T_INTERP_MODE::T_INTERP_FOURIER);

			if(doquads)
			{
				for (int q = 0; q < Elements(quadnum); q++)
					d_Scale(d_quadaverages + Elements(quaddims) * Elements(quadnum) * r + q * Elements(quaddims), d_quadaverages + Elements(outputquaddims) * Elements(quadnum) * r + q * Elements(outputquaddims), quaddims, outputquaddims, T_INTERP_MODE::T_INTERP_FOURIER);
			}
		}
	}
			
	cudaMemcpy(h_outputwhole, d_average, Elements(outputframedims) * numberoutputranges * sizeof(tfloat), cudaMemcpyDeviceToHost);
	cudaFree(d_average);
	if(Elements(quadnum) > 0)
		cudaMemcpy(h_outputquads, d_quadaverages, Elements(outputquaddims) * Elements(quadnum) * numberoutputranges * sizeof(tfloat), cudaMemcpyDeviceToHost);
	cudaFree(d_quadaverages);			

	if(correctgain && !useoriginalgain)
		cudaFree(d_gainfactorcut);
		
	for(int s = firstframe; s < n_subframes; s++)
		cudaFreeHost(h_subframes[s]);
	free(h_subframes);

	if(correctgain)
		cudaFree(d_gainfactor);
}