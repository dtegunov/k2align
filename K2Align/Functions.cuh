#include "../../gtom/include/GTOM.cuh"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <set>

bool d_HasZeroRects(tfloat* d_input, int3 dims, int3 rectdims);

void OptimizeTranslation(tfloat3* h_translations, 
						 tfloat2* subframetranslations, 
						 int adjacentgap, 
						 int firstframe, 
						 int n_subframes, 
						 int averageextent, 
						 int iterations, 
						 tfloat maxsigma,
						 tfloat &errorstddev,
						 set<int> exclude);

tfloat Mean(tfloat* h_input, size_t elements);
tfloat Mean(vector<tfloat> h_input);
tfloat StdDev(tfloat* h_input, size_t elements);
tfloat StdDev(vector<tfloat> h_input);

extern "C" __declspec(dllexport) void __stdcall h_FrameAlign(char* c_imagepath, tfloat* h_outputwhole, tfloat* h_outputquads, 
															 bool correctgain, tfloat* h_gainfactor, int3 gainfactordims,
															 bool correctxray, bool lookforblacksquares, 
															 float bandpasslow,
															 float bandpasshigh,
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
															 bool outputstack,
															 char* c_outputstackname,
															 bool* iscorrupt,
															 char* c_logpath);