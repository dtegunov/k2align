#include "Functions.cuh"
#include "tinydir.h"
#include <omp.h>

void OptimizeTranslation(tfloat3* h_translations, 
						 tfloat2* subframetranslations, 
						 int adjacentgap, 
						 int firstframe, 
						 int n_subframes, 
						 int averageextent, 
						 int iterations, 
						 tfloat maxsigma,
						 tfloat &errorstddev,
						 set<int> exclude)
{
	subframetranslations[0] = tfloat2(0, 0);
	for (int j = 1; j < n_subframes; j++)
		subframetranslations[j] = tfloat2(h_translations[((j - 1) * n_subframes + j)].x, h_translations[((j - 1) * n_subframes + j)].y);
	tfloat2* translationerrors = (tfloat2*)malloc(n_subframes * n_subframes * sizeof(tfloat2));
	for (int j = 0; j < n_subframes * n_subframes; j++)
		translationerrors[j] = tfloat2(0, 0);
	errorstddev = (tfloat)9999999;
	tfloat2* corrections = (tfloat2*)malloc(n_subframes * sizeof(tfloat2));
	int* correctionsamples = (int*)malloc(n_subframes * sizeof(int));
	bool* considertranslation = (bool*)malloc(n_subframes * n_subframes * sizeof(bool));
	for (int y = firstframe; y < n_subframes; y++)
		for (int x = firstframe; x < n_subframes; x++)
			considertranslation[y * n_subframes + x] = (x > y + adjacentgap);// && (exclude.count(x) == 0) && (exclude.count(y) == 0);

	for(int a = 0; a < iterations; a++)
	{
		tfloat correctionfactor = (tfloat)1 / (tfloat)(a + 1);
		vector<tfloat> errors;
		for (int j = 0; j < n_subframes; j++)
		{
			corrections[j] = tfloat2(0, 0);
			correctionsamples[j] = 0;
		}

		for(int s1 = max(firstframe, 0); s1 < n_subframes - 1 - (adjacentgap); s1++)
			for(int s2 = s1 + 1 + adjacentgap; s2 < n_subframes; s2++)
			{
				if(!considertranslation[s1 * n_subframes + s2])
					continue;

				tfloat2 currentsum = tfloat2(0, 0);
				for (int t = s1 + 1; t <= s2; t++)
					currentsum = tfloat2(currentsum.x + subframetranslations[t].x, currentsum.y + subframetranslations[t].y);
				if(currentsum.x == 0) currentsum.x = (tfloat)0.0001;
				if(currentsum.y == 0) currentsum.y = (tfloat)0.0001;

				tfloat correctxBy = (h_translations[s1 * n_subframes + s2].x - currentsum.x) * correctionfactor / (tfloat)(s2 - s1);
				tfloat correctyBy = (h_translations[s1 * n_subframes + s2].y - currentsum.y) * correctionfactor / (tfloat)(s2 - s1);

				tfloat2 currenterror = tfloat2((h_translations[s1 * n_subframes + s2].x - currentsum.x) / (tfloat)(s2 - s1), 
												(h_translations[s1 * n_subframes + s2].y - currentsum.y) / (tfloat)(s2 - s1));
				translationerrors[s1 * n_subframes + s2] = currenterror;
				errors.push_back(currenterror.x * currenterror.x + currenterror.y * currenterror.y);	//Take square of error

				for (int t = s1 + 1; t <= s2; t++)
				{
					corrections[t].x += correctxBy;
					corrections[t].y += correctyBy;
					correctionsamples[t]++;
				}
			}

		errorstddev = sqrt(Mean(errors));
		for (int j = 0; j < n_subframes; j++)
			if (correctionsamples[j] > 0)
			{
				subframetranslations[j].x += corrections[j].x / (tfloat)correctionsamples[j];
				subframetranslations[j].y += corrections[j].y / (tfloat)correctionsamples[j];
			}

		//printf("%d iteration: %f\n", a, errorstddev);
	}
	free(corrections);
	free(correctionsamples);
	free(considertranslation);
	free(translationerrors);
}