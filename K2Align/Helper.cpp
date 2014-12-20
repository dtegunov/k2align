#include "Functions.cuh"

tfloat Mean(tfloat* h_input, size_t elements)
{
	tfloat result = (tfloat)0;
	tfloat c = 0, y, t;
	for (int i = 0; i < elements; i++)
	{
		y = h_input[i] - c;
		t = result + y;
		c = (t - result) - y;
		result = t;
	}

	return result / (tfloat)elements;
}

tfloat Mean(vector<tfloat> h_input)
{
	size_t elements = h_input.size();
	tfloat result = (tfloat)0;
	tfloat c = 0, y, t;
	for (int i = 0; i < elements; i++)
	{
		y = h_input[i] - c;
		t = result + y;
		c = (t - result) - y;
		result = t;
	}

	return result / (tfloat)elements;
}

tfloat StdDev(tfloat* h_input, size_t elements)
{
	tfloat inputmean = Mean(h_input, elements);
	tfloat result = (tfloat)0;
	
	tfloat c = 0, y, t, diff;
	for (int i = 0; i < elements; i++)
	{
		diff = h_input[i] - inputmean;
		y = diff * diff - c;
		t = result + y;
		c = (t - result) - y;
		result = t;
	}

	return sqrt(result / (tfloat)(elements - 1));
}

tfloat StdDev(vector<tfloat> h_input)
{
	size_t elements = h_input.size();
	tfloat inputmean = Mean(h_input);
	tfloat result = (tfloat)0;
	
	tfloat c = 0, y, t, diff;
	for (int i = 0; i < elements; i++)
	{
		diff = h_input[i] - inputmean;
		y = diff * diff - c;
		t = result + y;
		c = (t - result) - y;
		result = t;
	}

	return sqrt(result / (tfloat)(elements - 1));
}