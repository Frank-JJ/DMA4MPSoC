#include <stdio.h>
#include <string>
#include "hls_stream.h"
#include "ap_axi_sdata.h"
#include "types.hpp"

#ifndef LAYER_INFO
#define LAYER_INFO
#include "layerInfo.hpp"
#endif

#include "../testImage.h"

#include <chrono>

void infer(long_uint_stream &infer_input, long_uint_stream &infer_output);

uint image_num = 0;

int test_using_image(int image_array[input_dim1][input_dim2][input_dim3],
		float prediction_array[output_dim1])
{
	printf("\nImage: %d\n", image_num++);


    // Get start time before stream in
    auto start_in = std::chrono::high_resolution_clock::now();

	long_uint_stream stream_in;
	long_uint_stream stream_out;

	long_uint_package input_package;
	uint8_t *input_package_pointer = (uint8_t*)&input_package.data;
	int *image_pointer = (int*)image_array;
	for (int i = 0; i < input_dim1*input_dim2*input_dim3; i += 4)
	{
		input_package_pointer[0] = (uint8_t)image_pointer[i];
		input_package_pointer[1] = (uint8_t)image_pointer[i+1];
		input_package_pointer[2] = (uint8_t)image_pointer[i+2];
		input_package_pointer[3] = (uint8_t)image_pointer[i+3];

		stream_in << input_package;
	}

	// Get start time before infer
	auto start_infer = std::chrono::high_resolution_clock::now();

	infer(stream_in, stream_out);

	// Get end time after infer
	auto end_infer = std::chrono::high_resolution_clock::now();

	float_uint results[output_dim1];
	long_uint_package output_package;

	// Check results
	bool result_same = true;
	bool type_same = true;
	int result_type = 0;
	int prediction_type = 0;
	int i = 0;
	bool last = false;
	while (last == false)
	{
		stream_out >> output_package;
		results[i].uint_val = output_package.data;
		printf("%f	| should be: %f 	| diff:	%f\n", results[i].float_val, prediction_array[i], results[i].float_val-prediction_array[i]);
		if ((int)(prediction_array[i]*10000) != (int)(results[i].float_val*10000))
		{
			result_same = false;
		}
		if (results[i].float_val > results[result_type].float_val)
		{
			result_type = i;
		}
		if (prediction_array[i] > prediction_array[prediction_type])
		{
			prediction_type = i;
		}
		last = output_package.last;
		i++;
	}

	// Get end time after stream out
	auto end_out = std::chrono::high_resolution_clock::now();

	// Get durations
	auto duration_in = std::chrono::duration_cast<std::chrono::microseconds>(start_infer - start_in);
	auto duration_infer = std::chrono::duration_cast<std::chrono::microseconds>(end_infer - start_infer);
	auto duration_out = std::chrono::duration_cast<std::chrono::microseconds>(end_out - end_infer);
	auto duration_total = std::chrono::duration_cast<std::chrono::microseconds>(end_out - start_in);

	printf("In took: %d us | ", duration_in);
	printf("Infer took: %d us | ", duration_infer);
	printf("Out took: %d us | ", duration_out);
	printf("Total took: %d us\n\r", duration_total);

	// Check if all predictions where received
	if (i != output_dim1)
	{
		printf("number of predictions wrong: %d should be: %d\n", i,output_dim1);
		return 1;
	}


	printf("type: %d	| should be: %d\n", result_type, prediction_type);
	if (result_type != prediction_type)
	{
		type_same = false;
	}



	// Show results
	if (result_same && type_same)
		printf("Results and type are correct!\n");
	else if (type_same) {
		printf("Type is correct!\n");
	}
	else
	{
		printf("######\n");
		printf("######Wrong, check layers are written correctly######\n");
		printf("######\n");
		return 1;
	}
	return 0;
}

int main()
{

	int zero_input[60][60][1] = {};
	float prediction_zero[4] = {0.015625, 0.980469, 0.000000, 0.000000};
	int ret_ = test_using_image(zero_input, prediction_zero);
	int ret0 = test_using_image(test_image0, prediction0);
	int ret1 = test_using_image(test_image1, prediction1);
	int ret2 = test_using_image(test_image2, prediction2);
	int ret3 = test_using_image(test_image3, prediction3);
	int ret4 = test_using_image(test_image4, prediction4);
	int ret5 = test_using_image(test_image5, prediction5);
	int ret6 = test_using_image(test_image6, prediction6);
	int ret7 = test_using_image(test_image7, prediction7);
	int ret8 = test_using_image(test_image8, prediction8);
	int ret9 = test_using_image(test_image9, prediction9);



    return ret_ || ret0 || ret1 || ret2 || ret3 || ret4 || ret5 || ret6 || ret7 || ret8 || ret9;
}
