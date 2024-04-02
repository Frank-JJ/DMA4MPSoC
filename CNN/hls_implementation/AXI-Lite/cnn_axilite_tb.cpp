#include <stdio.h>
#include <string>
#include "types.hpp"

#ifndef LAYER_INFO
#define LAYER_INFO
#include "layerInfo.hpp"
#endif

#include "testImage.h"

#include <chrono>

int infer(int in[3600]);

uint image_num = 0;

int test_using_image(int image_array[input_dim1][input_dim2][input_dim3],
		float prediction_array[output_dim1])
{
	printf("\nImage: %d\n", image_num++);


    // Get start time before stream in
    auto start_in = std::chrono::high_resolution_clock::now();


	int* image_pointer = (int*)image_array;
	int image_array_flat[input_dim1*input_dim2*input_dim3];
	for (int i = 0; i < input_dim1*input_dim2*input_dim3; i ++)
	{
		image_array_flat[i] = image_pointer[i];
	}

	// Get start time before infer
	auto start_infer = std::chrono::high_resolution_clock::now();

	int result = infer(image_array_flat);

	// Get end time after infer
	auto end_infer = std::chrono::high_resolution_clock::now();


	// Check results
	float last = 0;
	int prediction_type = 0;
	for(int i = 0; i < 4; i++){
    	if (prediction_array[i] > last){
			last = prediction_array[i];
			prediction_type = i;
    	}
	}
	bool resultbool = result == prediction_type ? 0 : 1;

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

	return resultbool;
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
