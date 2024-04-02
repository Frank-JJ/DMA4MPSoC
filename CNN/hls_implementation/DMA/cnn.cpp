#include "cnn.hpp"

/*
*   Set value to 0 if negative
*/
void relu(fixed &input)
{
	if (input < 0)
		input = 0;
}


/*
*   This assumes kernel is 3x3, stride is (1,1), and padding is valid
*/
template <const sizetype input_size_1, const sizetype input_size_2, const sizetype input_size_3,
			const sizetype weights_size_1, const sizetype weights_size_2, const sizetype weights_size_3, const sizetype weights_size_4,
			const sizetype bias_size,
			const sizetype output_size_1, const sizetype output_size_2, const sizetype output_size_3>
void conv2d(fixed (&input)[input_size_1][input_size_2][input_size_3],
            const fixed (&weights)[weights_size_1][weights_size_2][weights_size_3][weights_size_4],
			const fixed (&bias)[bias_size],
            fixed (&output)[output_size_1][output_size_2][output_size_3])
{
	fixed output_sum[weights_size_4] = { };
#pragma HLS array_partition variable=output_sum complete
    // Convolution
    // Input rows
	conv2d1: for (sizetype i = 1; i < input_size_1 - 1; i++)
    {
        // Input columns
		conv2d2: for (sizetype ii = 1; ii < input_size_2 - 1; ii++)
        {
			// Kernel number
			conv2d3_1: for (sizetype iii = 0; iii < weights_size_4; iii++)
			{
				// Add bias
				output_sum[iii] = bias[iii];
			}
			// Input and kernel channels
			conv2d3_2: for (sizetype iv = 0; iv < input_size_3; iv++)
			{
				// Kernel rows
				conv2d4: for (int v = -1; v < 2; v++)
				{
					// Kernel cols
					conv2d5: for (int vi = -1; vi < 2; vi++)
					{
						fixed input_val = input[(i + v)][(ii + vi)][iv];
						// Kernel number
						conv2d6: for (sizetype iii = 0; iii < weights_size_4; iii++)
						{
							output_sum[iii] += input_val * weights[(v + 1)][(vi + 1)][iv][iii];
						}
					}
				}
			}
			// Kernel number
			conv2d3_3: for (sizetype iii = 0; iii < weights_size_4; iii++)
			{
				// Apply relu activiation function
				relu(output_sum[iii]);
				output[i - 1][ii - 1][iii] = output_sum[iii];
            }
        }
    }
}

/*
*   This assumes filter is 2x2, stride is (2,2), and padding is valid
*/
template <const sizetype input_size_1, const sizetype input_size_2, const sizetype input_size_3,
            const sizetype output_size_1, const sizetype output_size_2, const sizetype output_size_3>
void max_pooling2d(fixed (&input)[input_size_1][input_size_2][input_size_3],
                fixed (&output)[output_size_1][output_size_2][output_size_3])
{
    // Input rows
	max_pooling2d1: for (sizetype i = 0; i < input_size_1 - 1; i += 2)
    {
        // Input cols
		max_pooling2d2: for (sizetype ii = 0; ii < input_size_2 - 1; ii += 2)
        {
            // Input channels
			max_pooling2d3: for (sizetype iii = 0; iii < input_size_3; iii++)
            {
                fixed maxVal = 0;
                // Filter rows
                max_pooling2d4: for (uint iv = 0; iv < 2; iv++)
                {
                    // Filter cols
                	max_pooling2d5: for (uint v = 0; v < 2; v++)
                    {
                    	fixed input_val = input[(i + iv)][(ii + v)][iii];
                        if (input_val > maxVal)
                        {
                            maxVal = input_val;
                        }
                    }
                }
                output[(i>>1)][(ii>>1)][iii] = maxVal;
            }
        }
    }
}


/*
* 	Convert 3d array to 1d array
*/
template <const sizetype size_1, const sizetype size_2, const sizetype size_3>
void array_3d_to_1d(fixed (&array)[size_1][size_2][size_3], fixed (&output)[size_1*size_2*size_3])
{
	for (sizetype i = 0; i < size_1; i++)
		for (sizetype ii = 0; ii < size_2; ii++)
			for (sizetype iii = 0; iii < size_3; iii++)
				output[i*size_2*size_3 + ii*size_3 + iii] = array[i][ii][iii];
}

/*
* 	Convert 1d array to 3d array
*/
template <const sizetype size_1, const sizetype size_2, const sizetype size_3>
void array_1d_to_3d(fixed (&array)[size_1*size_2*size_3], fixed (&output)[size_1][size_2][size_3])
{
	for (sizetype i = 0; i < size_1; i++)
		for (sizetype ii = 0; ii < size_2; ii++)
			for (sizetype iii = 0; iii < size_3; iii++)
				output[i][ii][iii] = array[i*size_2*size_3 + ii*size_3 + iii];
}


/*
*   Dense with relu
*/
template <const sizetype input_size,
            const sizetype weights_size_1, const sizetype weights_size_2,
            const sizetype bias_size,
            const sizetype output_size>
void dense_relu(fixed (&input)[input_size],
            const fixed (&weights)[weights_size_1][weights_size_2],
            const fixed (&bias)[bias_size],
            fixed (&output)[output_size])
{
	dense_relu1: for (sizetype i = 0; i < weights_size_2; i++)
    {
		// Add bias
		fixed output_sum = bias[i];
		dense_relu2: for (sizetype ii = 0; ii < weights_size_1; ii++)
        {
			output_sum += input[ii] * weights[ii][i];
        }

        output[i] = output_sum;
        // Apply relu activation function
        relu(output[i]);
    }
}

/*
*   Dense
*/
template <const sizetype input_size,
            const sizetype weights_size_1, const sizetype weights_size_2,
            const sizetype bias_size,
            const sizetype output_size>
void dense(fixed (&input)[input_size],
            const fixed (&weights)[weights_size_1][weights_size_2],
            const fixed (&bias)[bias_size],
            fixed (&output)[output_size])
{
	dense1: for (sizetype i = 0; i < weights_size_2; i++)
    {
		// Add bias
		fixed output_sum = bias[i];
		dense2: for (sizetype ii = 0; ii < weights_size_1; ii++)
        {
			output_sum += input[ii] * weights[ii][i];
        }

        output[i] = output_sum;
    }
}

/*
* 	Converts array values to probability distribution
*/
template <const sizetype size>
void softmax(fixed (&array)[size])
{
	softmax_fixed temp_array[size];
	softmax_fixed sum = 0;
	softmax1_1: for (sizetype i = 0; i < size; i++)
	{
		temp_array[i] = exp_reduce::exp((softmax_fixed)array[i]);
		sum += temp_array[i];
	}
	softmax1_2: for (sizetype i = 0; i < size; i++)
	{
		array[i] = temp_array[i] / sum;
	}
}

void infer(long_uint_stream &infer_input, long_uint_stream &infer_output)
{
	// Define ip ports
#pragma HLS INTERFACE axis port=infer_input
#pragma HLS INTERFACE axis port=infer_output
#pragma HLS INTERFACE s_axilite port=return

	// Define arrays
	static fixed cnn_input_flat[input_dim1*input_dim2*input_dim3] = {};
	static fixed cnn_input[input_dim1][input_dim2][input_dim3] = {};
	static fixed layer_2_out[layer_2_dim1][layer_2_dim2][layer_2_dim3] = {};
	static fixed layer_3_out[layer_3_dim1][layer_3_dim2][layer_3_dim3] = {};
	static fixed layer_4_out[layer_4_dim1][layer_4_dim2][layer_4_dim3] = {};
	static fixed layer_5_out[layer_5_dim1][layer_5_dim2][layer_5_dim3] = {};
	static fixed layer_6_out[layer_6_dim1][layer_6_dim2][layer_6_dim3] = {};
	static fixed layer_7_out[layer_7_dim1][layer_7_dim2][layer_7_dim3] = {};
	static fixed layer_8_out[layer_8_dim1] = {};
	static fixed layer_9_out[layer_9_dim1] = {};
	static fixed layer_10_out[layer_10_dim1] = {};
	static fixed layer_11_out[layer_11_dim1] = {};
	static fixed cnn_output[output_dim1] = {};

	// Partition arrays
#pragma HLS array_partition variable=cnn_input_flat cyclic factor=2

#pragma HLS array_partition variable=layer_2_out cyclic factor=2 dim=1
#pragma HLS array_partition variable=layer_4_out cyclic factor=2 dim=1
#pragma HLS array_partition variable=layer_6_out cyclic factor=2 dim=1

    // Load image from stream to input array
	// Rescale at the same time
	long_uint_package input_package;
    get_input1: for (sizetype i = 0; i < input_dim1*input_dim2*input_dim3; i += 4)
    {
		infer_input.read(input_package);
		uint8_t pixel_1 = (input_package.data.to_int() & 0xFF000000) >> 24;
		uint8_t pixel_2 = (input_package.data.to_int() & 0xFF0000) >> 16;
		uint8_t pixel_3 = (input_package.data.to_int() & 0xFF00) >> 8;
		uint8_t pixel_4 = (input_package.data.to_int() & 0xFF);

		cnn_input_flat[i] = (fixed)((float)pixel_1 / 255.0);
		cnn_input_flat[i+1] = (fixed)((float)pixel_2 / 255.0);
		cnn_input_flat[i+2] = (fixed)((float)pixel_3 / 255.0);
		cnn_input_flat[i+3] = (fixed)((float)pixel_4 / 255.0);
    }
    
    // Change 1d array to 3d
    array_1d_to_3d(cnn_input_flat, cnn_input);

    // Layer 1 rescaling
    // Already done in image load


    // Layer 2 convolution
    layer_2_conv2d: conv2d(cnn_input,
            layer_2_weights,
            layer_2_bias,
			layer_2_out);
    

    // Layer 3 max pooling
    layer_3_max_pooling2d: max_pooling2d(layer_2_out,
    		layer_3_out);


    // Layer 4 convolution
    layer_4_conv2d: conv2d(layer_3_out,
            layer_4_weights,
            layer_4_bias,
			layer_4_out);


    // Layer 5 max pooling
    layer_5_max_pooling2d: max_pooling2d(layer_4_out,
    		layer_5_out);


    // Layer 6 convolution
    layer_6_conv2d: conv2d(layer_5_out,
            layer_6_weights,
            layer_6_bias,
			layer_6_out);


    // Layer 7 max pooling
    layer_7_max_pooling2d: max_pooling2d(layer_6_out,
    		layer_7_out);


    // Layer 8 flatten
    layer_8_flatten: array_3d_to_1d(layer_7_out, layer_8_out);


    // Layer 9 dense
    layer_9_dense_relu: dense_relu(layer_8_out,
            layer_9_weights,
            layer_9_bias,
			layer_9_out);

    // Layer 10 dense
    layer_10_dense_relu: dense_relu(layer_9_out,
            layer_10_weights,
            layer_10_bias,
			layer_10_out);


    // Layer 11 dense
    layer_11_dense_relu: dense_relu(layer_10_out,
            layer_11_weights,
            layer_11_bias,
			layer_11_out);

	
    // Layer 12 dense
    layer_12_dense: dense(layer_11_out,
            layer_12_weights,
            layer_12_bias,
			cnn_output);


    // Softmax the output
    softmax_output: softmax(cnn_output);


    //Send result
    long_uint_package output_package = input_package; // Borrow side channel information
    float_uint probability;
    send_result: for (int i = 0; i < output_dim1; i++)
    {
    	probability.float_val = cnn_output[i];
    	output_package.data = probability.uint_val;
    	if (i == output_dim1-1)
    		output_package.last = 1;
    	else
    		output_package.last = 0;
    	infer_output.write(output_package);
    }

}
