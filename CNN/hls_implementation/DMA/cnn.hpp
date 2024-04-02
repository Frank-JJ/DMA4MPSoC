#ifndef CNN_HPP
#define CNN_HPP


#include "hls_exp_apfixed.h"
#include "types.hpp"


#ifndef LAYER_INFO
#define LAYER_INFO
#include "layerInfo.hpp"
#endif


void relu(fixed &input);

template <const sizetype input_size_1, const sizetype input_size_2, const sizetype input_size_3,
			const sizetype weights_size_1, const sizetype weights_size_2, const sizetype weights_size_3, const sizetype weights_size_4,
			const sizetype bias_size,
			const sizetype output_size_1, const sizetype output_size_2, const sizetype output_size_3>
void conv2d(fixed (&input)[input_size_1][input_size_2][input_size_3],
            const fixed (&weights)[weights_size_1][weights_size_2][weights_size_3][weights_size_4],
			const fixed (&bias)[bias_size],
            fixed (&output)[output_size_1][output_size_2][output_size_3]);

template <const sizetype input_size_1, const sizetype input_size_2, const sizetype input_size_3,
            const sizetype output_size_1, const sizetype output_size_2, const sizetype output_size_3>
void max_pooling2d(fixed (&input)[input_size_1][input_size_2][input_size_3],
                fixed (&output)[output_size_1][output_size_2][output_size_3]);

template <const sizetype size_1, const sizetype size_2, const sizetype size_3>
void array_3d_to_1d(fixed (&array)[size_1][size_2][size_3], fixed (&output)[size_1*size_2*size_3]);

template <const sizetype size_1, const sizetype size_2, const sizetype size_3>
void array_1d_to_3d(fixed (&array)[size_1*size_2*size_3], fixed (&output)[size_1][size_2][size_3]);

template <const sizetype input_size,
            const sizetype weights_size_1, const sizetype weights_size_2,
            const sizetype bias_size,
            const sizetype output_size>
void dense_relu(fixed (&input)[input_size],
            const fixed (&weights)[weights_size_1][weights_size_2],
            const fixed (&bias)[bias_size],
            fixed (&output)[output_size]);

template <const sizetype input_size,
            const sizetype weights_size_1, const sizetype weights_size_2,
            const sizetype bias_size,
            const sizetype output_size>
void dense(fixed (&input)[input_size],
            const fixed (&weights)[weights_size_1][weights_size_2],
            const fixed (&bias)[bias_size],
            fixed (&output)[output_size]);

template <const sizetype size>
void softmax(fixed (&array)[size]);

void infer(long_uint_stream &infer_input, long_uint_stream &infer_output);

#endif
