#ifndef TYPES_HPP
#define TYPES_HPP

#include "ap_fixed.h"
#include "stdint.h"

typedef ap_fixed<21,5> fixed;
typedef ap_fixed<40,32> softmax_fixed;
//typedef float fixed;
typedef uint16_t sizetype;

#include "hls_stream.h"
#include "ap_axi_sdata.h"

typedef ap_axiu<8,2,5,6> uint_package;
typedef ap_axiu<32,2,5,6> long_uint_package;

// Union to convert between float and uint
union float_uint {
	float float_val;
	uint32_t uint_val;
};

typedef hls::stream<uint_package> uint_stream;
typedef hls::stream<long_uint_package> long_uint_stream;

#endif
