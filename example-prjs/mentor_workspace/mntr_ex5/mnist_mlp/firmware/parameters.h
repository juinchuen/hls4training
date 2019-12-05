#ifndef PARAMETERS_H_
#define PARAMETERS_H_

//#include <complex>
//#include "ap_int.h"
//#include "ap_fixed.h"
//#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_dense_large.h"
//#include "nnet_utils/nnet_dense_compressed.h"
//#include "nnet_utils/nnet_conv.h"
//#include "nnet_utils/nnet_conv2d.h"
#include "nnet_utils/nnet_activation.h"
//#include "nnet_utils/nnet_common.h"
//#include "nnet_utils/nnet_batchnorm.h"
//#include "nnet_utils/nnet_pooling.h"
//#include "nnet_utils/nnet_merge.h"
//#include "nnet_utils/nnet_helpers.h"

//hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 784
#define N_LAYER_2 64
#define N_LAYER_4 64
#define N_LAYER_6 10

//hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<18,8> model_default_t;
typedef ap_fixed<18,8> input_t;
typedef ap_fixed<18,8> layer2_t;
typedef ap_fixed<18,8> layer3_t;
typedef ap_fixed<18,8> layer4_t;
typedef ap_fixed<18,8> layer5_t;
typedef ap_fixed<18,8> layer6_t;
typedef ap_fixed<18,8> result_t;

//hls-fpga-machine-learning insert layer-config
struct config2 : nnet::dense_config {
    static const unsigned n_in = N_INPUT_1_1;
    static const unsigned n_out = N_LAYER_2;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 3136; //(50176); // 3136 = 784*4
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 50176;
    static const bool store_weights_in_bram = true;
    typedef ap_fixed<18,8> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    typedef ap_uint<1> index_t;
};

struct relu_config3 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_2;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
};

struct config4 : nnet::dense_config {
    static const unsigned n_in = N_LAYER_2;
    static const unsigned n_out = N_LAYER_4;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 256; //(4096); // 256 = 64*4
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 4096;
    static const bool store_weights_in_bram = true;
    typedef ap_fixed<18,8> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    typedef ap_uint<1> index_t;
};

struct relu_config5 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_4;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
};

struct config6 : nnet::dense_config {
    static const unsigned n_in = N_LAYER_4;
    static const unsigned n_out = N_LAYER_6;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 128; //(640); // 128 = 64*2
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 640;
    static const bool store_weights_in_bram = true;
    typedef ap_fixed<18,8> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    typedef ap_uint<1> index_t;
};

struct softmax_config7 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_6;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
};


#endif