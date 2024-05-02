// WRITE INCLUDE DIRECTIVES
#include "/homes/jco1147/hls4training/hls4training/utils/dense_infer.h"
#include "/homes/jco1147/hls4training/hls4training/utils/relu.h"
#include "/homes/jco1147/hls4training/hls4training/utils/relu_grad.h"
#include "/homes/jco1147/hls4training/hls4training/utils/ms_grad.h"
#include "/homes/jco1147/hls4training/hls4training/utils/dense_backprop.h"

// WRITE DATA TYPE DEFINITIONS
typedef int data_T;

// WRITE DENSE CONFIG
struct dense_config_0 : nnet::dense_config {
    static const unsigned n_in = 5;
    static const unsigned n_out = 16;
};

struct dense_config_1 : nnet::dense_config {
    static const unsigned n_in = 16;
    static const unsigned n_out = 16;
};

struct dense_config_2 : nnet::dense_config {
    static const unsigned n_in = 16;
    static const unsigned n_out = 5;
};

// WRITE RELU CONFIG
struct relu_config_0 : nnet::relu_config {
    static const unsigned n_neuron = 16;
};

struct relu_config_1 : nnet::relu_config {
    static const unsigned n_neuron = 16;
};

struct relu_config_2 : nnet::relu_config {
    static const unsigned n_neuron = 5;
};

// WRITE ERROR GRADIENT CONFIG
struct ms_grad_config_0 : nnet::ms_grad_config {
    static const unsigned n_neuron = 5;
};

// WRITE FUNCTION PROTOTYPE

void myproject (

    // WRITE FUNCTION IO
    data_T input[5],

    data_T layer0_w[16][5],
    data_T layer0_b[16],

    data_T layer1_w[16][16],
    data_T layer1_b[16],

    data_T layer2_w[5][16],
    data_T layer2_b[5],

    data_T layer2_post_relu[5],

    data_T layer0_w_grad[16][5],
    data_T layer0_b_grad[16],

    data_T layer1_w_grad[16][16],
    data_T layer1_b_grad[16],

    data_T layer2_w_grad[5][16],
    data_T layer2_b_grad[5],

    data_T truth[5]

);