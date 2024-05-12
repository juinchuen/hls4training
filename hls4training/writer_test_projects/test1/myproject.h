// WRITE INCLUDE DIRECTIVES
#include <ap_fixed.h>
#include "/homes/jco1147/hls4training/hls4training/utils/dense_infer.h"
#include "/homes/jco1147/hls4training/hls4training/utils/relu.h"
#include "/homes/jco1147/hls4training/hls4training/utils/relu_grad.h"
#include "/homes/jco1147/hls4training/hls4training/utils/ms_grad.h"
#include "/homes/jco1147/hls4training/hls4training/utils/dense_backprop.h"

// WRITE DATA TYPE DEFINITIONS
typedef ap_fixed<12, 3> input_t;
typedef ap_fixed<12, 3> truth_t;
typedef ap_fixed<12, 3> input_act_grad_t;
typedef ap_fixed<12, 3> layer0_pre_relu_t;
typedef ap_fixed<12, 3> layer0_post_relu_t;
typedef ap_fixed<12, 3> layer0_weight_t;
typedef ap_fixed<12, 3> layer0_bias_t;
typedef ap_fixed<12, 3> layer0_relu_grad_t;
typedef ap_fixed<12, 3> layer0_act_grad_t;
typedef ap_fixed<12, 3> layer0_weight_grad_t;
typedef ap_fixed<12, 3> layer0_bias_grad_t;
typedef ap_ufixed<11, 3> layer1_pre_relu_t;
typedef ap_ufixed<11, 3> layer1_post_relu_t;
typedef ap_ufixed<11, 3> layer1_weight_t;
typedef ap_ufixed<11, 3> layer1_bias_t;
typedef ap_ufixed<11, 3> layer1_relu_grad_t;
typedef ap_ufixed<11, 3> layer1_act_grad_t;
typedef ap_ufixed<11, 3> layer1_weight_grad_t;
typedef ap_ufixed<11, 3> layer1_bias_grad_t;
typedef ap_fixed<12, 3> layer2_pre_relu_t;
typedef ap_fixed<12, 3> layer2_post_relu_t;
typedef ap_fixed<12, 3> layer2_weight_t;
typedef ap_fixed<12, 3> layer2_bias_t;
typedef ap_fixed<12, 3> layer2_relu_grad_t;
typedef ap_fixed<12, 3> layer2_act_grad_t;
typedef ap_fixed<12, 3> layer2_weight_grad_t;
typedef ap_fixed<12, 3> layer2_bias_grad_t;

// WRITE DENSE CONFIG
struct dense_config_0 : nnet::dense_config {
    static const unsigned n_in = 10;
    static const unsigned n_out = 11;
    typedef input_t data_t;
    typedef layer0_post_relu_t res_t;
    typedef layer0_weight_t weight_t;
    typedef layer0_bias_t bias_t;
    typedef layer0_weight_grad_t weight_grad_t;
    typedef layer0_bias_grad_t bias_grad_t;
    typedef layer0_act_grad_t act_grad_in_t;
    typedef layer0_relu_grad_t relu_grad_t;
    typedef input_act_grad_t act_grad_out_t;
};

struct dense_config_1 : nnet::dense_config {
    static const unsigned n_in = 11;
    static const unsigned n_out = 12;
    typedef layer0_post_relu_t data_t;
    typedef layer1_post_relu_t res_t;
    typedef layer1_weight_t weight_t;
    typedef layer1_bias_t bias_t;
    typedef layer1_weight_grad_t weight_grad_t;
    typedef layer1_bias_grad_t bias_grad_t;
    typedef layer1_act_grad_t act_grad_in_t;
    typedef layer1_relu_grad_t relu_grad_t;
    typedef layer0_act_grad_t act_grad_out_t;
};

struct dense_config_2 : nnet::dense_config {
    static const unsigned n_in = 12;
    static const unsigned n_out = 13;
    typedef layer1_post_relu_t data_t;
    typedef layer2_post_relu_t res_t;
    typedef layer2_weight_t weight_t;
    typedef layer2_bias_t bias_t;
    typedef layer2_weight_grad_t weight_grad_t;
    typedef layer2_bias_grad_t bias_grad_t;
    typedef layer2_act_grad_t act_grad_in_t;
    typedef layer2_relu_grad_t relu_grad_t;
    typedef layer1_act_grad_t act_grad_out_t;
};

// WRITE RELU CONFIG
struct relu_config_0 : nnet::relu_config {
    static const unsigned n_neuron = 11;
    typedef layer0_pre_relu_t data_t;
    typedef layer0_post_relu_t relu_t;
    typedef layer0_relu_grad_t grad_t;
};

struct relu_config_1 : nnet::relu_config {
    static const unsigned n_neuron = 12;
    typedef layer1_pre_relu_t data_t;
    typedef layer1_post_relu_t relu_t;
    typedef layer1_relu_grad_t grad_t;
};

struct relu_config_2 : nnet::relu_config {
    static const unsigned n_neuron = 13;
    typedef layer2_pre_relu_t data_t;
    typedef layer2_post_relu_t relu_t;
    typedef layer2_relu_grad_t grad_t;
};

// WRITE ERROR GRADIENT CONFIG
struct ms_grad_config_0 : nnet::ms_grad_config {
    static const unsigned n_neuron = 13;
    typedef layer2_post_relu_t pred_t;
    typedef truth_t real_t;
    typedef layer2_act_grad_t act_grad_t;
};

// WRITE FUNCTION PROTOTYPE

void myproject (

    // WRITE FUNCTION IO
    input_t input[10],

    layer0_weight_t layer0_w[11][10],
    layer0_bias_t layer0_b[11],

    layer1_weight_t layer1_w[12][11],
    layer1_bias_t layer1_b[12],

    layer2_weight_t layer2_w[13][12],
    layer2_bias_t layer2_b[13],

    layer2_post_relu_t layer2_post_relu[13],

    layer0_weight_grad_t layer0_w_grad[11][10],
    layer0_bias_grad_t layer0_b_grad[11],

    layer1_weight_grad_t layer1_w_grad[12][11],
    layer1_bias_grad_t layer1_b_grad[12],

    layer2_weight_grad_t layer2_w_grad[13][12],
    layer2_bias_grad_t layer2_b_grad[13],

    truth_t truth[13]

);