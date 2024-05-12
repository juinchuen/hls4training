#include "myproject.h"

void myproject (

    // WRITE IO LIST
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

){

    // WRITE INTERNAL VARIABLES
    input_act_grad_t input_act_grad[10];

    layer0_pre_relu_t layer0_pre_relu[11];
    layer0_post_relu_t layer0_post_relu[11];
    layer0_relu_grad_t layer0_relu_grad[11];
    layer0_act_grad_t layer0_act_grad[11];

    layer1_pre_relu_t layer1_pre_relu[12];
    layer1_post_relu_t layer1_post_relu[12];
    layer1_relu_grad_t layer1_relu_grad[12];
    layer1_act_grad_t layer1_act_grad[12];

    layer2_pre_relu_t layer2_pre_relu[13];
    layer2_relu_grad_t layer2_relu_grad[13];
    layer2_act_grad_t layer2_act_grad[13];

    // WRITE ARRAY PARTITION DIRECTIVES
    #pragma HLS ARRAY_PARTITION variable=input complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer0_pre_relu complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer1_pre_relu complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer2_pre_relu complete dim=0
    #pragma HLS ARRAY_PARTITION variable=truth complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer0_post_relu complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer1_post_relu complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer2_post_relu complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer0_relu_grad complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer1_relu_grad complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer2_relu_grad complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer0_w complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer1_w complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer2_w complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer0_b complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer1_b complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer2_b complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer0_w_grad complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer1_w_grad complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer2_w_grad complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer0_b_grad complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer1_b_grad complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer2_b_grad complete dim=0
    #pragma HLS ARRAY_PARTITION variable=input_act_grad complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer0_act_grad complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer1_act_grad complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer2_act_grad complete dim=0

// WRITE IO PROTOCOL DIRECTIVES

    // WRITE INFERENCE LAYERS                   
    nnet::dense_infer <dense_config_0> (input, layer0_pre_relu, layer0_w, layer0_b);
    nnet::relu <relu_config_0> (layer0_pre_relu, layer0_post_relu);

    nnet::dense_infer <dense_config_1> (layer0_post_relu, layer1_pre_relu, layer1_w, layer1_b);
    nnet::relu <relu_config_1> (layer1_pre_relu, layer1_post_relu);

    nnet::dense_infer <dense_config_2> (layer1_post_relu, layer2_pre_relu, layer2_w, layer2_b);
    nnet::relu <relu_config_2> (layer2_pre_relu, layer2_post_relu);

    // WRITE ERROR GRADIENT LAYER
    nnet::ms_grad <ms_grad_config_0> (layer2_post_relu, truth, layer2_act_grad);

    // WRITE BACKPROPAGATION LAYERS
    nnet::relu_grad <relu_config_2> (layer2_pre_relu, layer2_relu_grad);
    nnet::dense_backprop <dense_config_2> (layer2_relu_grad, layer1_post_relu, layer2_act_grad, layer2_w, layer2_w_grad, layer2_b_grad, layer1_act_grad);

    nnet::relu_grad <relu_config_1> (layer1_pre_relu, layer1_relu_grad);
    nnet::dense_backprop <dense_config_1> (layer1_relu_grad, layer0_post_relu, layer1_act_grad, layer1_w, layer1_w_grad, layer1_b_grad, layer0_act_grad);

    nnet::relu_grad <relu_config_0> (layer0_pre_relu, layer0_relu_grad);
    nnet::dense_backprop<dense_config_0> (layer0_relu_grad, input, layer0_act_grad, layer0_w, layer0_w_grad, layer0_b_grad, input_act_grad);

}