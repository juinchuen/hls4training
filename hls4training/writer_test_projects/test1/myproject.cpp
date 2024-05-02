#include "myproject.h"

void myproject (

    // WRITE IO LIST
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

){

    // WRITE INTERNAL VARIABLES
    data_T input_act_grad[5];

    data_T layer0_pre_relu[16];
    data_T layer0_post_relu[16];
    data_T layer0_relu_grad[16];
    data_T layer0_act_grad[16];

    data_T layer1_pre_relu[16];
    data_T layer1_post_relu[16];
    data_T layer1_relu_grad[16];
    data_T layer1_act_grad[16];

    data_T layer2_pre_relu[5];
    data_T layer2_relu_grad[5];
    data_T layer2_act_grad[5];

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
    nnet::dense_infer <data_T, dense_config_0> (input, layer0_pre_relu, layer0_w, layer0_b);
    nnet::relu <data_T, relu_config_0> (layer0_pre_relu, layer0_post_relu);

    nnet::dense_infer <data_T, dense_config_1> (layer0_post_relu, layer1_pre_relu, layer1_w, layer1_b);
    nnet::relu <data_T, relu_config_1> (layer1_pre_relu, layer1_post_relu);

    nnet::dense_infer <data_T, dense_config_2> (layer1_post_relu, layer2_pre_relu, layer2_w, layer2_b);
    nnet::relu <data_T, relu_config_2> (layer2_pre_relu, layer2_post_relu);

    // WRITE ERROR GRADIENT LAYER
    nnet::ms_grad <data_T, ms_grad_config_0> (layer2_post_relu, truth, layer2_act_grad);

    // WRITE BACKPROPAGATION LAYERS
    nnet::relu_grad <data_T, relu_config_2> (layer2_pre_relu, layer2_relu_grad);
    nnet::dense_backprop <data_T, dense_config_2> (layer2_relu_grad, layer1_post_relu, layer2_act_grad, layer2_w, layer2_w_grad, layer2_b_grad, layer1_act_grad);

    nnet::relu_grad <data_T, relu_config_1> (layer1_pre_relu, layer1_relu_grad);
    nnet::dense_backprop <data_T, dense_config_1> (layer1_relu_grad, layer0_post_relu, layer1_act_grad, layer1_w, layer1_w_grad, layer1_b_grad, layer0_act_grad);

    nnet::relu_grad <data_T, relu_config_0> (layer0_pre_relu, layer0_relu_grad);
    nnet::dense_backprop <data_T, dense_config_0> (layer0_relu_grad, input, layer0_act_grad, layer0_w, layer0_w_grad, layer0_b_grad, input_act_grad);

}