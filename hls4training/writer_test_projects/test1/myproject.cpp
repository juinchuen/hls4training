#include "project.h"

void project (

    // WRITE IO LIST
    data_T input[3],

    data_T layer0_w[5][3],
    data_T layer0_b[5],

    data_T layer0_post_relu[5],

    data_T layer0_w_grad[5][3],
    data_T layer0_b_grad[5],

    data_T truth[5]

){

    // WRITE INTERNAL VARIABLES
    data_T input_act_grad[3];

    data_T layer0_pre_relu[5];
    data_T layer0_relu_grad[5];
    data_T layer0_act_grad[5];

    // WRITE INFERENCE LAYERS                   
    nnet:dense_infer <data_T, dense_config_0> (input, layer0_pre_relu, layer0_w, layer0_b);
    nnet:relu <data_T, relu_config_0> (layer0_pre_relu, layer0_post_relu);

    // WRITE ERROR GRADIENT LAYER
    nnet::ms_grad <data_T, ms_grad_config> (layer0_post_relu, truth, layer0_act_grad);

    // WRITE BACKPROPAGATION LAYERS
    nnet::relu_grad <data_T, relu_config_0> (layer0_pre_relu, layer0_relu_grad);
    nnet::dense_backprop <data_T, dense_config_0> (layer0_relu_grad, input, layer0_act_grad, layer0_w, layer0_w_grad, layer0_b_grad, input_act_grad);

}