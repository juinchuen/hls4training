#include "utils/dense_infer.h"
#include "utils/relu.h"
#include "utils/relu_grad.h"
#include "utils/ms_grad.h"
#include "utils/dense_backprop.h"
#include <stdio.h>

typedef int data_T;

struct dense_config_0 : nnet::dense_config {

    static const unsigned n_in = 3;
    static const unsigned n_out = 5;

};

struct relu_config_0 : nnet::relu_config {

    static const unsigned n_neuron = 5;

};

struct ms_grad_config_0 : nnet::ms_grad_config{

    static const unsigned n_neuron = 5;

};

void project (

    data_T data_in[3],
    data_T data_0_relu[5],

    data_T y_truth[5],

    data_T act_grad[5],

    data_T weight[5][3],

    data_T weight_grad[5][3],
    data_T bias_grad[5],

    data_T act_grad_temp[3],

    data_T bias[5]

){

    // data_T data_in[3] = {1,2,3};
    data_T data_0_pre_relu[5];
    // data_T data_0_relu[5];
    data_T data_0_pre_relu_grad[5];

    // data_T y_truth[5] = {7,7,7,7,7};

    // data_T act_grad[5];

    // data_T weight[3][5] =   {{15,32,32,28, 5},
    //                          { 8, 0,22,11, 6},
    //                          {27,29,29,17,24}};

    // data_T weight_grad[3][5];
    // data_T bias_grad[5];
    // data_T act_grad_temp[3];

    // data_T bias[5] = {29,31,13, 7,16};

    // ##### inference ##### //

    // layer 0
    nnet::dense_infer <data_T, dense_config_0> (data_in, data_0_pre_relu, weight, bias);    
    nnet::relu <data_T, relu_config_0> (data_0_pre_relu, data_0_relu);                      

    // ##### error gradient ##### //

    nnet::ms_grad <data_T, ms_grad_config_0> (data_0_relu, y_truth, act_grad);

    // ##### backpropagation ##### //

    // layer 0
    nnet::relu_grad <data_T, relu_config_0> (data_0_pre_relu, data_0_pre_relu_grad);
    nnet::dense_backprop <data_T, dense_config_0> ( data_0_pre_relu_grad, data_in, act_grad, weight, weight_grad, bias_grad, act_grad_temp);

}
