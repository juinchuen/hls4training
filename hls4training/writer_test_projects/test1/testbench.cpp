#include "myproject.h"
#include <stdio.h>

int main () {

    input_t input[10];

    layer0_weight_t layer0_w[11][10];
    layer0_bias_t layer0_b[11];

    layer1_weight_t layer1_w[12][11];
    layer1_bias_t layer1_b[12];

    layer2_weight_t layer2_w[13][12];
    layer2_bias_t layer2_b[13];

    layer2_post_relu_t layer2_post_relu[13];

    layer0_weight_grad_t layer0_w_grad[11][10];
    layer0_bias_grad_t layer0_b_grad[11];

    layer1_weight_grad_t layer1_w_grad[12][11];
    layer1_bias_grad_t layer1_b_grad[12];

    layer2_weight_grad_t layer2_w_grad[13][12];
    layer2_bias_grad_t layer2_b_grad[13];

    truth_t truth[13];

    

}