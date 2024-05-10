#include "ms_grad_config.h"
#pragma once

namespace nnet{

    template <class data_T, class res_T, class act_grad_T, typename CONFIG_T>

        void ms_grad(   data_T y_pred_in    [CONFIG_T::n_neuron],
                        res_T y_real_in    [CONFIG_T::n_neuron],
                        act_grad_T act_grad_out [CONFIG_T::n_neuron]) {

            for (int i = 0; i < CONFIG_T::n_neuron; i++){
			#pragma HLS UNROLL

                act_grad_out[i] = 2 * (y_pred_in[i] - y_real_in[i]);

            }

        }

}
