#include "relu_config.h"
#pragma once
namespace nnet{

    template <typename CONFIG_T>

        void relu( typename CONFIG_T::data_t data_in  [CONFIG_T::n_neuron],
                   typename CONFIG_T::relu_t data_out [CONFIG_T::n_neuron]) {

            for (int i = 0; i < CONFIG_T::n_neuron; i++){
            #pragma HLS UNROLL

                if (data_in[i] > 0) data_out[i] = data_in[i];

                else data_out[i] = 0;

            }

        }

}
