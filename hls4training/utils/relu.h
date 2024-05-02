#include "relu_config.h"
#pragma once
namespace nnet{

    template <class data_T, typename CONFIG_T>

        void relu( data_T data_in  [CONFIG_T::n_neuron],
                        data_T data_out [CONFIG_T::n_neuron]) {

            for (int i = 0; i < CONFIG_T::n_neuron; i++){
            #pragma HLS UNROLL


                if (data_in[i] > 0) data_out[i] = data_in[i];

                else data_out[i] = 0;

            }

        }

}
