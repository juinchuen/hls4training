#include "dense_config.h"
#pragma once
namespace nnet {

    template <typename data_T, typename CONFIG_T>

        void dense_infer(   data_T data_in  [CONFIG_T::n_in],
                            data_T data_out [CONFIG_T::n_out],
                            data_T weight   [CONFIG_T::n_out][CONFIG_T::n_in],
                            data_T bias     [CONFIG_T::n_out]){
            
            // initialize output values with bias

            for (int i = 0; i < CONFIG_T::n_out; i++){

                data_out[i] = bias[i];

            }

            // matrix multiplication

            for (int i = 0; i < CONFIG_T::n_in; i++){

                for (int j = 0; j < CONFIG_T::n_out; j++){

                    data_out[j] += data_in[i] * weight[j][i];

                }

            }
    
        }

}
