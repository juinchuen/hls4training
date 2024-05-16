#include "dense_config.h"
#pragma once
namespace nnet {

    // template <class data_T, class res_T, typename CONFIG_T>
    //     void dense_infer(   data_T data_in  [CONFIG_T::n_in],
    //                         res_T data_out [CONFIG_T::n_out],
    //                         typename CONFIG_T::weight_t weight[CONFIG_T::n_out][CONFIG_T::n_in],
    //                         typename CONFIG_T::bias_t bias[CONFIG_T::n_out]){

    template <typename CONFIG_T>

        void dense_infer(   typename CONFIG_T::data_t data_in [CONFIG_T::n_in],
                            typename CONFIG_T::res_t data_out [CONFIG_T::n_out],
                            typename CONFIG_T::weight_t weight [CONFIG_T::n_out][CONFIG_T::n_in],
                            typename CONFIG_T::bias_t bias [CONFIG_T::n_out]){
        #pragma HLS PIPELINE
            
            // initialize output values with bias

            for (int i = 0; i < CONFIG_T::n_out; i++){
            #pragma HLS unroll

                data_out[i] = bias[i];

            }

            // matrix multiplication

            for (int i = 0; i < CONFIG_T::n_in; i++){
            #pragma HLS UNROLL

                for (int j = 0; j < CONFIG_T::n_out; j++){
                #pragma HLS UNROLL

                    data_out[j] += data_in[i] * weight[j][i];

                }

            }
    
        }

}
