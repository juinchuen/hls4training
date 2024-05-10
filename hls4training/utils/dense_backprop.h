#include "dense_config.h"
#pragma once
namespace nnet{

        template <class pre_relu_T, class prev_act_in_T, class act_out_T, typename CONFIG_T>
            void dense_backprop(
                pre_relu_T pre_relu_grad[CONFIG_T::n_out],
                prev_act_in_T prev_act_in[CONFIG_T::n_in],
                typename CONFIG_T::act_grad_t act_grad_in[CONFIG_T::n_out],
                typename CONFIG_T::weight_t weight_in[CONFIG_T::n_out][CONFIG_T::n_in],
                typename CONFIG_T::weight_grad_t weight_grad_out[CONFIG_T::n_out][CONFIG_T::n_in],
                typename CONFIG_T::bias_grad_t bias_grad_out[CONFIG_T::n_out],
                act_out_T act_grad_out[CONFIG_T::n_in]) {
                        // ----- bias gradient -----//

            for (int i = 0; i < CONFIG_T::n_out; i++){
            #pragma HLS UNROLL

            // bias gradient

                bias_grad_out[i] = pre_relu_grad[i] * act_grad_in[i];

            }

            // self._bias_dif.insert(0,np.multiply(self.relu_dif(self._pre_relu[-(i+1)]), self._activation_dif[0]))


            // ----- weight gradient ----- //

            for (int i = 0; i < CONFIG_T::n_out; i++){
            #pragma HLS UNROLL


                for (int j = 0; j < CONFIG_T::n_in; j++){
                #pragma HLS UNROLL


                    weight_grad_out[i][j] = bias_grad_out[i] * prev_act_in[j];

                }

            }

            // self._weight_dif.insert(0,np.dot(self._bias_dif[0], self._activation[-(i+2)].transpose()))

            // ----- activation gradient ----- //
            
            for (int i = 0; i < CONFIG_T::n_in; i++){
            #pragma HLS UNROLL


                act_grad_out[i] = 0;

            }

            for (int i = 0; i < CONFIG_T::n_out; i++){
            #pragma HLS UNROLL


                for (int j = 0; j < CONFIG_T::n_in; j++){
                #pragma HLS UNROLL


                    act_grad_out[j] += weight_in[i][j] * bias_grad_out[i];

                }

            }
            
            // self._activation_dif.insert(0,np.dot(self._weight[-(i+1)].transpose(), self._bias_dif[0]))
        
        }

}
