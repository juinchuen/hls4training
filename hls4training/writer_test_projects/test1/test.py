import sys

sys.path.append("../../writers")

from vivado_writer import vivado_writer

project_dir = "../test1/"
template_dir = "/homes/dcc3637/Documents/hls4training/hls4training/templates"
util_dir = "/homes/dcc3637/Documents/hls4training/hls4training/utils"

#include "utils/dense_infer.h"
#include "utils/relu.h"
#include "utils/relu_grad.h"
#include "utils/ms_grad.h"
#include "utils/dense_backprop.h"

headers = [ "dense_infer.h",
            "relu.h",
            "relu_grad.h",
            "ms_grad.h",
            "dense_backprop.h"]

layer_names = ["layer0", "layer1", "layer2"]
layer_sizes = [10,11,12,13]

in_q = ['signed', 12, 3]
out_q = ['signed', 12, 21]
in_act_grad = ['signed', 12, 3]

io_q = [in_q, out_q, in_act_grad]


# Order is weight bias and gradient weight and gradient bias
layer1_q = [['signed', 12, 3], ['signed', 12, 3], ['unsigned', 12, 3], ['unsigned', 12, 3], ['unsigned', 11, 3], ['signed', 12, 3], ['unsigned', 12, 3], ['unsigned', 12, 3]] 
layer2_q = [['unsigned', 11, 3], ['signed', 12, 3], ['unsigned', 12, 3], ['unsigned', 12, 3], ['unsigned', 9, 4], ['signed', 12, 3], ['signed', 12, 32], ['signed', 12, 3]]
layer3_q = [['signed', 12, 3], ['signed', 12, 32], ['signed', 12, 3], ['unsigned', 22, 12], ['unsigned', 11, 3], ['signed', 12, 3], ['unsigned', 12, 3], ['unsigned', 12, 3]] 
layer4_q = [['signed', 11, 3], ['signed', 9, 13], ['unsigned', 12, 3], ['unsigned', 8, 3], ['signed', 12, 3], ['signed', 12, 32], ['signed', 12, 3], ['unsigned', 22, 12]]
layer_quantization = [io_q, layer1_q, layer2_q, layer3_q]

writer = vivado_writer(project_dir=project_dir, template_dir=template_dir, util_dir=util_dir)

writer.set_names_and_sizes(layer_names=layer_names, layer_sizes=layer_sizes, layer_quantization=layer_quantization )

writer.write_project_cpp(layer_names=layer_names, layer_sizes=layer_sizes)

writer.write_header(layer_names=layer_names, layer_sizes=layer_sizes, headers=headers)
