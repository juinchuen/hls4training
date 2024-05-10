open_project myproject -reset
set_top myproject
add_files myproject.cpp
add_files myproject.h

# ADD UTIL FILES
add_files /homes/jco1147/hls4training/hls4training/utils/dense_infer.h
add_files /homes/jco1147/hls4training/hls4training/utils/relu.h
add_files /homes/jco1147/hls4training/hls4training/utils/relu_grad.h
add_files /homes/jco1147/hls4training/hls4training/utils/ms_grad.h
add_files /homes/jco1147/hls4training/hls4training/utils/dense_backprop.h

open_solution "solution1" -reset
set_part {xczu9eg-ffvb1156-2-e} -tool vivado
create_clock -period 10 -name default
csynth_design
export_design -rtl verilog -format ip_catalog
exit