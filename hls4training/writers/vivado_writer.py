import os

class vivado_writer:

    def __init__(self, project_dir = '', template_dir = '', util_dir=''):

        self.project_dir = project_dir
        self.template_dir = template_dir
        self.util_dir = util_dir

        # self.net_shape = net_shape

    def write_project_dir(self):

        if not os.path.isdir(self.project_dir):
            os.makedirs(self.project_dir)

    def set_names_and_sizes(self, layer_names, layer_sizes, layer_quantization):
        if(len(layer_names) + 1 != len(layer_quantization)):
            print("Error: Number of layer names doesn't match number of layers that are set to be quantized")
        self.layer_names = layer_names
        self.layer_sizes = layer_sizes
        self.layer_quantization = layer_quantization
        self.generate_variable_names()

    def generate_variable_names(self):

        self.pre_relu_names = ['input']
        self.post_relu_names = ['truth']
        self.relu_grad_names = []
        self.weight_names = []
        self.weight_grad_names = []
        self.bias_names = []
        self.bias_grad_names = []
        self.act_grad_names = ['input_act_grad']

        for i in range(len(self.layer_names)):

            self.pre_relu_names.append(self.layer_names[i] + '_pre_relu')
            self.post_relu_names.append(self.layer_names[i] + '_post_relu')
            self.relu_grad_names.append(self.layer_names[i] + '_relu_grad')

            self.weight_names.append(self.layer_names[i] + '_w')
            self.bias_names.append(self.layer_names[i] + '_b')

            self.weight_grad_names.append(self.layer_names[i] + '_w_grad')
            self.bias_grad_names.append(self.layer_names[i] + '_b_grad')

            self.act_grad_names.append(self.layer_names[i] + '_act_grad')

        self.all_names = [self.pre_relu_names,
                          self.post_relu_names,
                          self.relu_grad_names,
                          self.weight_names,
                          self.bias_names,
                          self.weight_grad_names,
                          self.bias_grad_names,
                          self.act_grad_names]

        self.data_types = []
        
        for i, layer in enumerate(self.layer_quantization):

            if i > 0: 
                self.data_types.append({  
                                        'pre_relu' :  self._create_ap_variables(layer[0]),
                                        'post_relu':  self._create_ap_variables(layer[1]),
                                        'width' :     self._create_ap_variables(layer[2]),
                                        'bias'  :     self._create_ap_variables(layer[3]),
                                        'relu_grad' : self._create_ap_variables(layer[4]),
                                        'act_grad' :  self._create_ap_variables(layer[5]),
                                        'width_grad': self._create_ap_variables(layer[6]),
                                        'bias_grad' : self._create_ap_variables(layer[7])
                                    })
            else:
                self.data_types.append({  
                                        'input' : self._create_ap_variables(layer[0]),
                                        'truth' : self._create_ap_variables(layer[1]),
                                        'input_act_grad' : self._create_ap_variables(layer[2])

                                }) 

        

    def _create_ap_variables(self, ap_var):
        curr_ind = 0
        if isinstance(ap_var[curr_ind], str):
            signed = not (ap_var[curr_ind].lower() == 'unsigned')
            if signed and ap_var[curr_ind].lower() != 'signed':
                print("Warning: Sign convention for input is ", ap_var[curr_ind]," which isn't recongnized. Defualting to signed")
            curr_ind += 1
        else:
            signed = True

        num_bit = ap_var[curr_ind]
        num_int = ap_var[curr_ind + 1]

        ap_type = f"ap_fixed<{num_bit}, {num_int}>" if signed else f"ap_ufixed<{num_bit}, {num_int}>"
        return ap_type



    def write_project_cpp(self, layer_names, layer_sizes):

        if (len(layer_names) != len(layer_sizes) - 1):

            # check whether layer_names and layer_sizes argument sizes are correct

            raise ValueError('Incorrect length of arguments. layer_names refers to computation layers. layer_sizes refers to variables in between computation layers')

        f_in = open(os.path.join(self.template_dir, 'project.cpp'), 'r')

        f_out = open(os.path.join(self.project_dir, 'myproject.cpp'), 'w')

        indent = '    '

        for line in f_in.readlines():

            if 'project' in line:

                line = line.replace('project', 'myproject')

            if '// WRITE IO LIST' in line:

                newline = ''

                newline += indent + line

                # write input variable

                newline += indent + self.pre_relu_names[0] + '_t input[' + str(layer_sizes[0]) + '],\n\n'

                # write weight and bias input variables

                for i in range(len(layer_names)):

                    newline += indent + layer_names[i] + '_width_t ' + layer_names[i] + '_w[' +  str(layer_sizes[i+1]) + '][' + str(layer_sizes[i]) + '],\n'
                    newline += indent + layer_names[i] + '_bias_t '+ layer_names[i] + '_b[' +  str(layer_sizes[i+1]) + '],\n\n'

                # write inference output variables

                newline += indent + self.post_relu_names[-1] + '_t ' + layer_names[-1] + '_post_relu[' + str(layer_sizes[-1]) + '],\n\n'

                # write gradient output variables

                for i in range(len(layer_names)):
                    newline += indent + layer_names[i] + '_width_grad_t ' + layer_names[i] + '_w_grad[' +  str(layer_sizes[i+1]) + '][' + str(layer_sizes[i]) + '],\n'
                    newline += indent + layer_names[i] + '_bias_grad_t ' + layer_names[i] + '_b_grad[' +  str(layer_sizes[i+1]) + '],\n\n'

                # write ground truth input variable

                newline += indent + self.post_relu_names[0] + '_t truth[' + str(layer_sizes[-1]) + ']\n'

            elif '// WRITE INTERNAL VARIABLES' in line:

                newline = ''

                newline += indent + line

                # write inference internal variables

                newline += indent + 'input_act_grad_t input_act_grad[' + str(layer_sizes[0]) + '];\n\n'

                for i in range(len(layer_names)-1):

                    newline += indent + layer_names[i] + '_pre_relu_t ' + layer_names[i] + '_pre_relu[' + str(layer_sizes[i+1]) + '];\n'
                    newline += indent + layer_names[i] + '_post_relu_t ' + layer_names[i] + '_post_relu[' + str(layer_sizes[i+1]) + '];\n'
                    newline += indent + layer_names[i] + '_relu_grad_t '+ layer_names[i] + '_relu_grad[' + str(layer_sizes[i+1]) + '];\n'
                    newline += indent + layer_names[i] + '_act_grad_t ' + layer_names[i] + '_act_grad[' + str(layer_sizes[i+1]) + '];\n\n'
                    

                newline += indent + layer_names[-1] + '_pre_relu_t ' + layer_names[-1] + '_pre_relu[' + str(layer_sizes[-1]) + '];\n'
                newline += indent + layer_names[-1] + '_relu_grad_t ' + layer_names[-1] + '_relu_grad[' + str(layer_sizes[-1]) + '];\n' 
                newline += indent + layer_names[-1] + '_act_grad_t ' + layer_names[-1] + '_act_grad[' + str(layer_sizes[-1]) + '];\n'

            elif '// WRITE ARRAY PARTITION DIRECTIVES' in line:

                # spam partition directives for everything

                newline = ''

                newline += indent + line

                def generate_pragma(var_name):

                    return '#pragma HLS ARRAY_PARTITION variable=' + var_name + ' complete dim=0\n'
                
                self.generate_variable_names()

                for g in self.all_names:

                    for v in g:

                        newline += indent + generate_pragma(v)
                    

                #pragma HLS ARRAY_PARTITION variable=layer0_w complete dim=0
                
            elif '// WRITE INFERENCE LAYERS ' in line:

                newline = ''

                newline += indent + line

                newline += indent
                newline += 'nnet::dense_infer <input_t, layer0_pre_relu_t, dense_config_0> (input, '
                newline += layer_names[0] + '_pre_relu, ' + layer_names[0] + '_w, ' + layer_names[0] + '_b);\n'

                newline += indent
                newline += f'nnet::relu <{layer_names[0]}_pre_relu_t, {layer_names[0]}_post_relu_t, relu_config_0> ('
                newline += layer_names[0] + '_pre_relu, ' + layer_names[0] + '_post_relu);'

                if len(layer_names) > 1:

                    newline += '\n\n'

                for i in range(len(layer_names)-1):

                    newline += indent
                    newline += f'nnet::dense_infer <{layer_names[i]}_post_relu_t, {layer_names[i + 1]}_pre_relu_t, dense_config_' + str(i+1) + '> (' + layer_names[i] + '_post_relu, '
                    newline += layer_names[i+1] + '_pre_relu, ' + layer_names[i+1] + '_w, ' + layer_names[i+1] + '_b);\n'

                    newline += indent
                    newline += f'nnet::relu <{layer_names[i+1]}_pre_relu_t, {layer_names[i+1]}_post_relu_t, relu_config_' + str(i+1) + '> ('
                    newline += layer_names[i+1] + '_pre_relu, ' + layer_names[i+1] + '_post_relu);'

                    if i != len(layer_names)-2:
                        newline += '\n\n'                

            elif '// WRITE ERROR GRADIENT LAYER' in line:

                newline += indent
                newline += line
                newline += indent
                newline += f'nnet::ms_grad <{layer_names[-1]}_post_relu_t, truth_t, {layer_names[-1]}_act_grad_t, ms_grad_config_0> ('
                newline += layer_names[-1] + '_post_relu, truth, ' + layer_names[-1] + '_act_grad);'

            elif '// WRITE BACKPROPAGATION LAYERS' in line:

                newline += indent
                newline += line

                for i in range(len(layer_names)-1):

                    j = len(layer_names) - i - 1

                    newline += indent

                    newline += f'nnet::relu_grad <{layer_names[j]}_pre_relu_t, {layer_names[j]}_relu_grad_t, relu_config_' + str(j) + '> ('
                    newline += layer_names[j] + '_pre_relu, ' + layer_names[j] + '_relu_grad);\n'

                    newline += indent
                    newline += f'nnet::dense_backprop <{layer_names[j]}_relu_grad_t, {layer_names[j-1]}_post_relu_t, {layer_names[j-1]}_act_grad_t, dense_config_' + str(j) + '> ('
                    newline += layer_names[j] + '_relu_grad, ' + layer_names[j-1] + '_post_relu, '
                    newline += layer_names[j] + '_act_grad, ' + layer_names[j] + '_w, '
                    newline += layer_names[j] + '_w_grad, ' + layer_names[j] + '_b_grad, '
                    newline += layer_names[j-1] + '_act_grad);\n\n'

                newline += indent
                newline += f'nnet::relu_grad <{layer_names[0]}_pre_relu_t, {layer_names[0]}_relu_grad_t, relu_config_0> ('
                newline += layer_names[0] + '_pre_relu, ' + layer_names[0] + '_relu_grad);\n'

                newline += indent
                newline += f'nnet::dense_backprop<{layer_names[0]}_relu_grad_t, input_t, input_act_grad_t, dense_config_0> ('
                newline += layer_names[0] + '_relu_grad, input, '
                newline += layer_names[0] + '_act_grad, ' + layer_names[0] + '_w, '
                newline += layer_names[0] + '_w_grad, ' + layer_names[0] + '_b_grad, '
                newline += 'input_act_grad);\n'

                # nnet::dense_backprop <data_t, dense_config_0>
                # ( data_0_pre_relu_grad, data_in, act_grad, weight, weight_grad, bias_grad, act_grad_temp);

            else:

                newline = line

            f_out.write(newline)

        f_in.close()
        f_out.close()

    def write_header (self, layer_names, layer_sizes, headers):

        if (len(layer_names) != len(layer_sizes) - 1):

            # check whether layer_names and layer_sizes argument sizes are correct

            raise ValueError('Incorrect length of arguments. layer_names refers to computation layers. layer_sizes refers to variables in between computation layers')

        f_in = open(os.path.join(self.template_dir, 'project.h'), 'r')

        f_out = open(os.path.join(self.project_dir, 'myproject.h'), 'w')

        indent = '    '

        for line in f_in.readlines():

            if 'project' in line:

                line = line.replace('project', 'myproject')

            newline = ''

            if "// WRITE INCLUDE DIRECTIVES" in line:

                newline += line
                newline += '#include <ap_fixed.h>\n'
                for h in headers:
                    newline += '#include \"' + self.util_dir + '/' + h + '\"\n'

            elif "// WRITE DATA TYPE DEFINITIONS" in line:

                newline += line
                for i, data_dict in enumerate(self.data_types):
                    for key, val in data_dict.items():
                        if i > 0:
                            newline += f"typedef {val} {layer_names[i-1]}_{key}_t;\n"
                        else:
                            newline += f"typedef {val} {key}_t;\n"
                        
               
            elif "// WRITE DENSE CONFIG" in line:

                newline += line

                for i, layer in enumerate(layer_names):
                    
                    newline += 'struct dense_config_' + str(i) + ' : nnet::dense_config {\n'
                    newline += indent + 'static const unsigned n_in = ' + str(layer_sizes[i]) + ';\n'
                    newline += indent + 'static const unsigned n_out = ' + str(layer_sizes[i+1]) + ';\n'
                    newline += indent + 'typedef ' + layer + '_width_t weight_t;\n'
                    newline += indent + 'typedef ' + layer + '_bias_t bias_t;\n'
                    newline += indent + 'typedef ' + layer + '_width_grad_t weight_grad_t;\n'
                    newline += indent + 'typedef ' + layer + '_bias_grad_t bias_grad_t;\n'
                    newline += indent + 'typedef ' + layer + '_act_grad_t act_grad_t;\n'
                    newline += '};\n'

                    if i != len(layer_names) - 1:
                        
                        newline += '\n'

            elif "// WRITE RELU CONFIG" in line:

                newline += line 

                for i in range(len(layer_names)):

                    newline += 'struct relu_config_' + str(i) + ' : nnet::relu_config {\n'
                    newline += indent + 'static const unsigned n_neuron = ' + str(layer_sizes[i+1]) + ';\n'
                    newline += '};\n'

                    if i != len(layer_names) - 1:

                        newline += '\n'

            elif "// WRITE ERROR GRADIENT CONFIG" in line:

                newline += line

                newline += 'struct ms_grad_config_0 : nnet::ms_grad_config {\n'
                newline += indent + 'static const unsigned n_neuron = ' + str(layer_sizes[-1]) + ';\n'
                newline += '};\n'

            elif "// WRITE FUNCTION PROTOTYPE" in line:

                newline += line

            elif "// WRITE FUNCTION IO" in line:

                newline = ''

                newline += indent + line

                # write input variable

                newline += indent + self.pre_relu_names[0] + '_t input[' + str(layer_sizes[0]) + '],\n\n'

                # write weight and bias input variables

                for i in range(len(layer_names)):

                    newline += indent + layer_names[i] + '_width_t ' + layer_names[i] + '_w[' +  str(layer_sizes[i+1]) + '][' + str(layer_sizes[i]) + '],\n'
                    newline += indent + layer_names[i] + '_bias_t ' + layer_names[i] + '_b[' +  str(layer_sizes[i+1]) + '],\n\n'

                # write inference output variables

                newline += indent + self.post_relu_names[-1] + '_t ' + layer_names[-1] + '_post_relu[' + str(layer_sizes[-1]) + '],\n\n'

                # write gradient output variables

                for i in range(len(layer_names)):

                    newline += indent + layer_names[i] + '_width_grad_t ' + layer_names[i] + '_w_grad[' +  str(layer_sizes[i+1]) + '][' + str(layer_sizes[i]) + '],\n'
                    newline += indent + layer_names[i] + '_bias_grad_t ' + layer_names[i] + '_b_grad[' +  str(layer_sizes[i+1]) + '],\n\n'

                # write ground truth input variable

                newline += indent + self.post_relu_names[0] + '_t truth[' + str(layer_sizes[0]) + ']\n'

            else:

                newline += line

            f_out.write(newline)

        f_in.close()
        f_out.close()





            

