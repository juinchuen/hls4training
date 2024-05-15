import os
import sys
import numpy as np

sys.path.append('../../../python_backprop')

from mlp import mlp

class vivado_writer:

    def __init__(self, project_dir = '', template_dir = '', util_dir='', mlp_dir='', csim_out_dir=''):

        self.project_dir = project_dir
        self.template_dir = template_dir
        self.util_dir = util_dir
        self.mlp_dir = mlp_dir
        self.csim_out_dir = csim_out_dir

        self.div_regularizer = 0.001

        # sys.path.append(self.mlp_dir)

        # from mlp import mlp

        # self.net_shape = net_shape

    def write_project_dir(self):

        if not os.path.isdir(self.project_dir):
            os.makedirs(self.project_dir)

    def set_names_and_sizes(self, layer_names, layer_sizes, data_type=['signed', 16, 8, 'AP_SAT']):
        if(len(layer_names) + 1 != len(layer_sizes)):
            print('Error: Number of layer names doesn\'t match number of layers that are set to be quantized')
        self.layer_names = layer_names
        self.layer_sizes = layer_sizes
        self.global_data_type = data_type[0:3]
        self.D_BITS = data_type[1]
        self.Q_BITS = data_type[2]
        self.RND_MODE = data_type[3]
        # self.layer_quantization = layer_quantization

        self.set_global_datatype(data_type=data_type)

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
                                        'post_relu':  self._create_ap_variables(layer[0]),
                                        'weight' :     self._create_ap_variables(layer[0]),
                                        'bias'  :     self._create_ap_variables(layer[0]),
                                        'relu_grad' : self._create_ap_variables(layer[0]),
                                        'act_grad' :  self._create_ap_variables(layer[0]),
                                        'weight_grad': self._create_ap_variables(layer[0]),
                                        'bias_grad' : self._create_ap_variables(layer[0])
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
                print('Warning: Sign convention for input is ', ap_var[curr_ind],' which isn\'t recongnized. Defaulting to signed')
            curr_ind += 1
        else:
            signed = True

        num_bit = ap_var[curr_ind]
        num_int = ap_var[curr_ind + 1]

        ap_type = f'ap_fixed<{num_bit}, {num_int}, AP_RND_CONV, {self.RND_MODE}>' if signed else f'ap_ufixed<{num_bit}, {num_int}, AP_RND_CONV, {self.RND_MODE}>'
        return ap_type

    def write_project_cpp(self, layer_names, layer_sizes):

        if (len(self.layer_names) != len(self.layer_sizes) - 1):

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

                for i in range(len(self.layer_names)):

                    newline += indent + layer_names[i] + '_weight_t ' + layer_names[i] + '_w[' +  str(layer_sizes[i+1]) + '][' + str(layer_sizes[i]) + '],\n'
                    newline += indent + layer_names[i] + '_bias_t '+ layer_names[i] + '_b[' +  str(layer_sizes[i+1]) + '],\n\n'

                # write inference output variables

                newline += indent + self.post_relu_names[-1] + '_t ' + layer_names[-1] + '_post_relu[' + str(layer_sizes[-1]) + '],\n\n'

                # write gradient output variables

                for i in range(len(layer_names)):
                    newline += indent + layer_names[i] + '_weight_grad_t ' + layer_names[i] + '_w_grad[' +  str(layer_sizes[i+1]) + '][' + str(layer_sizes[i]) + '],\n'
                    newline += indent + layer_names[i] + '_bias_grad_t ' + layer_names[i] + '_b_grad[' +  str(layer_sizes[i+1]) + '],\n\n'

                # write ground truth input variable

                newline += indent + self.post_relu_names[0] + '_t truth[' + str(layer_sizes[-1]) + ']\n'

            elif '// WRITE INTERNAL VARIABLES' in line:

                newline = ''

                newline += indent + line

                # write inference internal variables

                newline += indent + 'input_act_grad_t input_act_grad[' + str(layer_sizes[0]) + '];\n\n'

                for i in range(len(self.layer_names)-1):

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

                for g in self.all_names:

                    for v in g:

                        newline += indent + generate_pragma(v)
                    

                #pragma HLS ARRAY_PARTITION variable=layer0_w complete dim=0
                
            elif '// WRITE INFERENCE LAYERS ' in line:

                newline = ''

                newline += indent + line

                newline += indent
                newline += 'nnet::dense_infer <dense_config_0> (input, '
                newline += layer_names[0] + '_pre_relu, ' + layer_names[0] + '_w, ' + layer_names[0] + '_b);\n'

                newline += indent
                newline += f'nnet::relu <relu_config_0> ('
                newline += layer_names[0] + '_pre_relu, ' + layer_names[0] + '_post_relu);'

                if len(self.layer_names) > 1:

                    newline += '\n\n'

                for i in range(len(self.layer_names)-1):

                    newline += indent
                    newline += f'nnet::dense_infer <dense_config_' + str(i+1) + '> (' + layer_names[i] + '_post_relu, '
                    newline += layer_names[i+1] + '_pre_relu, ' + layer_names[i+1] + '_w, ' + layer_names[i+1] + '_b);\n'

                    newline += indent
                    newline += f'nnet::relu <relu_config_' + str(i+1) + '> ('
                    newline += layer_names[i+1] + '_pre_relu, ' + layer_names[i+1] + '_post_relu);'

                    if i != len(self.layer_names)-2:
                        newline += '\n\n'                

            elif '// WRITE ERROR GRADIENT LAYER' in line:

                newline += indent
                newline += line
                newline += indent
                newline += f'nnet::ms_grad <ms_grad_config_0> ('
                newline += layer_names[-1] + '_post_relu, truth, ' + layer_names[-1] + '_act_grad);'

            elif '// WRITE BACKPROPAGATION LAYERS' in line:

                newline += indent
                newline += line

                for i in range(len(self.layer_names)-1):

                    j = len(self.layer_names) - i - 1

                    newline += indent

                    newline += f'nnet::relu_grad <relu_config_' + str(j) + '> ('
                    newline += layer_names[j] + '_pre_relu, ' + layer_names[j] + '_relu_grad);\n'

                    newline += indent
                    newline += f'nnet::dense_backprop <dense_config_' + str(j) + '> ('
                    newline += layer_names[j] + '_relu_grad, ' + layer_names[j-1] + '_post_relu, '
                    newline += layer_names[j] + '_act_grad, ' + layer_names[j] + '_w, '
                    newline += layer_names[j] + '_w_grad, ' + layer_names[j] + '_b_grad, '
                    newline += layer_names[j-1] + '_act_grad);\n\n'

                newline += indent
                newline += f'nnet::relu_grad <relu_config_0> ('
                newline += layer_names[0] + '_pre_relu, ' + layer_names[0] + '_relu_grad);\n'

                newline += indent
                newline += f'nnet::dense_backprop<dense_config_0> ('
                newline += layer_names[0] + '_relu_grad, input, '
                newline += layer_names[0] + '_act_grad, ' + layer_names[0] + '_w, '
                newline += layer_names[0] + '_w_grad, ' + layer_names[0] + '_b_grad, '
                newline += 'input_act_grad);\n'

            else:

                newline = line

            f_out.write(newline)

        f_in.close()
        f_out.close()

    def write_header (self, headers):

        if (len(self.layer_names) != len(self.layer_sizes) - 1):

            # check whether layer_names and layer_sizes argument sizes are correct

            raise ValueError('Incorrect length of arguments. layer_names refers to computation layers. layer_sizes refers to variables in between computation layers')

        f_in = open(os.path.join(self.template_dir, 'project.h'), 'r')

        f_out = open(os.path.join(self.project_dir, 'myproject.h'), 'w')

        indent = '    '

        for line in f_in.readlines():

            if 'project' in line:

                line = line.replace('project', 'myproject')

            newline = ''

            if '// WRITE INCLUDE DIRECTIVES' in line:

                newline += line
                newline += '#include <ap_fixed.h>\n'
                for h in headers:
                    newline += '#include \"' + self.util_dir + '/' + h + '\"\n'

            elif '// WRITE DATA TYPE DEFINITIONS' in line:

                newline += line
                for i, data_dict in enumerate(self.data_types):
                    for key, val in data_dict.items():
                        if i > 0:
                            newline += f'typedef {val} {self.layer_names[i-1]}_{key}_t;\n'
                        else:
                            newline += f'typedef {val} {key}_t;\n'
                        
               
            elif '// WRITE DENSE CONFIG' in line:

                newline += line

                for i, layer in enumerate(self.layer_names):
                    
                    newline += 'struct dense_config_' + str(i) + ' : nnet::dense_config {\n'
                    newline += indent + 'static const unsigned n_in = ' + str(self.layer_sizes[i]) + ';\n'
                    newline += indent + 'static const unsigned n_out = ' + str(self.layer_sizes[i+1]) + ';\n'
                    if i == 0:
                        newline += indent + 'typedef input_t data_t;\n'
                    else:
                        newline += indent + 'typedef ' + self.layer_names[i-1] + '_post_relu_t data_t;\n'
                    newline += indent + 'typedef ' + layer + '_post_relu_t res_t;\n'
                    newline += indent + 'typedef ' + layer + '_weight_t weight_t;\n'
                    newline += indent + 'typedef ' + layer + '_bias_t bias_t;\n'
                    newline += indent + 'typedef ' + layer + '_weight_grad_t weight_grad_t;\n'
                    newline += indent + 'typedef ' + layer + '_bias_grad_t bias_grad_t;\n'
                    newline += indent + 'typedef ' + layer + '_act_grad_t act_grad_in_t;\n'
                    newline += indent + 'typedef ' + layer + '_relu_grad_t relu_grad_t;\n'
                    if i == 0:
                        newline += indent + 'typedef input_act_grad_t act_grad_out_t;\n'
                    else:
                        newline += indent + 'typedef ' + self.layer_names[i-1] + '_act_grad_t act_grad_out_t;\n'
                    newline += '};\n'

                    if i != len(self.layer_names) - 1:
                        
                        newline += '\n'

            elif '// WRITE RELU CONFIG' in line:

                newline += line 

                for i in range(len(self.layer_names)):

                    newline += 'struct relu_config_' + str(i) + ' : nnet::relu_config {\n'
                    newline += indent + 'static const unsigned n_neuron = ' + str(self.layer_sizes[i+1]) + ';\n'
                    
                    newline += indent + 'typedef ' + self.layer_names[i] + '_pre_relu_t data_t;\n'
                    newline += indent + 'typedef ' + self.layer_names[i] + '_post_relu_t relu_t;\n'
                    newline += indent + 'typedef ' + self.layer_names[i] + '_relu_grad_t grad_t;\n'

                    newline += '};\n'

                    if i != len(self.layer_names) - 1:

                        newline += '\n'

            elif '// WRITE ERROR GRADIENT CONFIG' in line:

                newline += line

                newline += 'struct ms_grad_config_0 : nnet::ms_grad_config {\n'
                newline += indent + 'static const unsigned n_neuron = ' + str(self.layer_sizes[-1]) + ';\n'
                newline += indent + 'typedef ' + self.layer_names[-1] + '_post_relu_t pred_t;\n'
                newline += indent + 'typedef truth_t real_t;\n'
                newline += indent + 'typedef ' + self.layer_names[-1] + '_act_grad_t act_grad_t;\n'
                newline += '};\n'

            elif '// WRITE FUNCTION PROTOTYPE' in line:

                newline += line

            elif '// WRITE FUNCTION IO' in line:

                newline = ''

                newline += indent + line

                # write input variable

                newline += indent + self.pre_relu_names[0] + '_t input[' + str(self.layer_sizes[0]) + '],\n\n'

                # write weight and bias input variables

                for i in range(len(self.layer_names)):

                    newline += indent + self.layer_names[i] + '_weight_t ' + self.layer_names[i] + '_w[' +  str(self.layer_sizes[i+1]) + '][' + str(self.layer_sizes[i]) + '],\n'
                    newline += indent + self.layer_names[i] + '_bias_t ' + self.layer_names[i] + '_b[' +  str(self.layer_sizes[i+1]) + '],\n\n'

                # write inference output variables

                newline += indent + self.post_relu_names[-1] + '_t ' + self.layer_names[-1] + '_post_relu[' + str(self.layer_sizes[-1]) + '],\n\n'

                # write gradient output variables

                for i in range(len(self.layer_names)):

                    newline += indent + self.layer_names[i] + '_weight_grad_t ' + self.layer_names[i] + '_w_grad[' +  str(self.layer_sizes[i+1]) + '][' + str(self.layer_sizes[i]) + '],\n'
                    newline += indent + self.layer_names[i] + '_bias_grad_t ' + self.layer_names[i] + '_b_grad[' +  str(self.layer_sizes[i+1]) + '],\n\n'

                # write ground truth input variable

                newline += indent + 'truth_t truth[' + str(self.layer_sizes[-1]) + ']\n'

            else:

                newline += line

            f_out.write(newline)

        f_in.close()
        f_out.close()

    def write_tcl (self, headers):

        if (len(self.layer_names) != len(self.layer_sizes) - 1):

            # check whether layer_names and layer_sizes argument sizes are correct

            raise ValueError('Incorrect length of arguments. layer_names refers to computation layers. layer_sizes refers to variables in between computation layers')

        f_in = open(os.path.join(self.template_dir, 'project.tcl'), 'r')

        f_out = open(os.path.join(self.project_dir, 'myproject.tcl'), 'w')

        for line in f_in.readlines():

            # if 'project' in line:

            #     line = line.replace('project', 'myproject')

            newline = ''

            if '# ADD UTIL FILES' in line:

                newline += line

                for h in headers:

                    newline += 'add_files ' + self.util_dir + h + '\n'

            else:

                newline += line

            f_out.write(newline)

        f_in.close()
        f_out.close()

    def set_global_datatype (self, data_type = ['signed', 16, 8]):

        self.layer_quantization = []

        try :

            self.layer_quantization = [ [data_type] * 3 ] * (len(self.layer_sizes))

        except:

            raise NameError('Run set_name_and_size() before running set_global_datatype() to initialize layer_names and layer_sizes internally')

    def write_testbench (self):

        f_in = open(os.path.join(self.template_dir, 'testbench.cpp'), 'r')

        f_out = open(os.path.join(self.project_dir, 'testbench.cpp'), 'w')

        ref = mlp()

        indent = '    '

        # defining reference model (sourced from custom backprop library)
        # ref.set_quant(self.D_BITS, self.Q_BITS, True)
        ref.set_quant(quant=False)
        ref.net_shape(self.layer_sizes)
        ref.init_params(mode='uniform_neg')

        x = np.random.random((self.layer_sizes[0], 1))

        ref.infer(x)

        truth = ref._activation[-1] + np.random.random((self.layer_sizes[-1], 1))

        ref.grad(truth)

        truth = truth.flatten()

        for line in f_in.readlines():

            if 'project' in line:

                line = line.replace('project', 'myproject')

            newline = ''

            if '// INITIALIZE INPUTS' in line:

                newline += line

                # actual input to the model
                newline += indent + 'input_t input [' + str(int(self.layer_sizes[0])) + '] = {'

                x = x.flatten()

                for i in range(len(x)):

                    if (i == len(x) - 1):

                        newline += str(x[i]) + '};\n\n'

                    else:

                        newline += str(x[i]) + ','

                # weights
                        
                N = len(self.layer_names)

                # for each layer
                for i in range(N):

                    newline += indent + self.layer_names[i] + '_weight_t ' + self.layer_names[i] + '_weight '
                    newline += '[' + str(int(self.layer_sizes[i+1])) + '][' + str(int(self.layer_sizes[i])) + '] = {\n'

                    # for each row in the weight matrix
                    for k in range(len(ref._weight[i]) - 1):

                        r = ref._weight[i][k]
                        newline += indent * 5
                        newline += '{'

                        # for each column in each row
                        for j in range(len(r) - 1):
                            newline += str(r[j]) + ','
                        newline += str(r[-1]) + '},\n'

                    # write last row (different to close out initializer)
                    r = ref._weight[i][-1]
                    newline += indent * 5
                    newline += '{'

                    for j in range(len(r) - 1):
                        newline += str(r[j]) + ','
                    newline += str(r[-1]) + '}};\n\n'

                # bias
                    
                for i in range(N):

                    newline += indent + self.layer_names[i] + '_bias_t ' + self.layer_names[i] + '_bias '
                    newline += '[' + str(int(self.layer_sizes[i+1])) + '] = {'

                    for j in range(len(ref._bias[i]) - 1):

                        newline += str(ref._bias[i][j][0]) + ','

                    newline += str(ref._bias[i][-1][0]) + '};\n\n'

                newline = newline[:-1]

            elif '// INITIALIZE TRUTHS' in line:

                newline += line

                # ground truth
                newline += indent + 'truth_t truth [' + str(int(self.layer_sizes[-1])) + '] = {'

                truth = truth.flatten()

                for i in range(len(truth)):

                    if (i == len(truth) - 1):

                        newline += str(truth[i]) + '};\n\n'

                    else:

                        newline += str(truth[i]) + ','

                # weight gradient
                        
                N = len(self.layer_names)

                # for each layer
                for i in range(N):

                    newline += indent + self.layer_names[i] + '_weight_grad_t ' + self.layer_names[i] + '_weight_grad_real '
                    newline += '[' + str(int(self.layer_sizes[i+1])) + '][' + str(int(self.layer_sizes[i])) + '] = {\n'

                    # for each row in the weight matrix
                    for k in range(len(ref._weight_dif[i]) - 1):

                        r = ref._weight_dif[i][k]
                        newline += indent * 5
                        newline += '{'

                        # for each column in each row
                        for j in range(len(r) - 1):
                            newline += str(r[j]) + ','
                        newline += str(r[-1]) + '},\n'

                    # write last row (different to close out initializer)
                    r = ref._weight_dif[i][-1]
                    newline += indent * 5
                    newline += '{'

                    for j in range(len(r) - 1):
                        newline += str(r[j]) + ','
                    newline += str(r[-1]) + '}};\n\n'

                # bias gradient
                    
                for i in range(N):

                    newline += indent + self.layer_names[i] + '_bias_grad_t ' + self.layer_names[i] + '_bias_grad_real '
                    newline += '[' + str(int(self.layer_sizes[i+1])) + '] = {'

                    for j in range(len(ref._bias_dif[i]) - 1):

                        newline += str(ref._bias_dif[i][j][0]) + ','

                    newline += str(ref._bias_dif[i][-1][0]) + '};\n'

            elif '// DECLARE RESULTS' in line:

                newline += line

                # ground truth
                newline += indent + self.layer_names[-1] + '_post_relu_t y_pred [' + str(int(self.layer_sizes[-1])) + '];\n\n'

                # weight gradient
                        
                N = len(self.layer_names)

                # for each layer
                for i in range(N):

                    newline += indent + self.layer_names[i] + '_weight_grad_t ' + self.layer_names[i] + '_weight_grad_pred '
                    newline += '[' + str(int(self.layer_sizes[i+1])) + '][' + str(int(self.layer_sizes[i])) + '];\n'

                newline += '\n'

                # bias gradient
                    
                for i in range(N):

                    newline += indent + self.layer_names[i] + '_bias_grad_t ' + self.layer_names[i] + '_bias_grad_pred '
                    newline += '[' + str(int(self.layer_sizes[i+1])) + '];\n'

            elif '// CALL PROJECT FUNCTION' in line:

                newline += line

                newline += indent + 'myproject(\n'

                newline += indent * 2 + 'input,\n'

                N = len(self.layer_names)

                for i in range(N):

                    newline += indent * 2 + self.layer_names[i] + '_weight,\n'
                    newline += indent * 2 + self.layer_names[i] + '_bias,\n'
                    
                newline += indent * 2 + 'y_pred,\n'

                for i in range(N):

                    newline += indent * 2 + self.layer_names[i] + '_weight_grad_pred,\n'
                    newline += indent * 2 + self.layer_names[i] + '_bias_grad_pred,\n'

                newline += indent * 2 + 'truth\n    );\n'

            elif '// CALCULATE PERCENT ERROR' in line:

                newline += line

                N = len(self.layer_names)

                newline += indent + 'if (DEBUG) fout = fopen("' + self.csim_out_dir + '/grad_results.txt", "w");\n\n'

                for i in range(N):

                    newline += indent + '// weight error for ' + self.layer_names[i] + '\n'

                    newline += indent + 'if (DEBUG) fprintf(fout, "' + self.layer_names[i] + ' weight grad\\n");\n\n'

                    newline += indent + 'for (int i = 0; i < ' +  str(int(self.layer_sizes[i + 1])) + '; i++){\n'
                    newline += indent * 2 + 'for (int j = 0; j < ' +  str(int(self.layer_sizes[i])) + '; j++){\n\n'

                    newline += indent * 3 + 'val_real = (float) ' + self.layer_names[i] + '_weight_grad_real[i][j];\n'
                    newline += indent * 3 + 'val_pred = (float) ' + self.layer_names[i] + '_weight_grad_pred[i][j];\n\n'

                    newline += indent * 3 + 'if (DEBUG) fprintf(fout, "%f, ", val_pred);\n\n'
                    
                    newline += indent * 3 + 'error += abs((val_real - val_pred)/(val_real + ' + str(self.div_regularizer) + '));\n'
                    newline += indent * 3 + 'count += 1;\n'

                    newline += indent * 2 + '}\n\n'

                    newline += indent * 2 + 'if (DEBUG) fprintf(fout, "\\n");\n\n'
                    
                    newline += indent + '}\n\n'

                    newline += indent + '// bias error for ' + self.layer_names[i] + '\n'

                    newline += indent + 'if (DEBUG) fprintf(fout, "' + self.layer_names[i] + ' bias grad\\n");\n\n'

                    newline += indent + 'for (int i = 0; i < ' +  str(int(self.layer_sizes[i + 1])) + '; i++){\n\n'

                    newline += indent * 2 + 'val_real = (float) ' + self.layer_names[i] + '_bias_grad_real[i];\n'
                    newline += indent * 2 + 'val_pred = (float) ' + self.layer_names[i] + '_bias_grad_pred[i];\n'

                    newline += indent * 2 + 'if (DEBUG) fprintf(fout, "%f, ", val_pred);\n\n'
                    
                    newline += indent * 2 + 'error += abs((val_real - val_pred)/(val_real + ' + str(self.div_regularizer) + '));\n'
                    newline += indent * 2 + 'count += 1;\n'

                    newline += indent + '}\n\n'

                    newline += indent + 'if (DEBUG) fprintf(fout, "\\n");\n\n'

                newline += indent + 'error = error / count;\n'

                newline += indent + 'printf("Test Complete: Average Percent Error across %d gradients is %2.3f%\\n", count, error * 100);\n'

                newline += indent + 'if (DEBUG) fclose(fout);\n'

            else:

                newline += line

            f_out.write(newline)

        f_in.close()
        f_out.close()







            

