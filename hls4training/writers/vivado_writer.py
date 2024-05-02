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

    def set_names_and_sizes(self, layer_names, layer_sizes):

        self.layer_names = layer_names
        self.layer_sizes = layer_sizes

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

                newline += indent + 'data_T input[' + str(layer_sizes[0]) + '],\n\n'

                # write weight and bias input variables

                for i in range(len(layer_names)):

                    newline += indent + 'data_T ' + layer_names[i] + '_w[' +  str(layer_sizes[i+1]) + '][' + str(layer_sizes[i]) + '],\n'
                    newline += indent + 'data_T ' + layer_names[i] + '_b[' +  str(layer_sizes[i+1]) + '],\n\n'

                # write inference output variables

                newline += indent + 'data_T ' + layer_names[-1] + '_post_relu[' + str(layer_sizes[-1]) + '],\n\n'

                # write gradient output variables

                for i in range(len(layer_names)):

                    newline += indent + 'data_T ' + layer_names[i] + '_w_grad[' +  str(layer_sizes[i+1]) + '][' + str(layer_sizes[i]) + '],\n'
                    newline += indent + 'data_T ' + layer_names[i] + '_b_grad[' +  str(layer_sizes[i+1]) + '],\n\n'

                # write ground truth input variable

                newline += indent + 'data_T truth[' + str(layer_sizes[-1]) + ']\n'

            elif '// WRITE INTERNAL VARIABLES' in line:

                newline = ''

                newline += indent + line

                # write inference internal variables

                newline += indent + 'data_T input_act_grad[' + str(layer_sizes[0]) + '];\n\n'

                for i in range(len(layer_names)-1):

                    newline += indent + 'data_T ' + layer_names[i] + '_pre_relu[' + str(layer_sizes[i+1]) + '];\n'
                    newline += indent + 'data_T ' + layer_names[i] + '_post_relu[' + str(layer_sizes[i+1]) + '];\n'
                    newline += indent + 'data_T ' + layer_names[i] + '_relu_grad[' + str(layer_sizes[i+1]) + '];\n'
                    newline += indent + 'data_T ' + layer_names[i] + '_act_grad[' + str(layer_sizes[i+1]) + '];\n\n'
                    

                newline += indent + 'data_T ' + layer_names[-1] + '_pre_relu[' + str(layer_sizes[-1]) + '];\n'
                newline += indent + 'data_T ' + layer_names[-1] + '_relu_grad[' + str(layer_sizes[-1]) + '];\n' 
                newline += indent + 'data_T ' + layer_names[-1] + '_act_grad[' + str(layer_sizes[-1]) + '];\n'

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
                newline += 'nnet::dense_infer <data_T, dense_config_0> (input, '
                newline += layer_names[0] + '_pre_relu, ' + layer_names[0] + '_w, ' + layer_names[0] + '_b);\n'

                newline += indent
                newline += 'nnet::relu <data_T, relu_config_0> ('
                newline += layer_names[0] + '_pre_relu, ' + layer_names[0] + '_post_relu);'

                if len(layer_names) > 1:

                    newline += '\n\n'

                for i in range(len(layer_names)-1):

                    newline += indent
                    newline += 'nnet::dense_infer <data_T, dense_config_' + str(i+1) + '> (' + layer_names[i] + '_post_relu, '
                    newline += layer_names[i+1] + '_pre_relu, ' + layer_names[i+1] + '_w, ' + layer_names[i+1] + '_b);\n'

                    newline += indent
                    newline += 'nnet::relu <data_T, relu_config_' + str(i+1) + '> ('
                    newline += layer_names[i+1] + '_pre_relu, ' + layer_names[i+1] + '_post_relu);'

                    if i != len(layer_names)-2:
                        newline += '\n\n'                

            elif '// WRITE ERROR GRADIENT LAYER' in line:

                newline += indent
                newline += line
                newline += indent
                newline += 'nnet::ms_grad <data_T, ms_grad_config_0> ('
                newline += layer_names[-1] + '_post_relu, truth, ' + layer_names[-1] + '_act_grad);'

            elif '// WRITE BACKPROPAGATION LAYERS' in line:

                newline += indent
                newline += line

                for i in range(len(layer_names)-1):

                    j = len(layer_names) - i - 1

                    newline += indent

                    newline += 'nnet::relu_grad <data_T, relu_config_' + str(j) + '> ('
                    newline += layer_names[j] + '_pre_relu, ' + layer_names[j] + '_relu_grad);\n'

                    newline += indent
                    newline += 'nnet::dense_backprop <data_T, dense_config_' + str(j) + '> ('
                    newline += layer_names[j] + '_relu_grad, ' + layer_names[j-1] + '_post_relu, '
                    newline += layer_names[j] + '_act_grad, ' + layer_names[j] + '_w, '
                    newline += layer_names[j] + '_w_grad, ' + layer_names[j] + '_b_grad, '
                    newline += layer_names[j-1] + '_act_grad);\n\n'

                newline += indent
                newline += 'nnet::relu_grad <data_T, relu_config_0> ('
                newline += layer_names[0] + '_pre_relu, ' + layer_names[0] + '_relu_grad);\n'

                newline += indent
                newline += 'nnet::dense_backprop <data_T, dense_config_0> ('
                newline += layer_names[0] + '_relu_grad, input, '
                newline += layer_names[0] + '_act_grad, ' + layer_names[0] + '_w, '
                newline += layer_names[0] + '_w_grad, ' + layer_names[0] + '_b_grad, '
                newline += 'input_act_grad);\n'

                # nnet::dense_backprop <data_T, dense_config_0>
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

                for h in headers:

                    newline += '#include \"' + self.util_dir + h + '\"\n'

            elif "// WRITE DATA TYPE DEFINITIONS" in line:

                newline += line
                newline += "typedef int data_T;\n"

            elif "// WRITE DENSE CONFIG" in line:

                newline += line

                for i in range(len(layer_names)):

                    newline += 'struct dense_config_' + str(i) + ' : nnet::dense_config {\n'
                    newline += indent + 'static const unsigned n_in = ' + str(layer_sizes[i]) + ';\n'
                    newline += indent + 'static const unsigned n_out = ' + str(layer_sizes[i+1]) + ';\n'
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

                newline += indent + 'data_T input[' + str(layer_sizes[0]) + '],\n\n'

                # write weight and bias input variables

                for i in range(len(layer_names)):

                    newline += indent + 'data_T ' + layer_names[i] + '_w[' +  str(layer_sizes[i+1]) + '][' + str(layer_sizes[i]) + '],\n'
                    newline += indent + 'data_T ' + layer_names[i] + '_b[' +  str(layer_sizes[i+1]) + '],\n\n'

                # write inference output variables

                newline += indent + 'data_T ' + layer_names[-1] + '_post_relu[' + str(layer_sizes[-1]) + '],\n\n'

                # write gradient output variables

                for i in range(len(layer_names)):

                    newline += indent + 'data_T ' + layer_names[i] + '_w_grad[' +  str(layer_sizes[i+1]) + '][' + str(layer_sizes[i]) + '],\n'
                    newline += indent + 'data_T ' + layer_names[i] + '_b_grad[' +  str(layer_sizes[i+1]) + '],\n\n'

                # write ground truth input variable

                newline += indent + 'data_T truth[' + str(layer_sizes[-1]) + ']\n'

            else:

                newline += line

            f_out.write(newline)

        f_in.close()
        f_out.close()





            

