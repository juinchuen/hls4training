import os

class vivado_writer:

    def __init__(self, project_dir = '', template_dir = '', net_shape = [3,5]):

        self.project_dir = project_dir
        self.template_dir = template_dir
        self.net_shape = net_shape

    def write_project_dir(self):

        if not os.path.isdir(self.project_dir):
            os.makedirs(self.project_dir)

    def generate_variable_names(self, layer_names, layer_sizes):

        self.pre_relu_names = []
        self.post_relu_names = []
        self.relu_grad_names = []
        self.weight_names = []
        self.weight_grad_names = []
        self.bias_names = []
        self.bias_grad_names = []
        self.act_grad_names = []

        for i in range(len(layer_names)):

            pass

    def write_project_cpp(self, layer_names, layer_sizes):

        if (len(layer_names) != len(layer_sizes) - 1):

            # check whether layer_names and layer_sizes argument sizes are correct

            raise ValueError('Incorrect length of arguments. layer_names refers to computation layers. layer_sizes refers to variables in between computation layers')

        f_in = open(os.path.join(self.template_dir, 'project.cpp'), 'r')

        f_out = open(os.path.join(self.project_dir, 'myproject.cpp'), 'w')

        indent = '    '

        for line in f_in.readlines():

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
                
            elif '// WRITE INFERENCE LAYERS ' in line:

                newline = ''

                newline += indent + line

                newline += indent
                newline += 'nnet:dense_infer <data_T, dense_config_0> (input, '
                newline += layer_names[0] + '_pre_relu, ' + layer_names[0] + '_w, ' + layer_names[0] + '_b);\n'

                newline += indent
                newline += 'nnet:relu <data_T, relu_config_0> ('
                newline += layer_names[0] + '_pre_relu, ' + layer_names[0] + '_post_relu);'

                if len(layer_names) > 1:

                    newline += '\n\n'

                for i in range(len(layer_names)-1):

                    newline += indent
                    newline += 'nnet:dense_infer <data_T, dense_config_' + str(i+1) + '> (' + layer_names[i] + '_post_relu, '
                    newline += layer_names[i+1] + '_pre_relu, ' + layer_names[i+1] + '_w, ' + layer_names[0] + '_b);\n'

                    newline += indent
                    newline += 'nnet:relu <data_T, relu_config_' + str(i+1) + '> ('
                    newline += layer_names[i+1] + '_pre_relu, ' + layer_names[i+1] + '_post_relu);'

                    if i != len(layer_names)-2:
                        newline += '\n\n'                

            elif '// WRITE ERROR GRADIENT LAYER' in line:

                newline += indent
                newline += line
                newline += indent
                newline += 'nnet::ms_grad <data_T, ms_grad_config> ('
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

            

