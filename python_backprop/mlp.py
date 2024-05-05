import numpy as np
import matplotlib.pyplot as plt

# we use a 1 hidden layer, 16 neuron nn
# to map 3 x,y,z values to 5 angle values

# INPUT_N = 3
# HIDDEN_N = 16
# OUTPUT_N = 5

class mlp:

    def __init__(self):
        self._net_shape = []
        self._num_weight = 0
        self._weight = []
        self._bias = []
        self._weight_dif = []
        self._bias_dif = []
        self._weight_dif_min=[]
        self._weight_dif_max=[]
        self._bias_dif_min=[]
        self._bias_dif_max=[]
        self._activation = []
        self._pre_relu = []
        self._param_history = []
        self._cost_history = []
        self._d_bits = 0
        self._q_bits = 0
        self._quant = False

    def check_shape(self): #NAbQ

        for i in range(len(self._weight)):

            try:
                if (self._weight[i].shape != (self._net_shape[i+1], self._net_shape[i])):
                    print("Wrong Shape: weight[" + str(i) + "]")
            except:
                print("Wrong Shape: weight[" + str(i) + "]")
        
        for i in range(len(self._bias)):

            try:
                if (self._bias[i].shape != (self._net_shape[i+1], 1)):
                    print("Wrong Shape: bias[" + str(i) + "]")
            except:
                print("Wrong Shape: bias[" + str(i) + "]")

    def relu(self, x): #NAbQ
        return np.maximum(x,0)
        
    def relu_dif(self, x): #NAbQ

        if (self._quant):
            return np.multiply((np.sign(x) + 1) / 2, 2**self._q_bits)
        else:
            return (np.sign(x) + 1) / 2

    def init_params(self, mode = "uniform"): #quanted

        if (mode == "uniform_neg"):

            self._weight = []
            self._bias = []

            if (self._quant):
                for i in range(len(self._net_shape) - 1):
                    self._weight.append(np.round(np.multiply(2 * np.random.rand(self._net_shape[i+1], self._net_shape[i]) - 1, 2**self._q_bits)))
                    self._bias.append(np.round(np.multiply(2 * np.random.rand(self._net_shape[i+1], 1) - 1, 2**self._q_bits)))
            else:
                for i in range(len(self._net_shape) - 1):
                    self._weight.append(2 * np.random.rand(self._net_shape[i+1], self._net_shape[i]) - 1)
                    self._bias.append(2 * np.random.rand(self._net_shape[i+1], 1) - 1)

        elif (mode == "uniform"):

            self._weight = []
            self._bias = []

            if (self._quant):
                for i in range(len(self._net_shape) - 1):
                    self._weight.append(np.round(np.multiply(np.random.rand(self._net_shape[i+1], self._net_shape[i]), 2**self._q_bits)))
                    self._bias.append(np.round(np.multiply(np.random.rand(self._net_shape[i+1], 1), 2**self._q_bits)))
            else:
                for i in range(len(self._net_shape) - 1):
                    self._weight.append(np.random.rand(self._net_shape[i+1], self._net_shape[i]))
                    self._bias.append(np.random.rand(self._net_shape[i+1], 1))

        elif (mode == "ones"):

            self._weight = []
            self._bias = []

            if (self._quant):
                for i in range(len(self._net_shape) - 1):
                    self._weight.append(np.multiply(np.ones((self._net_shape[i+1], self._net_shape[i])), 2**self._q_bits))
                    self._bias.append(np.multiply(np.ones((self._net_shape[i+1], 1), 2**self._q_bits), 2**self._q_bits))
            else:
                for i in range(len(self._net_shape) - 1):
                    self._weight.append(np.ones((self._net_shape[i+1], self._net_shape[i])))
                    self._bias.append(np.ones((self._net_shape[i+1], 1)))

        self.check_shape()

        return 0

    def grad_buffer(self): #NAbQ

        weight_buf = []
        bias_buf = []

        for i in range(len(self._net_shape) - 1):
            weight_buf.append(np.zeros((self._net_shape[i+1], self._net_shape[i])))
            bias_buf.append(np.zeros((self._net_shape[i+1], 1)))

        return weight_buf, bias_buf

    def infer(self, x): #quanted
        
        try:
            if (x.shape == (self._net_shape[0],1)):
                self._activation = []
                self._activation.append(x)
                self._pre_relu.append(x)
            else:
                raise ValueError("infer: Input is wrong size, should be (" + str(self._net_shape[0]) + ", " + "1)")

        except:
            raise ValueError("infer: Input is wrong size, should be (" + str(self._net_shape[0]) + ", " + "1)")

        if (self._quant): #bit shift to maintain quantization, np.minimum to simulate saturation
            for i in range(len(self._net_shape)-1):
                self._pre_relu.append(self.enforce_quant_range( np.divide(np.dot(self._weight[i], self._activation[i]), 2**self._q_bits) + self._bias[i]))
                self._activation.append(self.relu(self._pre_relu[-1]))
        else:
            for i in range(len(self._net_shape)-1):
                self._pre_relu.append(np.dot(self._weight[i], self._activation[i]) + self._bias[i])
                self._activation.append(self.relu(self._pre_relu[-1]))

    def net_shape(self, x): #NAbQ

        if (not type(x) == list):
            raise ValueError("net_shape: Input must be a list of ints")

        for n in x:
            if (not type(n) == int):
                raise ValueError("net_shape: Input must be a list of ints")

        self._net_shape = x

        self._num_weight = len(x) - 1

    def set_quant(self, d_bits = 16, q_bits = 10, quant = True): #quanted

        if (not isinstance(d_bits, int) or not isinstance(q_bits, int)):

            raise TypeError("d_bits and q_bits must be ints")

        self._d_bits = d_bits
        self._q_bits = q_bits
        self._quant  = quant

    def post_mult_quant(self, x):
        # right shift after multiplication to maintain quantization
        # then enforce quantization range

        return self.enforce_quant_range(np.divide(x, 2**self._q_bits))
    
    def enforce_quant_range(self, x):

        # apply round and minimum to enforce quantization range

        return np.maximum(np.minimum(np.round(x), (2**(self._d_bits-1))-1), -(2**(self._d_bits-1)))
    
    def float_to_quant(self, x):

        return np.maximum(np.minimum(np.round(np.multiply(x, 2**self._q_bits)) ,(2**self._d_bits)-1), -(2**(self._d_bits-1)))
    
    def quant_to_float(self, x):

        return np.divide(x, 2**self._q_bits)

    def print_act(self): #NAbQ
        
        for i in range(len(self._activation)):

            print("Activation of layer " + str(i) + ":")
            print(self._activation[i])
            print("\n")

    def grad(self, truth, debug=False):

        # assuming mean square error for now

        try:
            if (truth.shape != (self._net_shape[-1],1)):
                raise ValueError("grad: Input is wrong size, should be (" + str(self._net_shape[-1]) + ", " + "1)")

        except:
            raise ValueError("grad: Input is wrong size, should be (" + str(self._net_shape[-1]) + ", " + "1)")

        # partial derivative of error with respect to output of neural network

        self._activation_dif = []
        self._weight_dif = []
        self._bias_dif = []

        self._activation_dif.insert(0,2 * (self._activation[-1] - truth))

        if (self._quant):
            for i in range(len(self._net_shape) - 1):
                self._bias_dif.insert(0,self.post_mult_quant(np.multiply(self.relu_dif(self._pre_relu[-(i+1)]), self._activation_dif[0])))
                self._weight_dif.insert(0,self.post_mult_quant(np.dot(self._bias_dif[0], self._activation[-(i+2)].transpose())))
                self._activation_dif.insert(0,self.post_mult_quant(np.dot(self._weight[-(i+1)].transpose(), self._bias_dif[0])))
        else:
            for i in range(len(self._net_shape) - 1):
                self._bias_dif.insert(0,np.multiply(self.relu_dif(self._pre_relu[-(i+1)]), self._activation_dif[0]))
                self._weight_dif.insert(0,np.dot(self._bias_dif[0], self._activation[-(i+2)].transpose()))
                self._activation_dif.insert(0,np.dot(self._weight[-(i+1)].transpose(), self._bias_dif[0]))

    def print_grad(self): #NAbQ

        for i in range(len(self._weight_dif)):

            print("Weight Gradient Layer " + str(i) + ":")
            print(self._weight_dif[i])
            print("\n")

        for i in range(len(self._bias_dif)):

            print("Bias Gradient Layer " + str(i) + ":")
            print(self._bias_dif[i])
            print("\n")

        for i in range(len(self._activation_dif)):

            print("Activation Gradient Layer " + str(i) + ":")
            print(self._activation_dif[i])
            print("\n")

    def grad_step(self, alpha = 0.001):

        for i in range(len(self._weight_dif)):

            self._weight[i] = self._weight[i] - alpha * self._weight_dif[i]

        for i in range(len(self._bias_dif)):

            self._bias[i] = self._bias[i] - alpha * self._bias_dif[i]

    def error(self, truth, show=False):

        val = 0

        for i in (self._activation[-1] - truth):

            val += i[0] * i[0]

        if show : print("Error is " + str(val))

        return val
        
    def print_params(self): #NAbQ

        for i in range(len(self._weight)):

            print("Weight Layer " + str(i) + ":")
            print(self._weight[i])
            print("\n")

        for i in range(len(self._bias)):

            print("Bias Layer " + str(i) + ":")
            print(self._bias[i])
            print("\n")

    def train(self, x, y, iter, alpha = 0.000001, init_mode = "uniform", resume=False, momentum=False, beta=0.2, batch_size = 100, min_max = False):

        INPUT_WIDTH = np.size(x, 0)
        OUTPUT_WIDTH = np.size(y,0)
        NUM_WEIGHT = len(self._net_shape) - 1
        NUM_DATA = np.size(x,1)

        if (not resume): self.init(mode=init_mode)

        self._param_history = []
        self._cost_history = []

        self._cost_history.append(self.set_error(x,y))

        self._weight_dif, self._bias_dif = self.grad_buffer()

        if (min_max): self.init_minmax()

        print("starting training")

        for i in range(iter):

            if (i%50 == 0): print("i = " + str(i))

            # reset gradient buffer
            weight_grad_buf, bias_grad_buf = self.grad_buffer()

            # reset cost variable
            val = 0

            # iterate through each point in the dataset
            for j in range(batch_size):

                sel = np.random.randint(0, NUM_DATA)

                # perform forward propagation to get activation
                self.infer(np.reshape(x[:,sel], (INPUT_WIDTH, 1)))

                # user forward pass to calculate square error
                val += self.error(np.reshape(y[:,sel], (OUTPUT_WIDTH, 1)), show=False)

                # perform backpropagation to calculate gradient
                self.grad(np.reshape(y[:,sel], (OUTPUT_WIDTH, 1)))

                # add gradient to running sum buffer
                for k in range(NUM_WEIGHT):

                    weight_grad_buf[k] += self._weight_dif[k]
                    bias_grad_buf[k] += self._bias_dif[k]

            # save MSE using val
            self._cost_history.append(val / NUM_DATA)

            # record absolute min max values for gradient (DEV PURPOSES)
            if (min_max): self.update_minmax()

            # set real gradient as average of individual gradients
            for k in range(NUM_WEIGHT):

                if (momentum):
                    self._weight_dif[k] = (1-beta) * self._weight_dif[k] + beta * weight_grad_buf[k] / NUM_DATA
                    self._bias_dif[k] = (1-beta) * self._bias_dif[k] + beta * bias_grad_buf[k] / NUM_DATA
                else:
                    self._weight_dif[k] = weight_grad_buf[k] / NUM_DATA
                    self._bias_dif[k] = bias_grad_buf[k] / NUM_DATA

            # take one step in the direction of the gradient
            self.grad_step(alpha)

            # save every 50th set of weights
            if (i%50 == 0):
                self._param_history.append([self._weight, self._bias])   

    def set_error(self, x, y): #NAbQ

        NUM_DATA = np.size(x, 1)
        INPUT_WIDTH = np.size(x, 0)
        OUTPUT_WIDTH = np.size(y,0)

        val = 0

        for i in range(NUM_DATA):

            self.infer(np.reshape(x[:,i], (INPUT_WIDTH, 1)))

            val += self.error(np.reshape(y[:,i], (OUTPUT_WIDTH, 1)), show=False)

        return val / NUM_DATA

    def init_minmax(self): #NAbQ

        self._weight_dif_max, self._bias_dif_max = self.grad_buffer()
        self._weight_dif_min, self._bias_dif_min = self.grad_buffer()

    def update_minmax(self): #NAbQ

        for i in range(self._num_weight):

            self._weight_dif_max[i] = np.maximum(self._weight_dif_max[i], np.absolute(self._weight_dif[i]))
            self._weight_dif_min[i] = np.minimum(self._weight_dif_min[i], np.absolute(self._weight_dif[i]))

            self._bias_dif_max[i] = np.maximum(self._bias_dif_max[i], np.absolute(self._bias_dif[i]))
            self._bias_dif_min[i] = np.minimum(self._bias_dif_min[i], np.absolute(self._bias_dif[i]))

    def print_minmax(self, plot = False): #NAbQ

        data = []
        labels = []

        for i in range(self._num_weight):

            print("Layer " + str(i))
            
            print("Weight MIN: " + str(np.min(self._weight_dif_min[i])) + " MAX: " + str(np.min(self._weight_dif_max[i])))

            print("Bias MIN: " + str(np.min(self._bias_dif_min[i])) + " MAX: " + str(np.min(self._bias_dif_max[i])))

            print("\n")

            if (plot):
                data.append(np.absolute(self._weight[i].flatten()))
                labels.append("W" + str(i))

                data.append(np.absolute(self._bias[i].flatten()))
                labels.append("B" + str(i))

        if (plot):
            plt.boxplot(data, labels=labels)
            
    def error_for_autograd(self, input, truth):

        self.infer(input)

        return self.error(truth)





            


        
     





    

