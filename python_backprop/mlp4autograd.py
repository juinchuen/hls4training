# import numpy as np

# we use a 1 hidden layer, 16 neuron nn
# to map 3 x,y,z values to 5 angle values

# INPUT_N = 3
# HIDDEN_N = 16
# OUTPUT_N = 5

class mlp4autograd:

    def __init__(self):
        self._net_shape = []
        self._num_weight = 0
        self._weight = []
        self._bias = []
        self._weight_dif = []
        self._bias_dif = []
        self._activation = []
        self._pre_relu = []
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

    def infer(self, x, params): #quanted

        weights = params[0]
        bias = params[1]
        
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
                self._pre_relu.append(self.enforce_quant_range( np.divide(np.dot(weights[i], self._activation[i]), 2**self._q_bits) + bias[i]))
                self._activation.append(self.relu(self._pre_relu[-1]))
        else:
            for i in range(len(self._net_shape)-1):
                self._pre_relu.append(np.dot(weights[i], self._activation[i]) + bias[i])
                self._activation.append(self.relu(self._pre_relu[-1]))

        return self._activation[-1]

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

        return self._weight_dif, self._bias_dif

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

    def error(self, truth, show=False):

        val = 0

        for i in (self._activation[-1] - truth):

            val += i[0] * i[0]

        if show : print("Error is " + str(val))

        return val