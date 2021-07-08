from . import *

class Digits_Model:

    def __init__(self, dim_input=784, dim_hidden=[100], dim_out=10):

        self.layers = []
        
        #YOUR_CODE

        self._dim_input = dim_input
        self._dim_hidden = dim_hidden
        self._dim_out = dim_out

        self.layers.append(Linear(self._dim_input, self._dim_hidden[0]))
        self.layers.append(Tanh())
        self.layers.append(Linear(self._dim_hidden[len(self._dim_hidden) - 1], self._dim_out))
        #self.layers.append(Tanh())

        #-----
        self._predict = softmax
        
    def __str__(self):

        return "Simple Neural Network\n\
        \tInput dimension: %d\n\
        \tHidden dimensions: %d\n\
        \tOutput dimension: %d\n" % (self._dim_input, self._dim_hidden, self._dim_out)

    def __call__(self, x):
        
        prediction = None
        # YOUR_CODE
        # hint: use the forward and predict functions to obtain a probability distribution over digits, then
        # pick the highest probability digit
        prediction = np.argmax(self._predict(self.forward(x)))
        return prediction

    def load_model(self, file_path):
        
        with open(file_path, mode='rb') as f:
            loaded_model = pkl.load(f)

        self.__dict__.update(loaded_model.__dict__)

    def save_model(self):

        with open('results/digits_model.pkl','wb') as f:
            pkl.dump(self, f)
    
    def forward(self, inputs):
        
        outputs = None
        outputs = inputs
        # YOUR_CODE
        # hint: transform a batch of inputs (images) (64, 784) into a batch of outputs (final layer activations) using
        # your model's layers in the order that they are supposed to be applied
        for layer in self.layers:
            outputs = layer.forward(outputs)
        # -------
        return outputs

    def backward(self, grad):
        
        for layer in reversed(self.layers):
            
            grad = layer.backward(grad)
            
        return grad

    def params_and_grads(self):
        
        for layer in self.layers:
            
            for name, param in layer.params.items():
                
                grad = layer.grads[name]
                yield param, grad
