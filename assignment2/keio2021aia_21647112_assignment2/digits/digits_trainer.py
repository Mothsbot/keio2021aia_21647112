from . import *

class Digits_Trainer:
    
    def __init__(self, dataset, model):
        
        self.dataset = dataset
        self.model = model
        self.loss = CrossEntropy()
        
    def accuracy(self):

        acc = None
        # YOUR_CODE
        # hint: return the accuracy (i.e. the percentage of digits classified correctly) of the current model on the dataset given in the constructor
        acc = 100 * np.mean([1 if self.model(x) == np.argmax(y) else 0 for x, y in self.dataset.samples])
        return acc
    
    def step(self, lr):

        for param, grad in self.model.params_and_grads():
            
            param -= lr * grad
            
    def train(self, lr, ne):
        
        print('initial accuracy: %.3f\n\n' %(self.accuracy()))
        
        print('training model on data...\n')
        print('='*80+'\n')
    
        for epoch in range(1, ne + 1):

            epoch_loss = 0.0

            for batch in self.dataset:
               
                predicted = None
                # YOUR_CODE
                # hint: use the model to generate predictions (digit labels) for a batch, and then use the loss given in
                # the constructor to update the epoch_loss variable
                outputs = []

                for (img, target) in zip(batch['inputs'], batch['targets']):
                    predicted = np.zeros(10)
                    yhat = self.model(img)
                    predicted[yhat] += 1
                    outputs.append(predicted)

                epoch_loss += self.loss.loss(np.array(outputs), (np.array(target)))

                grad = None
                #YOUR_CODE
                # hint: compute the gradient of the loss with respect to its inputs (not the model inputs - watch the
                # lecture video for explanation)

                for i in outputs:
                    grad = self.loss.grad(np.array(i), np.array(target))
                    inputs = grad

                self.model.backward(grad)
                self.step(lr)
                
            print("""epoch %d:\n
            \t loss = %.3f\n
            \t accuracy=%.3f""" % (epoch, epoch_loss, self.accuracy()))
            
        print('='*80+'\n')
        print('training complete!\n\n')
        print('final accuracy: %.3f' %(self.accuracy()))
