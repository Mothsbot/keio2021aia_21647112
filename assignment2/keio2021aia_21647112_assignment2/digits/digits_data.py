from . import *

class Digits_Data:

    def __init__(self, relative_path='../data/', data_file_name='digits_data.pkl', batch_size=64):

        self.batch_size = batch_size
        self.index = -1
        
        with open('%s%s' % (relative_path, data_file_name), mode='rb') as f:
            
            digits_data = pkl.load(f)
            
        self.samples = []
        #YOUR CODE
        for i in range(10):
            for data in digits_data['train'][i]:
                flattened_img = data.reshape(-1)
                self.samples.append([flattened_img, i])

        # hint: you will need to flatten the images to represent them as vectors (numpy arrays) and pair them with digit
        # labels from the training data
        #dict['train']['test'], dict[0]...[9] w/ list of arrays of pixels for image

        self.shuffle()
        
        self.starts = np.arange(0, len(self.samples), self.batch_size)
        #Initialization array (start = 0, end = # of samples, split by batches i.e. 64)

    def __iter__(self):
                
        return self

    def __next__(self):

        self.index += 1
        
        if self.index + 1 > len(self.starts):
            
            self.index = -1
            self.shuffle()
            raise StopIteration
            
        inputs = None
        targets = None
        #YOUR_CODE hint: use the starts initialized in the last line of the constructor and the batch size to generate a batch of inputs and the corresponding batch of targets
        target = []
        # Array y0..y9, Make target = 1
        for i in range(len(self.samples)):
            digits = np.zeros(10)
            y = self.samples[i][1]
            digits[y] += 1
            target.append(digits)

        features = [self.samples[i][0] for i in range(len(self.samples))]

        # slice samples w/ starts
        # inputs = [features[self.starts[i]:self.starts[i + 1]] for i in range(self.starts.shape[0] - 1)]
        # targets = [target[self.starts[i]:self.starts[i + 1]] for i in range(self.starts.shape[0] - 1)]
        inputs = [features[self.starts[i]:self.starts[i+1]] for i in range(len(self.starts)-1)]
        targets = [target[self.starts[i]:self.starts[i+1]] for i in range(len(self.starts)-1)]

        return {'inputs': inputs, 'targets': targets}

    def shuffle(self):
        
        random.shuffle(self.samples)

