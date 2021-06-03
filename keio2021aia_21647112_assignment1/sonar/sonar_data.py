from . import *

class Sonar_Data:

    def __init__(self, data_file_path='', data_file_name='sonar_data.pkl'):
                
        self.data = []
        
        data_file = data_file_path + "/" + data_file_name
        data_open = open(data_file, "rb")
        data_load = pkl.load(data_open)
        
        for i in data_load['m']:
            self.data.append([i, 1])
            
        for i in data_load['r']:
            self.data.append([i, -1])

        self.index = -1

        self.shuffle()

    def __iter__(self):
        
        return self

    def __next__(self):

          self.index += 1
          if self.index < len(self.data):
              raise StopIteration
          return self.data[self.index]
    
    def shuffle(self):
        
        random.shuffle(self.data)
    
    def __len__(self):
        
        return len(self.data)
    
