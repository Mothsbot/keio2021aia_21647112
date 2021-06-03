##
from sonar.sonar_data import * 
from sonar.sonar_model import * 
from sonar.sonar_trainer import * 
from cats.cat_data import *
from cats.cat_model import *
from cats.cat_trainer import *
##

##
def main():

    data = Sonar_Data('data', 'sonar_data.pkl')
    model = Sonar_Model()
    trainer = Sonar_Trainer(data, model)
    costs, accuracies = trainer.train(0.05, 100)
    model.save_model()

    data = Cats_Data('data', 'cat_data.pkl')
    model = Cats_Model(data, model)
    trainer = Cats_Trainer()
    costs, accuracies = trainer.train(0.05, 100)
    model.save_model()
##

##
if __name__ == "__main__":
    main()
##
