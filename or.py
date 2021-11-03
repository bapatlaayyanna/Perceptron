from utils.model import Perceptron
from utils.all_utils import prepare_data,save_model
import pandas as pd
import numpy as np

def main(data,eta,epoch,filename):
    
    df = pd.DataFrame(data)
    print(df)
    
    X,y = prepare_data(df)
    
    
    model = Perceptron(eta=eta, epochs=epoch)
    model.fit(X, y)
    
    _ = model.total_loss()
    
    save_model(model,filename)

if __name__=='__main__':#Entry point
    OR = {
    "x1": [0,0,1,1],
    "x2": [0,1,0,1],
    "y": [0,1,1,1]}

    ETA = 0.3 
    EPOCHS = 10
    
    main(data=OR,eta=ETA,epoch=EPOCHS,filename="or.model")
