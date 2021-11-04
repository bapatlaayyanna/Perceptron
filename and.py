from utils.model import Perceptron
from utils.all_utils import prepare_data,save_model
import pandas as pd
import numpy as np
import logging
import os

logging_str="[%(asctime)s:%(levelname)s:%(module)s] %(message)s"
os.makedirs("logs",exist_ok=True)
logging.basicConfig(filename=os.path.join("logs","running_log.log"),level=logging.INFO,format=logging_str)

def main(data,eta,epoch,filename):
    
    df = pd.DataFrame(data)
    logging.info(f"This is and dataframe {df}")
    
    X,y = prepare_data(df)
    
    
    model = Perceptron(eta=eta, epochs=epoch)
    model.fit(X, y)
    
    _ = model.total_loss()
    
    save_model(model,filename)

if __name__=='__main__':#Entry point
    AND = {
    "x1": [0,0,1,1],
    "x2": [0,1,0,1],
    "y": [0,0,0,1]}

    ETA = 0.3 
    EPOCHS = 10
    
    main(data=AND,eta=ETA,epoch=EPOCHS,filename="and.model")
