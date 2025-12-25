import sys
import os
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import(

        AdaBoostRegressor,
        RandomForestRegressor,
        GradientBoostingRegressor
)
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor



from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evalution

@dataclass
class ModelTrainerConifg:
    trained_model_file_path = os.path.join("artifacts","model.pkl") # save my trained model 

class Model_trainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConifg()

    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Spliting training and test input data")
            X_train,y_train,X_test,y_test= (

                    train_array [:,:-1],
                    train_array[:,-1],
                    test_array[:,:-1],
                    test_array[:,-1]
            )
            models={

                            "Linear Regression":LinearRegression(),
                            "Lasso":Lasso(),
                            "Ridge" :Ridge(),
                            "Grident SearchCV":GradientBoostingRegressor(),
                            "K-Neighbors Regressor": KNeighborsRegressor(),
                            "DecisionTreeRegressor": DecisionTreeRegressor(),
                            "RandomForestRegressor":RandomForestRegressor(),
                            "XG Boot Regressor":XGBRegressor(),
                            "Ada Boost Regressor":AdaBoostRegressor(),
                            "Cat Boost Regressor":CatBoostRegressor(verbose=False),
                    }

            model_report:dict=evalution(X_train = X_train, X_test = X_test,y_train = y_train, y_test = y_test, models = models)   

            #to find the best score form dict
            best_model_score = max(sorted((model_report.values())))

            #find the best model fomr dict 
            best_model_name = list(model_report.keys())[

                list(model_report.values()).index(best_model_score)
            ]

            if best_model_score<0.6:
                raise CustomException("No Best Model Found")
            
            logging.info("Best model found on both training ds & testing ds")

            best_model = models[best_model_name]


            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj=best_model

            )

            predict = best_model.predict(X_test)
            r2_scoreee = r2_score(y_test,predict)
            return r2_scoreee

        except Exception as e:
            raise CustomException(e,sys)
            


