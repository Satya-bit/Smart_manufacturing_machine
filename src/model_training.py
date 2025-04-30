import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.logger import get_logger
from src.custom_exception import CustomException
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier



logger=get_logger(__name__)

class ModelTraining:
    def __init__(self,processed_data_path, model_output_path):
        self.processed_data_path=processed_data_path
        self.model_output_path=model_output_path
        self.clf=None
        self.stk=None
        self.X_train, self.X_test, self.y_train, self.y_test=None, None, None, None
        
        os.makedirs(self.model_output_path, exist_ok=True)
        logger.info("Model Training initialized....")
        
    def load_data(self):
        try:
            self.X_train=joblib.load(os.path.join(self.processed_data_path, 'X_train.pkl'))
            self.X_test=joblib.load(os.path.join(self.processed_data_path, 'X_test.pkl'))
            self.y_train=joblib.load(os.path.join(self.processed_data_path, 'y_train.pkl'))
            self.y_test=joblib.load(os.path.join(self.processed_data_path, 'y_test.pkl'))
            
            logger.info("Data loaded successfully ...")
            
        except Exception as e:
            logger.error(f"Error in loading data {e}")
            raise CustomException("Failed to load data", e)
        
    def train_model(self):
            try:
                self.clf=LogisticRegression(random_state=42,max_iter=1000)
                self.clf.fit(self.X_train, self.y_train)
                joblib.dump(self.clf, os.path.join(self.model_output_path, 'logistic_model.pkl'))
                
                logger.info("Logistic Regression trained successfully ...")
              
                
                # self.stk=StackingClassifier(
                #     estimators=[
                #         ('rf', RandomForestClassifier(random_state=42)),
                #         ('gb', GradientBoostingClassifier(random_state=42))
                #     ],
                #     final_estimator=LogisticRegression(random_state=42)
                # )
                
                # self.stk.fit(self.X_train, self.y_train)
                # joblib.dump(self.stk, os.path.join(self.model_output_path, 'stacking_model.pkl'))
                
                # logger.info("Stacking trained successfully ...")
            except Exception as e:
                logger.error(f"Error in training model {e}")
                raise CustomException("Failed to train model", e)
                
                
    def evaluate_model(self):
                try:
                    y_pred=self.clf.predict(self.X_test)
                    acc=accuracy_score(self.y_test, y_pred)
                    recall=recall_score(self.y_test, y_pred,average='weighted')
                    precision=precision_score(self.y_test, y_pred,average='weighted')
                    f1=f1_score(self.y_test, y_pred,average='weighted')
                    logger.info(f"Logistic Regression trained with accuracy: {acc}, recall: {recall}, precision: {precision}, f1: {f1}")
                    
                    
                    # y_pred=self.stk.predict(self.X_test)
                    # acc=accuracy_score(self.y_test, y_pred)
                    # recall=recall_score(self.y_test, y_pred,average='weighted')
                    # precision=precision_score(self.y_test, y_pred,average='weighted')
                    # f1=f1_score(self.y_test, y_pred,average='weighted')
                    # logger.info(f"Stacking trained with accuracy: {acc}, recall: {recall}, precision: {precision}, f1: {f1}")
                except Exception as e:
                    logger.error(f"Error while evaluating model {e}")
                    raise CustomException("Failed to evaluate model", e)
    def run(self):
        self.load_data()
        self.train_model()
        self.evaluate_model()
    
if __name__=="__main__": #If you don’t write the if __name__ == "__main__": block in your script, the trainer.run() part at the bottom will execute even when the script is imported into another script as a module
    trainer=ModelTraining(processed_data_path='artifacts/processed/', model_output_path='artifacts/models/')
    trainer.run()
                
            
            
        
        
        
     