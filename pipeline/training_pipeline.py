from src.data_processing import DataProcessing
from src.model_training import ModelTraining
if __name__=="__main__":
    
    processor=DataProcessing('artifacts/raw/data.csv', "artifacts/processed")
    processor.run()
   
    trainer=ModelTraining(processed_data_path='artifacts/processed/', model_output_path='artifacts/models/')
    trainer.run()