

# for FastAPI
from fastapi import FastAPI
import uvicorn
import pydantic
from datetime import *
import pandas_datareader as pdr
import numpy as np
# for transformer
from utils.Time2Vector import Time2Vector
from utils.Attention import MultiAttention, SingleAttention
from utils.Encoder import TransformerEncoder
from tensorflow import keras
from keras.utils import custom_object_scope
from keras.models import load_model

def Transformer(ticker):
  seq_len = 32

  start_date = datetime.now() - timedelta(48)
  start_date = datetime.strftime(start_date, '%Y-%m-%d')

  df = pdr.DataReader(ticker, data_source='yahoo', start=start_date)

  df.drop('Volume', axis=1, inplace=True)

  # df[df.columns] = scaler.fit_transform(df)
  df = df[['High', 'Low', 'Open', 'Adj Close', 'Close']]

  '''Create training, validation and test split'''

  test_data = df.values

  # Test data
  X_test, y_test = [], []
  for i in range(seq_len, len(test_data)):
      X_test.append(test_data[i - seq_len:i])
      y_test.append(test_data[:, 4][i])
  X_test, y_test = np.array(X_test), np.array(y_test)

  custom_objects = {"Time2Vector": Time2Vector,
                    "MultiAttention": MultiAttention,
                    'TransformerEncoder': TransformerEncoder}
  with custom_object_scope(custom_objects):
      final_model = load_model('Transformer+TimeEmbedding.hdf5')

  trans_prediction = float(final_model.predict(X_test)[-1])
  trans_difference = trans_prediction - df.Close[-1]

  return trans_prediction, trans_difference


app = FastAPI()


@app.get('/')
def index():
    return {'message': 'This is your fav stock predictor!'}


@app.post('/predict')
async def predict_price(data: str):
    if data == 'F':
      
      trans_prediction, trans_difference = Transformer(data)

      return {
        
        'Transformer prediction': trans_prediction
            }

    else:
      return {"the ticker not supported yet"}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
