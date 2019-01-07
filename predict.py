#!/usr/local/bin/python3
import sys, getopt
import pandas as pd
import numpy as np
import gc
import utilities
from keras.callbacks import TensorBoard
from model import LSTM_model

path = sys.argv[1]
result_loc = sys.argv[2]
lstm_model  = LSTM_model()
model = lstm_model.get_model()
model.load_weights('RNN_Alpha_LSTM.h5')

print("Load test data...")
test_df = pd.read_csv(path, dtype=utilities.dtypes, usecols=['ip','app','device','os', 'channel','click_time'])
test_df['hour'] = utilities.date_unpacker(test_df)[0]
test_df['day'] = utilities.date_unpacker(test_df)[1]
test_df['wday']  = utilities.date_unpacker(test_df)[2]
test_df.drop(['click_time',],1,inplace=True)

result_holder = pd.DataFrame()
result_holder['id'] = test_df.index.tolist()
test_df = utilities.data_stream_molder(test_df)



print("predicting....")
result_holder['is_attributed'] = model.predict(test_df, batch_size=20000,verbose=2)
del test_df; gc.collect()
print("writing....")
result_holder.to_csv(sys.argv[2],index=False)