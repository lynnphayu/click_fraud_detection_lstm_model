#!/usr/local/bin/python3
import sys, getopt
import pandas as pd
import numpy as np
import gc
import utilities
from keras.callbacks import TensorBoard
from model import LSTM_model

path = "./" 
print("Load training data...")
train_df = pd.read_csv(path+"train.csv", dtype=utilities.dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])
# train_df['hour'] = utilities.date_unpacker(train_df)[0]
# train_df['day'] = utilities.date_unpacker(train_df)[1]
# train_df['wday']  = utilities.date_unpacker(train_df)[2]
train_df.drop(['click_time'],1,inplace=True)
y_train = train_df['is_attributed']
train_df.drop(['is_attributed'],1,inplace=True)

train_df = utilities.data_stream_molder(train_df)

lstm_model  = LSTM_model()
model = lstm_model.get_model()

tensorboard = TensorBoard(log_dir='./logs_lstm_no_time', histogram_freq=0,
                          write_graph=True, write_images=False)

model.fit(train_df, y_train,
            batch_size=100000, epochs=2, validation_split=0.2,
            shuffle=True,callbacks=[tensorboard],verbose=2)

model.save_weights('LSTM_alpha.h5')
del train_df,y_train;
gc.collect()


# def main(argv):
#    training_data = ''
#    output_model = ''
#    try:
#       opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
#    except getopt.GetoptError:
#       print 'test.py -i <traindata> -o <outputmodel>'
#       sys.exit(2)
#    for opt, arg in opts:
#       if opt == '-h':
#          print 'test.py -i <traindata> -o <outputmodel>'
#          sys.exit()
#       elif opt in ("-i", "--ifile"):
#          training_data = arg
#       elif opt in ("-o", "--ofile"):
#          outputfile = arg

# if __name__ == "__main__":
#    main(sys.argv[1:])