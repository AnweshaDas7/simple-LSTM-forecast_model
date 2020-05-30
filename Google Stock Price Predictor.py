import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout
from sklearn.preprocessing import MinMaxScaler
np.random.seed(1337) # for reproducibility

df = pd.read_csv('C:\\Users\\Anwesha\\Desktop\\GOOGL.csv')#GETTING DATA AS A PANDAS DATAFRAME
df=df.iloc[::-1]#REVERSING THE DATAFRAME TO CHANGE TO ASCENDING
data=df[['Date','Close']]#EXTRACTING THE CLOSING PRICE COLUMN WHICH IS TO BE WORKED UPON

scaler = MinMaxScaler(feature_range=(0, 1))#FOR NORMALIZING DATA


data.drop('Date', axis=1, inplace=True)
train = data.values[0:-500,:] #TAKING TRAINING DATASET AS THE ENTIRE DATASET EXCEPT THE MOST RECENT 500 DATAPOINTS
test = data.values[-500:,:]   #TAKING TESTING DATASET AS THE MOST RECENT 500 DATAPOINTS

scaled_data = scaler.fit_transform(data.values)#SCALING ENTIRE DATASET

x_train, y_train = [], []
for i in range(0,len(data)-530):
    x_train.append(scaled_data[i:i+30,0])#SETTING X(GIVEN DATA) AS 30 SAMPLES
    y_train.append(scaled_data[i+30:i+45,0])#SETTING Y(TO-BE-PREDICTED DATA) AS THE NEXT 15 SAMPLES
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))


# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(15))
model.add(Dropout(0.1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=5, batch_size=1, verbose=2)#5 EPOCHS GIVE REASONABLY GOOD ACCURACY

#predicting 500 values, using past 30 from the train data
inputs = data[-len(test):].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)#STANDARDIZING DATASET TO FEED INPUT INTO MODEL

X_test = []
for i in range(0,len(inputs)-45):
    X_test.append(inputs[i:i+30,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))


inputs  = scaler.inverse_transform(inputs)#RETURNING TO NON STANDARDIZED FORM TO GET OUTPUT VALUES
Y_test = []
for i in range(30,len(inputs)-15):
    Y_test.append(inputs[i:i+15,0])
Y_test = np.array(Y_test)


closing_price = model.predict(X_test)#FEEDING STANDARDIZED DATA INTO MODEL FOR PREDICTION
closing_price = scaler.inverse_transform(closing_price)#CONVERTING PREDICTIONS TO NON STANDARDIZED FORM
mse = ((Y_test-closing_price)**2).mean(axis=0)#CALCULATING RMSE FOR ACTUAL VS PREDICTED DATA
rms=(mse.mean())**0.5
print(rms)

x_predictions=np.reshape(scaler.transform(inputs)[-30:],(1,30,1))#TAKING 30 LATEST VALUES TO PREDICT NEXT 15 DAYS
predictions=model.predict(x_predictions)
predictions = scaler.inverse_transform(predictions)
predictions=np.reshape(predictions,15)
print(predictions)


data['Date']=df['Date']
data.set_index('Date')
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(data.Date, data.Close, label='Given Closing Price', c='r')#PLOTTING GIVEN DATASET
w=np.arange(len(data),len(data)+15,1)
ax.plot(w,predictions,label='Predicted Closing Price',c='b')#PLOTTING PREDICTED DATA AFTER TRAINING MODEL
plt.title('Predicted Closing Price')
plt.xticks(np.arange(0,2000,100), data.Date[::100], rotation=45)
ax.legend(loc='best')
plt.show()


