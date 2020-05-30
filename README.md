# simple-LSTM-forecast_model
This is a simple, easy to understand LSTM forecast model for Stock Price Prediction. The dataset used here contains daywise details of the GOOGL stock  from May,2009 to May,2018. 


For simplicity's sake, the model is trained only on the daily closing price. It is trained on the closing prices of the past 30 days to predict the closing prices of the next 15 days.

A sequential model is used containing two LSTM layers and a final regression layer with a dropout of 0.1. 5 epochs through the model gives reasonable accuracy. 


As this is an elementary project with limited data, no validation set is used. The train-test split is
