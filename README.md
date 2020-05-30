# simple-LSTM-forecast_model
This is a simple, easy to understand LSTM forecast model for Stock Price Prediction. The dataset used here contains daywise details of the GOOGL stock  from May,2009 to May,2018. 


For simplicity's sake, the model is trained only on the daily closing price. It is trained on the closing prices of the past 30 days to predict the closing prices of the next 15 days. The data is normalized using MinMaxScaler before being fed to the model. No other preprocessing is done.

A sequential model is used containing two LSTM layers and a final regression layer with a dropout of 0.5. 5 epochs through the model gives reasonable accuracy. The loss is measured using mean squared error metric with Adam optimizer.  


As this is an elementary project with limited data, no validation set is used. The train-test split is 3:1.(here, percentage of training data is way less than usual models(about 99.9:0.1) with  dataset size in the millions. As a result, this simplistic model results in gross overfitting.To rectify this, a high dropout value is used.


The output of the model is processed back into its original non standardized form. Finally, the predicted and actual values of the stock prices is displayed via a plot.
