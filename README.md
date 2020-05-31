# simple-LSTM-forecast_model
This is a simple, easy to understand LSTM forecast model for Stock Price Prediction. The dataset used here contains daywise details of the GOOGL stock  from May,2009 to May,2018. 


For simplicity's sake, the model is trained only on the daily closing price. It is trained on the closing prices of the past 30 days to predict the closing prices of the next 15 days. The data is normalized using MinMaxScaler before being fed to the model. No other preprocessing is done.

A sequential model is used containing two LSTM layers and a final regression layer with a dropout of 0.5. 5 epochs through the model gives reasonable accuracy. The loss is measured using mean squared error metric with Adam optimizer.  



The output of the model is processed back into its original non standardized form. Finally, the predicted and actual values of the stock prices is displayed via a plot.


OBSERVATIONS(THINGS TO RECTIFY):


1.As this is an elementary project with limited data, no validation set is used. 


2.The train-test split is 3:1. This ratio is way less than usual models having train-validation-test split of about 0.94:0.4:02 which have  dataset size in the millions. 


3.Owing to low ratio of train-test and the absence of validation set, this simplistic model results in gross overfitting.To rectify this, a high dropout value is used.

