library(dplyr)
library(forecast)
library(keras)
library(tidyverse)
library(tensorflow)

data <- read.csv("D:/Data for Research/Tourism/Tourism_Data.csv")

ts.plot(ts(data$Cases, start = c(2008,1), end = c(2023,12), frequency = 12), lwd = 2, ylab = 'Tourist Arrivals')

# Data Preprocessing
data <- data %>% select(Cases)

# Splitting the data into training and validation sets
train_data <- data[1:180, ]
valid_data <- data[181:192, ]

# Scaling the data
max_value <- max(train_data)
min_value <- min(train_data)
scale_data <- function(data) {
  scaled_data <- (data - min_value) / (max_value - min_value)
  return(as.data.frame(scaled_data))  # Ensure the result is a data frame
}
inverse_scale_data <- function(data) data * (max_value - min_value) + min_value

train_scaled <- scale_data(train_data)
valid_scaled <- scale_data(valid_data)

# Preparing the data for the deep learning models
create_dataset <- function(data, look_back = 1) {
  x_data <- list()
  y_data <- list()
  
  for (i in 1:(nrow(data) - look_back)) {
    x_data[[i]] <- data[i:(i + look_back - 1), ]
    y_data[[i]] <- data[i + look_back, ]
  }
  
  x_array <- array(unlist(x_data), dim = c(length(x_data), look_back, ncol(data)))
  y_array <- array(unlist(y_data), dim = c(length(y_data)))
  
  return(list(x = x_array, y = y_array))
}


look_back <- 1
train_dataset <- create_dataset(train_scaled, look_back)
valid_dataset <- create_dataset(valid_scaled, look_back)


##########################
#                        #
#  Gated Recurrent Unit  #
#                        #
##########################


Sys.setenv(TF_ENABLE_ONEDNN_OPTS = "0")

# Defining the GRU model
model <- keras_model_sequential() %>%
  layer_gru(units = 32, input_shape = c(1, 1)) %>%
  layer_dense(units = 1)

model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam()
)


# Training the model
history <- model %>% fit(
  x = train_dataset$x, y = train_dataset$y,
  epochs = 100,
  batch_size = 1,
  validation_data = list(valid_dataset$x, valid_dataset$y)
)

# Making predictions
train_predict <- model %>% predict(train_dataset$x)
valid_predict <- model %>% predict(valid_dataset$x)

# Inverse scaling for plotting and evaluation
train_predict <- inverse_scale_data(train_predict)
train_predict <- ts(train_predict, start = c(2008,2), end = c(2022,12), frequency = 12)
train_y <- inverse_scale_data(train_dataset$y)
train_y <- ts(train_y, start = c(2008,2), end = c(2022,12), frequency = 12)

valid_predict_gru <- inverse_scale_data(valid_predict)
valid_predict_gru <- ts(valid_predict_gru, start = c(2023,2), end = c(2023,12), frequency = 12)
valid_y_gru <- inverse_scale_data(valid_dataset$y)
valid_y_gru <- ts(valid_y, start = c(2023,2), end = c(2023,12), frequency = 12)

# Plotting results
plot(train_y, type = 'l', col = 'blue', ylab = 'Tourist Arrivals', lwd = 2)
lines(train_predict, col = 'red', lwd = 2)
legend('topleft', lty=c(1,1), col = c('blue','red'), legend = c('Actual','Fitted'), cex=0.7,inset=0.025, lwd = 2)

plot(valid_y_gru, type = 'l', col = 'blue', ylab = 'Tourist Arrivals', lwd = 2, ylim = c(0,600000))
lines(valid_predict_gru, col = 'red', lwd = 2)
legend('topleft', lty=c(1,1), col = c('blue','red'), legend = c('Actual','Fitted'), cex=0.7,inset=0.025, lwd = 2)

#Accuracy Measures

Metrics::rmse(as.numeric(train_y),as.numeric(train_predict))
Metrics::mae(as.numeric(train_y),as.numeric(train_predict))
Metrics::mape(as.numeric(train_y),as.numeric(train_predict))*100

Metrics::rmse(as.numeric(valid_y_gru),as.numeric(valid_predict_gru))
Metrics::mae(as.numeric(valid_y_gru),as.numeric(valid_predict_gru))
Metrics::mape(as.numeric(valid_y_gru),as.numeric(valid_predict_gru))*100

# Forecasting function
forecast_next_points <- function(model, last_data, n_forecast, look_back) {
  forecasted_data <- numeric(n_forecast)
  current_batch <- last_data
  
  for(i in 1:n_forecast) {
    # Ensure current_batch is a matrix
    if (!is.matrix(current_batch)) {
      current_batch <- matrix(current_batch, nrow = look_back, ncol = ncol(last_data))
    }
    
    # Reshape the input to match the expected shape: (1, look_back, number of features)
    current_batch_reshaped <- array(current_batch, dim = c(1, look_back, ncol(last_data)))
    
    # Predict the next point
    next_prediction <- model %>% predict(current_batch_reshaped)
    
    # Append the prediction (ensure it is a scalar)
    forecasted_data[i] <- next_prediction[1]
    
    # Update the batch to include the new prediction and remove the oldest data point
    current_batch <- rbind(current_batch[-1, ], next_prediction)
  }
  
  return(inverse_scale_data(forecasted_data))
}

# Ensuring last_train_data has the correct shape and type
last_train_data <- as.matrix(train_scaled[(nrow(train_scaled) - look_back + 1):nrow(train_scaled), ])

# Forecasting the next 12 points
n_forecast <- 12
forecasted_values <- forecast_next_points(model, last_train_data, n_forecast, look_back)

print(forecasted_values)

############################
#                          #
#  Long Short-Term Memory  #
#                          #
############################

# Defining the LSTM model
model_lstm <- keras_model_sequential() %>%
  layer_lstm(units = 50, input_shape = c(look_back, 1)) %>% # Assuming 1 feature per time step
  layer_dense(units = 1)

model_lstm %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam()
)

# Training the model
history_lstm <- model_lstm %>% fit(
  x = train_dataset$x, y = train_dataset$y,
  epochs = 100,
  batch_size = 1,
  validation_data = list(valid_dataset$x, valid_dataset$y)
)

# Making predictions
train_predict_lstm <- model_lstm %>% predict(train_dataset$x)
valid_predict_lstm <- model_lstm %>% predict(valid_dataset$x)

# Inverse scaling for plotting and evaluation
train_predict_lstm <- inverse_scale_data(train_predict_lstm)
train_predict_lstm <- ts(train_predict_lstm, start = c(2008,2), end = c(2022,12), frequency = 12)
train_y_lstm <- inverse_scale_data(train_dataset$y)
train_y_lstm <- ts(train_y_lstm, start = c(2008,2), end = c(2022,12), frequency = 12)

valid_predict_lstm <- inverse_scale_data(valid_predict_lstm)
valid_predict_lstm <- ts(valid_predict_lstm, start = c(2023,2), end = c(2023,12), frequency = 12)
valid_y_lstm <- inverse_scale_data(valid_dataset$y)
valid_y_lstm <- ts(valid_y_lstm, start = c(2023,2), end = c(2023,12), frequency = 12)

# Plotting results
plot(train_y_lstm, type = 'l', col = 'blue', ylab = 'Tourist Arrivals', lwd = 2)
lines(train_predict_lstm, col = 'red', lwd = 2)
legend('topleft', lty=c(1,1), col = c('blue','red'), legend = c('Actual','Fitted'), cex=0.7,inset=0.025, lwd = 2)

plot(valid_y_lstm, type = 'l', col = 'blue', ylab = 'Tourist Arrivals', lwd = 2, ylim = c(0,600000))
lines(valid_predict_lstm, col = 'red', lwd = 2)
legend('topleft', lty=c(1,1), col = c('blue','red'), legend = c('Actual','Fitted'), cex=0.7,inset=0.025, lwd = 2)

Metrics::rmse(as.numeric(train_y_lstm),as.numeric(train_predict_lstm))
Metrics::mae(as.numeric(train_y_lstm),as.numeric(train_predict_lstm))
Metrics::mape(as.numeric(train_y_lstm),as.numeric(train_predict_lstm))*100

Metrics::rmse(as.numeric(valid_y_lstm),as.numeric(valid_predict_lstm))
Metrics::mae(as.numeric(valid_y_lstm),as.numeric(valid_predict_lstm))
Metrics::mape(as.numeric(valid_y_lstm),as.numeric(valid_predict_lstm))*100

# Forecasting function for LSTM
forecast_next_points_lstm <- function(model, last_data, n_forecast, look_back) {
  forecasted_data <- numeric(n_forecast)
  current_batch <- last_data
  
  for(i in 1:n_forecast) {
    # Reshape the input to match the expected shape
    current_batch_reshaped <- array(current_batch, dim = c(1, look_back, ncol(last_data)))
    
    # Predict the next point
    next_prediction <- model %>% predict(current_batch_reshaped)
    
    # Append the prediction
    forecasted_data[i] <- next_prediction[1]
    
    # Update the batch
    current_batch <- rbind(current_batch[-1, ], next_prediction)
  }
  
  return(inverse_scale_data(forecasted_data))
}

# Preparing the last data point from the training set
last_train_data_lstm <- as.matrix(train_scaled[(nrow(train_scaled) - look_back + 1):nrow(train_scaled), ])

# Forecasting the next 12 points using LSTM
n_forecast <- 12
forecasted_values_lstm <- forecast_next_points_lstm(model_lstm, last_train_data_lstm, n_forecast, look_back)

print(forecasted_values_lstm)

##################################
#                                #
#  Convolutional Neural Network  #
#                                #
##################################

# We preprocess the data again for the CNN

scale_data <- function(data) {
  scaled_data <- (data - min_value) / (max_value - min_value)
  return(data.frame(Cases = scaled_data))  # Creating a dataframe with the 'Cases' column
}

train_scaled <- scale_data(train_data)
valid_scaled <- scale_data(valid_data)

look_back <- 3  
train_dataset <- create_dataset(train_scaled, look_back)
valid_dataset <- create_dataset(valid_scaled, look_back)

# Define the CNN model
model <- keras_model_sequential() %>%
  layer_conv_1d(filters = 64, kernel_size = 2, activation = 'relu', input_shape = c(look_back, 1)) %>%
  layer_max_pooling_1d(pool_size = 2) %>%
  layer_flatten() %>%
  layer_dense(units = 50, activation = 'relu') %>%
  layer_dense(units = 1)

model %>% compile(
  optimizer = 'adam',
  loss = 'mean_squared_error'
)


# Train the model
history <- model %>% fit(
  x = train_dataset$x, y = train_dataset$y,
  epochs = 100,
  batch_size = 32,
  validation_data = list(valid_dataset$x, valid_dataset$y)
)

# Make predictions
train_predict_cnn <- model %>% predict(train_dataset$x)
valid_predict_cnn <- model %>% predict(valid_dataset$x)

# Inverse scaling for plotting and evaluation
train_predict_cnn <- inverse_scale_data(train_predict_cnn)
train_predict_cnn <- ts(train_predict_cnn, start = c(2008,2), end = c(2022,12), frequency = 12)

train_y_cnn <- inverse_scale_data(train_dataset$y)
train_y_cnn <- ts(train_y_cnn, start = c(2008,2), end = c(2022,12), frequency = 12)

valid_predict_cnn <- inverse_scale_data(valid_predict_cnn)
valid_predict_cnn <- ts(valid_predict_cnn, start = c(2023,4), end = c(2023,12), frequency = 12)
valid_y_cnn <- inverse_scale_data(valid_dataset$y)

# Plotting results
plot(train_y_cnn, type = 'l', col = 'blue', ylab = 'Tourist Arrivals', lwd = 2)
lines(train_predict_cnn, col = 'red', lwd = 2)
legend('topleft', lty=c(1,1), col = c('blue','red'), legend = c('Actual','Fitted'), cex=0.7,inset=0.025, lwd = 2)

#Accuracy Measures

Metrics::rmse(as.numeric(train_y_cnn),as.numeric(train_predict_cnn))
Metrics::mae(as.numeric(train_y_cnn),as.numeric(train_predict_cnn))
Metrics::mape(as.numeric(train_y_cnn),as.numeric(train_predict_cnn))*100

Metrics::rmse(as.numeric(valid_y_cnn),as.numeric(valid_predict_cnn))
Metrics::mae(as.numeric(valid_y_cnn),as.numeric(valid_predict_cnn))
Metrics::mape(as.numeric(valid_y_cnn),as.numeric(valid_predict_cnn))*100

# Forecasting function for CNN
forecast_next_points_cnn <- function(model, last_data, n_forecast, look_back) {
  forecasted_data <- numeric(n_forecast)
  current_batch <- last_data
  
  for(i in 1:n_forecast) {
    # Ensure current_batch is a matrix
    if (!is.matrix(current_batch)) {
      current_batch <- matrix(current_batch, nrow = look_back, ncol = 1)
    }
    
    # Reshape the input to match the expected shape: (1, look_back, number of features)
    current_batch_reshaped <- array(current_batch, dim = c(1, look_back, 1))
    
    # Predict the next point
    next_prediction <- model %>% predict(current_batch_reshaped)
    
    # Append the prediction (ensure it's a scalar)
    forecasted_data[i] <- next_prediction[1]
    
    # Update the batch to include the new prediction and drop the oldest data point
    current_batch <- rbind(current_batch[-1, , drop = FALSE], next_prediction)
  }
  
  return(inverse_scale_data(forecasted_data))
}

# Prepare the last known data points as input for forecasting
last_train_data_cnn <- tail(train_scaled$Cases, look_back)
last_train_data_cnn <- matrix(last_train_data_cnn, nrow = look_back, ncol = 1)

# Forecast the next 12 points with CNN model
n_forecast_cnn <- 12
forecasted_values_cnn <- forecast_next_points_cnn(model, last_train_data_cnn, n_forecast_cnn, look_back)

print(forecasted_values_cnn)

####################
#                  #
#  Seasonal ARIMA  #
#                  #
####################

data_ts <- ts(data$Cases[1:180], start = c(2008,1), end = c(2022,12), frequency = 12)

ARIMA_Tourism <- auto.arima(data_ts, ic = "aic", test = "adf", seasonal = TRUE, trace = TRUE, stationary = FALSE)
lmtest::coeftest(ARIMA_Tourism)
checkresiduals(ARIMA_Tourism)
accuracy(data_ts,ARIMA_Tourism$fitted)

ARIMA_forecast <- forecast(ARIMA_Tourism, h = 12)
accuracy(data$Cases[181:192],ARIMA_forecast$mean)

ARIMA_forecast_12 <- forecast(ts(data$Cases, start = c(2008,1), end = c(2023,12), frequency = 12),
                              model = ARIMA_Tourism, h = 12)

###############################
#                             #
#  Artificial Neural Network  #
#                             #
###############################

NNAR_Tourism <- nnetar(data_ts)
accuracy(data_ts,NNAR_Tourism$fitted)

NNAR_forecast <- forecast(NNAR_Tourism, h = 12)
accuracy(data$Cases[181:192],NNAR_forecast$mean)

NNAR_forecast_12 <- forecast(ts(data$Cases, start = c(2008,1), end = c(2023,12), frequency = 12), 
                             model = NNAR_Tourism, h = 12)


#Time series plot based on the validation phase
ts.plot(ts(data$Cases[181:192], start = c(2023,1), end = c(2023,12), frequency = 12), 
        valid_predict_gru, valid_predict_lstm, valid_predict_cnn, ARIMA_forecast$mean, NNAR_forecast$mean,
        col = c("black","blue","red","green","orange","purple"), ylim = c(0,600000), lwd = 2)
legend("bottomleft",bty="o",lty=c(1,1),col=c("black","blue","red","green","orange","purple"),
       legend=c("Actual","GRU","LSTM","CNN","SARIMA","NNAR"),cex=0.7,inset=0.025,lwd=2)
