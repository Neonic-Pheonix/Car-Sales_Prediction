import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow.keras

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense

# Input variables
# Bit   0 = male, 1 = female
# Float between 18 to 100
# Float between 10,000 and 100,000
# Float between 10,000 and 100,000
# Float between 100,000 and 1,000,000

gender = 0                      
age = 41.8                      
annual_salary = 62812.09        
credit_card_debt = 11609.38     
net_worth = 238961.25           

# Data import for training network
data = pd.read_csv("car_sales_dataset.txt",encoding='ISO-8859-1')
print(data)

# Plot data and show in window 
sns.pairplot(data)
plt.show

# Selecting data inputs removing unnecessary datapoints and print in console input shape
inputs = data.drop(['Customer_Name', 'Customer_Email', 'Country', 'Purchase_Amount'], axis = 1)
print(inputs)
print("Input data Shape=",inputs.shape)

# Selecting data output as purchase amount, reshaping and printing in console
output = data['Purchase_Amount']
print(output)
output = output.values.reshape(-1,1)
print("Output Data Shape=",output.shape)

# Scale data inputs 
scaler_in = MinMaxScaler()
input_scaled = scaler_in.fit_transform(inputs)
print(input_scaled)

# Scale data outputs
scaler_out = MinMaxScaler()
output_scaled = scaler_out.fit_transform(output)
print(output_scaled)

# Create model
model = Sequential() 
model.add(Dense(25, input_dim=5, activation='relu')) # hidden layer with 25 neurons connected to the input layer with Rectified Linear Unit activation
model.add(Dense(25, activation='relu')) # hidden layer with 25 neurons
model.add(Dense(1, activation='linear')) # output layer with 1 neuron (purchase amount unscaled)
print(model.summary())

# Train model with dataset optimising and ploting dictionary keys
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
epochs_hist = model.fit(input_scaled, output_scaled, epochs=20, batch_size=10, verbose=1, validation_split=0.2)
print(epochs_hist.history.keys())

# Plotting and Printing the training for the model
plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])

plt.title('Model Loss Progression During Training/Validation')
plt.ylabel('Training and Validation Losses')
plt.xlabel('Epoch Number')
plt.legend(['Training Loss', 'Validation Loss'])
plt.show()


input_test_sample = np.array([[gender, age,  annual_salary, credit_card_debt, net_worth]])
input_test_sample_scaled = scaler_in.transform(input_test_sample)

output_predict_sample_scaled = model.predict(input_test_sample_scaled)
print('Predicted Output (Scaled) =', output_predict_sample_scaled)

output_predict_sample = scaler_out.inverse_transform(output_predict_sample_scaled)
print('Predicted Output / Purchase Amount ', output_predict_sample)
