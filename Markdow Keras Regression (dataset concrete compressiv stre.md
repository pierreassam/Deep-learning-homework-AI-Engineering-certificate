
# Neural network with Keras

# Part A: Building the base line model

## A1. Preparing the dataset


```python
# Downloading the data and reading it into a pandas dataframe
import pandas as pd
concrete_data = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv')
concrete_data.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cement</th>
      <th>Blast Furnace Slag</th>
      <th>Fly Ash</th>
      <th>Water</th>
      <th>Superplasticizer</th>
      <th>Coarse Aggregate</th>
      <th>Fine Aggregate</th>
      <th>Age</th>
      <th>Strength</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>540.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>162.0</td>
      <td>2.5</td>
      <td>1040.0</td>
      <td>676.0</td>
      <td>28</td>
      <td>79.99</td>
    </tr>
    <tr>
      <th>1</th>
      <td>540.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>162.0</td>
      <td>2.5</td>
      <td>1055.0</td>
      <td>676.0</td>
      <td>28</td>
      <td>61.89</td>
    </tr>
    <tr>
      <th>2</th>
      <td>332.5</td>
      <td>142.5</td>
      <td>0.0</td>
      <td>228.0</td>
      <td>0.0</td>
      <td>932.0</td>
      <td>594.0</td>
      <td>270</td>
      <td>40.27</td>
    </tr>
    <tr>
      <th>3</th>
      <td>332.5</td>
      <td>142.5</td>
      <td>0.0</td>
      <td>228.0</td>
      <td>0.0</td>
      <td>932.0</td>
      <td>594.0</td>
      <td>365</td>
      <td>41.05</td>
    </tr>
    <tr>
      <th>4</th>
      <td>198.6</td>
      <td>132.4</td>
      <td>0.0</td>
      <td>192.0</td>
      <td>0.0</td>
      <td>978.4</td>
      <td>825.5</td>
      <td>360</td>
      <td>44.30</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Checking how many data point we have
concrete_data.shape
```




    (1030, 9)




```python
# Before building the model, I'm going to check if there ismissing value in the dataset
concrete_data.isnull().sum()
```




    Cement                0
    Blast Furnace Slag    0
    Fly Ash               0
    Water                 0
    Superplasticizer      0
    Coarse Aggregate      0
    Fine Aggregate        0
    Age                   0
    Strength              0
    dtype: int64




```python
# The data seem correct. Let's describe it now
concrete_data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cement</th>
      <th>Blast Furnace Slag</th>
      <th>Fly Ash</th>
      <th>Water</th>
      <th>Superplasticizer</th>
      <th>Coarse Aggregate</th>
      <th>Fine Aggregate</th>
      <th>Age</th>
      <th>Strength</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1030.000000</td>
      <td>1030.000000</td>
      <td>1030.000000</td>
      <td>1030.000000</td>
      <td>1030.000000</td>
      <td>1030.000000</td>
      <td>1030.000000</td>
      <td>1030.000000</td>
      <td>1030.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>281.167864</td>
      <td>73.895825</td>
      <td>54.188350</td>
      <td>181.567282</td>
      <td>6.204660</td>
      <td>972.918932</td>
      <td>773.580485</td>
      <td>45.662136</td>
      <td>35.817961</td>
    </tr>
    <tr>
      <th>std</th>
      <td>104.506364</td>
      <td>86.279342</td>
      <td>63.997004</td>
      <td>21.354219</td>
      <td>5.973841</td>
      <td>77.753954</td>
      <td>80.175980</td>
      <td>63.169912</td>
      <td>16.705742</td>
    </tr>
    <tr>
      <th>min</th>
      <td>102.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>121.800000</td>
      <td>0.000000</td>
      <td>801.000000</td>
      <td>594.000000</td>
      <td>1.000000</td>
      <td>2.330000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>192.375000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>164.900000</td>
      <td>0.000000</td>
      <td>932.000000</td>
      <td>730.950000</td>
      <td>7.000000</td>
      <td>23.710000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>272.900000</td>
      <td>22.000000</td>
      <td>0.000000</td>
      <td>185.000000</td>
      <td>6.400000</td>
      <td>968.000000</td>
      <td>779.500000</td>
      <td>28.000000</td>
      <td>34.445000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>350.000000</td>
      <td>142.950000</td>
      <td>118.300000</td>
      <td>192.000000</td>
      <td>10.200000</td>
      <td>1029.400000</td>
      <td>824.000000</td>
      <td>56.000000</td>
      <td>46.135000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>540.000000</td>
      <td>359.400000</td>
      <td>200.100000</td>
      <td>247.000000</td>
      <td>32.200000</td>
      <td>1145.000000</td>
      <td>992.600000</td>
      <td>365.000000</td>
      <td>82.600000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Now I'm going to split the dataset into predictors and target column
concrete_data_columns = concrete_data.columns

predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']] # all columns except Strength
target = concrete_data['Strength'] # Strength column

```


```python
# Quick check of predictors
predictors.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cement</th>
      <th>Blast Furnace Slag</th>
      <th>Fly Ash</th>
      <th>Water</th>
      <th>Superplasticizer</th>
      <th>Coarse Aggregate</th>
      <th>Fine Aggregate</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>540.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>162.0</td>
      <td>2.5</td>
      <td>1040.0</td>
      <td>676.0</td>
      <td>28</td>
    </tr>
    <tr>
      <th>1</th>
      <td>540.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>162.0</td>
      <td>2.5</td>
      <td>1055.0</td>
      <td>676.0</td>
      <td>28</td>
    </tr>
    <tr>
      <th>2</th>
      <td>332.5</td>
      <td>142.5</td>
      <td>0.0</td>
      <td>228.0</td>
      <td>0.0</td>
      <td>932.0</td>
      <td>594.0</td>
      <td>270</td>
    </tr>
    <tr>
      <th>3</th>
      <td>332.5</td>
      <td>142.5</td>
      <td>0.0</td>
      <td>228.0</td>
      <td>0.0</td>
      <td>932.0</td>
      <td>594.0</td>
      <td>365</td>
    </tr>
    <tr>
      <th>4</th>
      <td>198.6</td>
      <td>132.4</td>
      <td>0.0</td>
      <td>192.0</td>
      <td>0.0</td>
      <td>978.4</td>
      <td>825.5</td>
      <td>360</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Quick check of target
target.head()
```




    0    79.99
    1    61.89
    2    40.27
    3    41.05
    4    44.30
    Name: Strength, dtype: float64




```python
# Let save the numbur of predictor to col since I need this number when building the network
cols = predictors.shape[1] # number of predictors
```


```python
# Importing keras now
import keras
```


```python
# Importing the different package
from keras.models import Sequential
from keras.layers import Dense

```

## A 2. Building the neural network by defining the Keras regression model


```python
# define regression model wtih One hidden layer of 10 nodes, and a ReLU activation function
def regression1_model():
    # create model
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(cols,)))
    model.add(Dense(10, activation='relu'))
    
    model.add(Dense(1))
    
    # compile model using the adam optimizer and the mean squared error as the loss function
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

```

## A 3. Training the model on the training data


```python
# Calling the function to build the model
model1 = regression1_model()

```

    WARNING:tensorflow:From /opt/conda/envs/Python36/lib/python3.6/site-packages/tensorflow/python/framework/op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Colocations handled automatically by placer.



```python
# Training the model using the fit method and leaving out 30% of the dataset for validation and also 50 epochs on training datasets
model1.fit(predictors, target, validation_split=0.3, epochs=50, verbose=2)

```

    WARNING:tensorflow:From /opt/conda/envs/Python36/lib/python3.6/site-packages/tensorflow/python/ops/math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.cast instead.
    Train on 721 samples, validate on 309 samples
    Epoch 1/50
     - 2s - loss: 863.8018 - val_loss: 342.7006
    Epoch 2/50
     - 0s - loss: 315.7981 - val_loss: 186.9044
    Epoch 3/50
     - 0s - loss: 256.8128 - val_loss: 160.4274
    Epoch 4/50
     - 0s - loss: 243.6352 - val_loss: 144.9552
    Epoch 5/50
     - 0s - loss: 235.1872 - val_loss: 138.9744
    Epoch 6/50
     - 0s - loss: 226.0295 - val_loss: 137.0841
    Epoch 7/50
     - 0s - loss: 218.4120 - val_loss: 149.7563
    Epoch 8/50
     - 0s - loss: 208.8240 - val_loss: 142.7386
    Epoch 9/50
     - 0s - loss: 197.4885 - val_loss: 134.5166
    Epoch 10/50
     - 0s - loss: 187.8900 - val_loss: 119.0504
    Epoch 11/50
     - 3s - loss: 180.4801 - val_loss: 116.6253
    Epoch 12/50
     - 0s - loss: 175.7822 - val_loss: 136.8550
    Epoch 13/50
     - 0s - loss: 168.4591 - val_loss: 126.6997
    Epoch 14/50
     - 0s - loss: 162.1084 - val_loss: 126.1610
    Epoch 15/50
     - 0s - loss: 151.2556 - val_loss: 117.1262
    Epoch 16/50
     - 0s - loss: 141.4873 - val_loss: 105.9191
    Epoch 17/50
     - 0s - loss: 138.6390 - val_loss: 102.6335
    Epoch 18/50
     - 0s - loss: 130.9680 - val_loss: 105.9935
    Epoch 19/50
     - 0s - loss: 131.7570 - val_loss: 130.6577
    Epoch 20/50
     - 0s - loss: 126.8413 - val_loss: 102.1175
    Epoch 21/50
     - 0s - loss: 120.7874 - val_loss: 99.0265
    Epoch 22/50
     - 0s - loss: 118.3356 - val_loss: 98.7692
    Epoch 23/50
     - 0s - loss: 116.4269 - val_loss: 94.8122
    Epoch 24/50
     - 0s - loss: 114.2960 - val_loss: 107.4620
    Epoch 25/50
     - 0s - loss: 111.6838 - val_loss: 95.4677
    Epoch 26/50
     - 0s - loss: 107.5051 - val_loss: 90.0978
    Epoch 27/50
     - 0s - loss: 108.2125 - val_loss: 99.3194
    Epoch 28/50
     - 0s - loss: 104.7158 - val_loss: 92.8718
    Epoch 29/50
     - 0s - loss: 103.6438 - val_loss: 94.1274
    Epoch 30/50
     - 0s - loss: 106.3939 - val_loss: 99.2964
    Epoch 31/50
     - 0s - loss: 100.7706 - val_loss: 102.6657
    Epoch 32/50
     - 0s - loss: 102.4352 - val_loss: 100.6604
    Epoch 33/50
     - 0s - loss: 103.8647 - val_loss: 110.0445
    Epoch 34/50
     - 0s - loss: 99.8248 - val_loss: 101.5696
    Epoch 35/50
     - 1s - loss: 98.3419 - val_loss: 94.8927
    Epoch 36/50
     - 0s - loss: 98.9900 - val_loss: 97.8305
    Epoch 37/50
     - 0s - loss: 98.4427 - val_loss: 95.0829
    Epoch 38/50
     - 0s - loss: 99.4225 - val_loss: 108.2746
    Epoch 39/50
     - 0s - loss: 99.5984 - val_loss: 97.5440
    Epoch 40/50
     - 0s - loss: 94.8126 - val_loss: 91.5513
    Epoch 41/50
     - 0s - loss: 94.1101 - val_loss: 93.1939
    Epoch 42/50
     - 0s - loss: 91.4917 - val_loss: 89.7457
    Epoch 43/50
     - 0s - loss: 92.1310 - val_loss: 94.5629
    Epoch 44/50
     - 0s - loss: 90.1284 - val_loss: 87.6071
    Epoch 45/50
     - 0s - loss: 91.0435 - val_loss: 90.5702
    Epoch 46/50
     - 1s - loss: 92.8217 - val_loss: 87.5480
    Epoch 47/50
     - 1s - loss: 86.8968 - val_loss: 87.2570
    Epoch 48/50
     - 0s - loss: 94.5134 - val_loss: 93.3497
    Epoch 49/50
     - 1s - loss: 87.3070 - val_loss: 81.8800
    Epoch 50/50
     - 0s - loss: 83.1198 - val_loss: 78.6419





    <keras.callbacks.History at 0x7f0803e39550>



## A 4. Evaluate the model on the test data


```python
predictions1 = model1.predict(predictors)
# Computig the mean squared error between the predicted concrete strength and the actual concrete strength
from sklearn.metrics import mean_squared_error
y_pred=predictions1
y_true=target

mean_squared_error (y_pred,y_true)
```




    80.08026939414025



## A 5. Creating a list of 50 mean squared error


```python

```

# Part B: Normalize the data


```python
# Normalizing data
predictors_norm = (predictors - predictors.mean()) / predictors.std()
predictors_norm.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cement</th>
      <th>Blast Furnace Slag</th>
      <th>Fly Ash</th>
      <th>Water</th>
      <th>Superplasticizer</th>
      <th>Coarse Aggregate</th>
      <th>Fine Aggregate</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.476712</td>
      <td>-0.856472</td>
      <td>-0.846733</td>
      <td>-0.916319</td>
      <td>-0.620147</td>
      <td>0.862735</td>
      <td>-1.217079</td>
      <td>-0.279597</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.476712</td>
      <td>-0.856472</td>
      <td>-0.846733</td>
      <td>-0.916319</td>
      <td>-0.620147</td>
      <td>1.055651</td>
      <td>-1.217079</td>
      <td>-0.279597</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.491187</td>
      <td>0.795140</td>
      <td>-0.846733</td>
      <td>2.174405</td>
      <td>-1.038638</td>
      <td>-0.526262</td>
      <td>-2.239829</td>
      <td>3.551340</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.491187</td>
      <td>0.795140</td>
      <td>-0.846733</td>
      <td>2.174405</td>
      <td>-1.038638</td>
      <td>-0.526262</td>
      <td>-2.239829</td>
      <td>5.055221</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.790075</td>
      <td>0.678079</td>
      <td>-0.846733</td>
      <td>0.488555</td>
      <td>-1.038638</td>
      <td>0.070492</td>
      <td>0.647569</td>
      <td>4.976069</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Saving the predictor_norm to n_cols beacause I need this number to build the nnetwork
n_cols = predictors_norm.shape[1] # number of predictors
```


```python
# Importing keras now
import keras
# Importing the different package
from keras.models import Sequential
from keras.layers import Dense

```

## B 1. Building the neural network by defining the Keras regression model


```python
# define regression model wtih One hidden layer of 10 nodes, and a ReLU activation function
def regression1_model():
    # create model
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(10, activation='relu'))
    
    model.add(Dense(1))
    
    # compile model using the adam optimizer and the mean squared error as the loss function
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

```

## B 2. Training the model on the training data


```python
# Calling the function to build the model
model1 = regression1_model()
```


```python
# Training the model using the fit method and leaving out 30% of the dataset for validation and also 50 epochs on training datasets
model1.fit(predictors_norm, target, validation_split=0.3, epochs=50, verbose=2)

```

    Train on 721 samples, validate on 309 samples
    Epoch 1/50
     - 1s - loss: 1699.6910 - val_loss: 1215.0307
    Epoch 2/50
     - 0s - loss: 1671.7877 - val_loss: 1195.9047
    Epoch 3/50
     - 0s - loss: 1645.4113 - val_loss: 1176.4683
    Epoch 4/50
     - 0s - loss: 1616.9730 - val_loss: 1154.6039
    Epoch 5/50
     - 0s - loss: 1582.4447 - val_loss: 1127.5991
    Epoch 6/50
     - 0s - loss: 1538.6431 - val_loss: 1092.7531
    Epoch 7/50
     - 0s - loss: 1482.9571 - val_loss: 1050.9356
    Epoch 8/50
     - 0s - loss: 1415.4996 - val_loss: 1001.8340
    Epoch 9/50
     - 0s - loss: 1336.6514 - val_loss: 946.4881
    Epoch 10/50
     - 0s - loss: 1247.2759 - val_loss: 884.7598
    Epoch 11/50
     - 0s - loss: 1149.0562 - val_loss: 815.7310
    Epoch 12/50
     - 0s - loss: 1041.1306 - val_loss: 745.6932
    Epoch 13/50
     - 0s - loss: 931.3313 - val_loss: 672.8063
    Epoch 14/50
     - 0s - loss: 820.2856 - val_loss: 597.0477
    Epoch 15/50
     - 0s - loss: 711.6269 - val_loss: 523.0540
    Epoch 16/50
     - 0s - loss: 609.1373 - val_loss: 452.3304
    Epoch 17/50
     - 0s - loss: 520.5462 - val_loss: 390.7389
    Epoch 18/50
     - 0s - loss: 449.0113 - val_loss: 340.9855
    Epoch 19/50
     - 2s - loss: 396.3668 - val_loss: 300.6067
    Epoch 20/50
     - 0s - loss: 357.9903 - val_loss: 270.2475
    Epoch 21/50
     - 1s - loss: 330.2528 - val_loss: 246.0147
    Epoch 22/50
     - 0s - loss: 308.6790 - val_loss: 230.7004
    Epoch 23/50
     - 0s - loss: 291.6926 - val_loss: 215.7417
    Epoch 24/50
     - 0s - loss: 278.2451 - val_loss: 202.5470
    Epoch 25/50
     - 0s - loss: 265.3096 - val_loss: 195.9090
    Epoch 26/50
     - 0s - loss: 254.2434 - val_loss: 188.7218
    Epoch 27/50
     - 1s - loss: 243.8033 - val_loss: 180.9928
    Epoch 28/50
     - 0s - loss: 235.5240 - val_loss: 174.6949
    Epoch 29/50
     - 0s - loss: 227.2358 - val_loss: 170.8528
    Epoch 30/50
     - 0s - loss: 220.2542 - val_loss: 168.0294
    Epoch 31/50
     - 0s - loss: 213.7392 - val_loss: 163.0179
    Epoch 32/50
     - 0s - loss: 207.8543 - val_loss: 160.1526
    Epoch 33/50
     - 0s - loss: 202.5391 - val_loss: 158.6256
    Epoch 34/50
     - 0s - loss: 197.5842 - val_loss: 155.4137
    Epoch 35/50
     - 0s - loss: 193.1123 - val_loss: 153.5901
    Epoch 36/50
     - 0s - loss: 188.7810 - val_loss: 151.0797
    Epoch 37/50
     - 0s - loss: 185.1288 - val_loss: 149.7661
    Epoch 38/50
     - 0s - loss: 181.2504 - val_loss: 148.0706
    Epoch 39/50
     - 0s - loss: 177.8612 - val_loss: 146.3155
    Epoch 40/50
     - 0s - loss: 174.5860 - val_loss: 145.7542
    Epoch 41/50
     - 0s - loss: 171.6871 - val_loss: 144.5331
    Epoch 42/50
     - 0s - loss: 168.8050 - val_loss: 143.0742
    Epoch 43/50
     - 0s - loss: 165.8972 - val_loss: 141.9157
    Epoch 44/50
     - 0s - loss: 163.3845 - val_loss: 139.9473
    Epoch 45/50
     - 0s - loss: 160.9382 - val_loss: 139.3525
    Epoch 46/50
     - 0s - loss: 158.3126 - val_loss: 138.0473
    Epoch 47/50
     - 0s - loss: 156.2439 - val_loss: 136.4373
    Epoch 48/50
     - 0s - loss: 153.9597 - val_loss: 135.5348
    Epoch 49/50
     - 0s - loss: 151.6879 - val_loss: 134.6780
    Epoch 50/50
     - 0s - loss: 149.6664 - val_loss: 132.7608





    <keras.callbacks.History at 0x7f072016c0b8>



## B 3. Evaluate the model on the test data


```python
predictions1 = model1.predict(predictors_norm)
# Computig the mean squared error between the predicted concrete strength and the actual concrete strength
from sklearn.metrics import mean_squared_error
y_pred1=predictions1
y_true=target

mean_squared_error (y_pred1,y_true)
```




    143.54516640284422



## B 4. Comparing the two mean_squared_error

The mean_squared_error of the part A (80.08026939414025) <The mean_squared_error of the part B (143.54516640284422)

# Part C Increasing the number

## C 1. Increasing the number of epochs and reapeting the part B


```python
# Training the model using the fit method and leaving out 30% of the dataset for validation and also 100 epochs on training datasets
model1.fit(predictors_norm, target, validation_split=0.3, epochs=100, verbose=2)

```

    Train on 721 samples, validate on 309 samples
    Epoch 1/100
     - 2s - loss: 147.3497 - val_loss: 131.7275
    Epoch 2/100
     - 0s - loss: 145.5052 - val_loss: 131.0325
    Epoch 3/100
     - 0s - loss: 143.5461 - val_loss: 129.2707
    Epoch 4/100
     - 0s - loss: 141.3557 - val_loss: 128.9313
    Epoch 5/100
     - 0s - loss: 139.4003 - val_loss: 127.2130
    Epoch 6/100
     - 0s - loss: 137.1795 - val_loss: 126.1195
    Epoch 7/100
     - 0s - loss: 135.2382 - val_loss: 124.4518
    Epoch 8/100
     - 0s - loss: 133.2513 - val_loss: 123.7796
    Epoch 9/100
     - 0s - loss: 131.3099 - val_loss: 122.5841
    Epoch 10/100
     - 0s - loss: 129.4205 - val_loss: 121.0230
    Epoch 11/100
     - 0s - loss: 127.3867 - val_loss: 119.7178
    Epoch 12/100
     - 0s - loss: 125.4714 - val_loss: 118.5189
    Epoch 13/100
     - 1s - loss: 123.4905 - val_loss: 116.6655
    Epoch 14/100
     - 0s - loss: 121.7604 - val_loss: 116.0380
    Epoch 15/100
     - 0s - loss: 120.2061 - val_loss: 115.2875
    Epoch 16/100
     - 0s - loss: 118.6805 - val_loss: 113.6136
    Epoch 17/100
     - 0s - loss: 117.3396 - val_loss: 112.2528
    Epoch 18/100
     - 0s - loss: 115.6427 - val_loss: 112.2350
    Epoch 19/100
     - 0s - loss: 114.2596 - val_loss: 110.4961
    Epoch 20/100
     - 0s - loss: 113.1830 - val_loss: 110.3155
    Epoch 21/100
     - 0s - loss: 111.6165 - val_loss: 109.0482
    Epoch 22/100
     - 0s - loss: 110.5269 - val_loss: 108.9579
    Epoch 23/100
     - 0s - loss: 109.4617 - val_loss: 108.1607
    Epoch 24/100
     - 0s - loss: 108.4649 - val_loss: 107.5693
    Epoch 25/100
     - 0s - loss: 107.2510 - val_loss: 106.5674
    Epoch 26/100
     - 0s - loss: 106.2588 - val_loss: 107.0651
    Epoch 27/100
     - 0s - loss: 105.1938 - val_loss: 104.7122
    Epoch 28/100
     - 0s - loss: 104.4472 - val_loss: 104.8697
    Epoch 29/100
     - 0s - loss: 103.4560 - val_loss: 104.8465
    Epoch 30/100
     - 0s - loss: 102.6099 - val_loss: 103.8373
    Epoch 31/100
     - 0s - loss: 101.7241 - val_loss: 103.4946
    Epoch 32/100
     - 0s - loss: 101.1051 - val_loss: 103.9542
    Epoch 33/100
     - 0s - loss: 99.9744 - val_loss: 102.2479
    Epoch 34/100
     - 0s - loss: 99.3692 - val_loss: 101.3041
    Epoch 35/100
     - 0s - loss: 98.6711 - val_loss: 102.1624
    Epoch 36/100
     - 2s - loss: 97.7744 - val_loss: 102.0636
    Epoch 37/100
     - 0s - loss: 97.2954 - val_loss: 101.6737
    Epoch 38/100
     - 1s - loss: 96.4546 - val_loss: 100.7281
    Epoch 39/100
     - 0s - loss: 95.8726 - val_loss: 101.2278
    Epoch 40/100
     - 0s - loss: 95.2298 - val_loss: 99.7972
    Epoch 41/100
     - 0s - loss: 94.4605 - val_loss: 99.1519
    Epoch 42/100
     - 0s - loss: 93.7638 - val_loss: 98.2433
    Epoch 43/100
     - 0s - loss: 93.1918 - val_loss: 99.1384
    Epoch 44/100
     - 0s - loss: 92.5107 - val_loss: 98.1176
    Epoch 45/100
     - 0s - loss: 91.7020 - val_loss: 98.1051
    Epoch 46/100
     - 0s - loss: 91.2433 - val_loss: 96.9676
    Epoch 47/100
     - 0s - loss: 90.4729 - val_loss: 97.8481
    Epoch 48/100
     - 0s - loss: 89.9902 - val_loss: 97.0744
    Epoch 49/100
     - 0s - loss: 89.4173 - val_loss: 95.9011
    Epoch 50/100
     - 0s - loss: 88.6406 - val_loss: 96.1865
    Epoch 51/100
     - 0s - loss: 88.0024 - val_loss: 95.3637
    Epoch 52/100
     - 1s - loss: 87.1416 - val_loss: 96.1308
    Epoch 53/100
     - 0s - loss: 86.6308 - val_loss: 94.7279
    Epoch 54/100
     - 0s - loss: 85.8880 - val_loss: 95.4443
    Epoch 55/100
     - 0s - loss: 85.3612 - val_loss: 94.4521
    Epoch 56/100
     - 0s - loss: 84.4011 - val_loss: 95.0679
    Epoch 57/100
     - 0s - loss: 83.8769 - val_loss: 94.0206
    Epoch 58/100
     - 0s - loss: 83.2817 - val_loss: 94.0387
    Epoch 59/100
     - 0s - loss: 82.6556 - val_loss: 93.8279
    Epoch 60/100
     - 0s - loss: 82.3193 - val_loss: 93.9198
    Epoch 61/100
     - 0s - loss: 81.3772 - val_loss: 93.7808
    Epoch 62/100
     - 0s - loss: 80.8805 - val_loss: 93.0831
    Epoch 63/100
     - 0s - loss: 80.3497 - val_loss: 92.7216
    Epoch 64/100
     - 0s - loss: 79.6202 - val_loss: 92.6201
    Epoch 65/100
     - 0s - loss: 78.8934 - val_loss: 92.3490
    Epoch 66/100
     - 0s - loss: 78.3138 - val_loss: 91.9029
    Epoch 67/100
     - 0s - loss: 77.6135 - val_loss: 92.3501
    Epoch 68/100
     - 0s - loss: 76.8788 - val_loss: 92.8432
    Epoch 69/100
     - 0s - loss: 76.2279 - val_loss: 91.8356
    Epoch 70/100
     - 0s - loss: 75.6323 - val_loss: 90.7335
    Epoch 71/100
     - 0s - loss: 74.9714 - val_loss: 91.5295
    Epoch 72/100
     - 0s - loss: 74.2915 - val_loss: 90.9397
    Epoch 73/100
     - 0s - loss: 73.8483 - val_loss: 91.8539
    Epoch 74/100
     - 2s - loss: 72.9288 - val_loss: 90.8012
    Epoch 75/100
     - 0s - loss: 72.0524 - val_loss: 89.9499
    Epoch 76/100
     - 1s - loss: 71.3248 - val_loss: 89.9601
    Epoch 77/100
     - 0s - loss: 70.2836 - val_loss: 91.6327
    Epoch 78/100
     - 0s - loss: 69.2785 - val_loss: 91.3696
    Epoch 79/100
     - 0s - loss: 68.5587 - val_loss: 91.1502
    Epoch 80/100
     - 0s - loss: 67.0429 - val_loss: 90.1940
    Epoch 81/100
     - 0s - loss: 66.1452 - val_loss: 90.0346
    Epoch 82/100
     - 0s - loss: 65.2290 - val_loss: 90.4707
    Epoch 83/100
     - 0s - loss: 64.3251 - val_loss: 89.1638
    Epoch 84/100
     - 0s - loss: 63.6067 - val_loss: 89.4472
    Epoch 85/100
     - 0s - loss: 62.9412 - val_loss: 89.1935
    Epoch 86/100
     - 0s - loss: 62.1505 - val_loss: 88.1300
    Epoch 87/100
     - 0s - loss: 61.5142 - val_loss: 88.8480
    Epoch 88/100
     - 0s - loss: 61.0464 - val_loss: 89.5373
    Epoch 89/100
     - 0s - loss: 60.5484 - val_loss: 89.9625
    Epoch 90/100
     - 1s - loss: 59.7063 - val_loss: 87.8136
    Epoch 91/100
     - 0s - loss: 59.3689 - val_loss: 88.7352
    Epoch 92/100
     - 0s - loss: 58.8650 - val_loss: 88.8268
    Epoch 93/100
     - 0s - loss: 58.2955 - val_loss: 89.5571
    Epoch 94/100
     - 0s - loss: 57.7990 - val_loss: 88.1872
    Epoch 95/100
     - 0s - loss: 57.3082 - val_loss: 89.8960
    Epoch 96/100
     - 0s - loss: 56.9098 - val_loss: 88.3206
    Epoch 97/100
     - 0s - loss: 56.4289 - val_loss: 88.3993
    Epoch 98/100
     - 0s - loss: 56.1644 - val_loss: 88.4108
    Epoch 99/100
     - 0s - loss: 55.5476 - val_loss: 87.9299
    Epoch 100/100
     - 0s - loss: 54.9845 - val_loss: 88.1701





    <keras.callbacks.History at 0x7f0700102a58>



## C2. Evaluating the model on the test data


```python
predictions1 = model1.predict(predictors_norm)
# Computig the mean squared error between the predicted concrete strength and the actual concrete strength
from sklearn.metrics import mean_squared_error
y_pred1=predictions1
y_true=target

mean_squared_error (y_pred1,y_true)
```




    64.5926180229111



## C 3. Comparing the two mean_squared_error of step B and C

The mean_squared_error of the part C (64.5926180229111) < The mean_squared_error of the part B (143.54516640284422)

# Part D: Increasing the number of hidden layers of part B


```python
# define regression model wtih three hidden layer of 10 nodes, and a ReLU activation function
def regression3_model():
    # create model
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    
    # compile model using the adam optimizer and the mean squared error as the loss function
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

```

## D 1. Training the model 


```python
# Calling the function to build the model
model3 = regression3_model()
```


```python
# Training the model using the fit method and leaving out 30% of the dataset for validation and also 50 epochs on training datasets
model3.fit(predictors_norm, target, validation_split=0.3, epochs=50, verbose=2)

```

    Train on 721 samples, validate on 309 samples
    Epoch 1/50
     - 2s - loss: 1688.5873 - val_loss: 1211.4673
    Epoch 2/50
     - 0s - loss: 1664.9830 - val_loss: 1184.5104
    Epoch 3/50
     - 1s - loss: 1623.2799 - val_loss: 1133.4396
    Epoch 4/50
     - 0s - loss: 1533.4877 - val_loss: 1023.5989
    Epoch 5/50
     - 0s - loss: 1346.6074 - val_loss: 827.0700
    Epoch 6/50
     - 0s - loss: 1040.8350 - val_loss: 563.9414
    Epoch 7/50
     - 1s - loss: 677.1364 - val_loss: 349.3658
    Epoch 8/50
     - 0s - loss: 423.0256 - val_loss: 286.2074
    Epoch 9/50
     - 1s - loss: 342.5540 - val_loss: 265.4976
    Epoch 10/50
     - 0s - loss: 295.2712 - val_loss: 241.8337
    Epoch 11/50
     - 0s - loss: 266.4507 - val_loss: 227.1864
    Epoch 12/50
     - 1s - loss: 247.6141 - val_loss: 216.5103
    Epoch 13/50
     - 0s - loss: 235.0680 - val_loss: 208.8693
    Epoch 14/50
     - 0s - loss: 224.2156 - val_loss: 204.2457
    Epoch 15/50
     - 0s - loss: 215.9390 - val_loss: 203.4396
    Epoch 16/50
     - 0s - loss: 209.0945 - val_loss: 196.7599
    Epoch 17/50
     - 1s - loss: 203.2392 - val_loss: 197.5266
    Epoch 18/50
     - 2s - loss: 198.3137 - val_loss: 192.1309
    Epoch 19/50
     - 2s - loss: 194.2567 - val_loss: 190.2858
    Epoch 20/50
     - 1s - loss: 190.3807 - val_loss: 190.4194
    Epoch 21/50
     - 0s - loss: 187.4988 - val_loss: 189.1768
    Epoch 22/50
     - 0s - loss: 185.0324 - val_loss: 185.1790
    Epoch 23/50
     - 0s - loss: 182.5868 - val_loss: 184.3445
    Epoch 24/50
     - 0s - loss: 180.6220 - val_loss: 183.9429
    Epoch 25/50
     - 0s - loss: 177.2231 - val_loss: 178.8344
    Epoch 26/50
     - 1s - loss: 174.8116 - val_loss: 180.6030
    Epoch 27/50
     - 0s - loss: 173.8654 - val_loss: 174.1353
    Epoch 28/50
     - 0s - loss: 170.8584 - val_loss: 176.2234
    Epoch 29/50
     - 0s - loss: 169.1087 - val_loss: 172.1981
    Epoch 30/50
     - 0s - loss: 167.5677 - val_loss: 170.1972
    Epoch 31/50
     - 1s - loss: 165.9024 - val_loss: 172.1044
    Epoch 32/50
     - 0s - loss: 164.3232 - val_loss: 168.7552
    Epoch 33/50
     - 0s - loss: 162.6112 - val_loss: 170.0309
    Epoch 34/50
     - 0s - loss: 161.3478 - val_loss: 169.6678
    Epoch 35/50
     - 0s - loss: 159.8551 - val_loss: 164.0566
    Epoch 36/50
     - 1s - loss: 158.2030 - val_loss: 164.5459
    Epoch 37/50
     - 0s - loss: 157.2483 - val_loss: 160.0802
    Epoch 38/50
     - 2s - loss: 155.3552 - val_loss: 156.3266
    Epoch 39/50
     - 1s - loss: 153.4369 - val_loss: 163.8925
    Epoch 40/50
     - 0s - loss: 152.5118 - val_loss: 155.6555
    Epoch 41/50
     - 0s - loss: 150.7805 - val_loss: 157.4896
    Epoch 42/50
     - 0s - loss: 149.5548 - val_loss: 156.3395
    Epoch 43/50
     - 0s - loss: 148.7437 - val_loss: 155.4986
    Epoch 44/50
     - 0s - loss: 147.0307 - val_loss: 155.7093
    Epoch 45/50
     - 0s - loss: 147.2138 - val_loss: 155.1043
    Epoch 46/50
     - 0s - loss: 144.4073 - val_loss: 154.0955
    Epoch 47/50
     - 0s - loss: 143.4255 - val_loss: 149.7669
    Epoch 48/50
     - 0s - loss: 142.5302 - val_loss: 154.6385
    Epoch 49/50
     - 0s - loss: 141.7875 - val_loss: 149.5502
    Epoch 50/50
     - 1s - loss: 140.6159 - val_loss: 147.5203





    <keras.callbacks.History at 0x7f070005ed68>



## D 2. Evaluate the model on the test data


```python
predictions3 = model3.predict(predictors_norm)
# Computig the mean squared error between the predicted concrete strength and the actual concrete strength
from sklearn.metrics import mean_squared_error
y_pred3=predictions3
y_true=target

mean_squared_error (y_pred3,y_true)
```




    141.5566747400289



## D 3. Comparing the two mean_squared_error of step B and D

The mean_squared_error of the part D (141.5566747400289) < The mean_squared_error of the part B (143.54516640284422)
