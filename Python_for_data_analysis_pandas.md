## PANDAS

* Pandas is an open source library built on top of numpy. 
* It allows fast ananlysis of data. It can work with data wide variety of sources.
* It also has built-in visualization features.


```python
# import libraries

import numpy as np
import pandas as pd
```

### Series is same as Numpy array but we can label the index

* Create series using array
* Create series using list with default index
* Create series using dictionary
* Series can contain data with any datatype


```python
# create an array
my_arr = np.array([1, 2, 3, 4])
my_arr
```




    array([1, 2, 3, 4])




```python
# create an array of labels
my_labels = np.array(['a', 'b', 'c', 'd'])
my_labels
```




    array(['a', 'b', 'c', 'd'], dtype='<U1')




```python
# create series
my_series = pd.Series(my_arr, my_labels)
my_series
```




    a    1
    b    2
    c    3
    d    4
    dtype: int32




```python
# default index will be 0, 1, 2.... and we can also pass a list
pd.Series([1, 2, 3])
```




    0    1
    1    2
    2    3
    dtype: int64




```python
# create a dictionary
my_dict = {'p': 10, 'q': 20, 'r': 30}
my_dict
```




    {'p': 10, 'q': 20, 'r': 30}




```python
# create series using dictionary - keys will be index
pd.Series(my_dict)
```




    p    10
    q    20
    r    30
    dtype: int64




```python
# series can hold any data type
pd.Series([1, 'c', sum])
```




    0                          1
    1                          c
    2    <built-in function sum>
    dtype: object



### Adding two series

* If common index is found then values will be added otherwise nan is displayed


```python
# create series1
series1 = pd.Series([1, 2, 3, 4], ['A', 'B', 'C', 'D'])
series1
```




    A    1
    B    2
    C    3
    D    4
    dtype: int64




```python
# create series2
series2 = pd.Series([1, 2, 4, 5], ['A', 'B', 'D', 'E'])
series2
```




    A    1
    B    2
    D    4
    E    5
    dtype: int64




```python
# add
series1 + series2 # index C and E are not common to both series and therefore nan
```




    A    2.0
    B    4.0
    C    NaN
    D    8.0
    E    NaN
    dtype: float64



### DataFrame is a bunch of series that share same index

* Create DataFrame
* Add new column
* Drop new column
* DataFrame shape
* Access rows, columns, elements and subset of DataFrame



```python
# import randn
from numpy.random import randn

# to get same random number every time based on seed value
np.random.seed(101) 
```


```python
# create dataframe
df = pd.DataFrame(randn(5, 4), ['A', 'B', 'C', 'D', 'E'], ['W', 'X', 'Y', 'Z']) # pd.DataFrame(data, rows, cols)
df
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
      <th>W</th>
      <th>X</th>
      <th>Y</th>
      <th>Z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>2.706850</td>
      <td>0.628133</td>
      <td>0.907969</td>
      <td>0.503826</td>
    </tr>
    <tr>
      <th>B</th>
      <td>0.651118</td>
      <td>-0.319318</td>
      <td>-0.848077</td>
      <td>0.605965</td>
    </tr>
    <tr>
      <th>C</th>
      <td>-2.018168</td>
      <td>0.740122</td>
      <td>0.528813</td>
      <td>-0.589001</td>
    </tr>
    <tr>
      <th>D</th>
      <td>0.188695</td>
      <td>-0.758872</td>
      <td>-0.933237</td>
      <td>0.955057</td>
    </tr>
    <tr>
      <th>E</th>
      <td>0.190794</td>
      <td>1.978757</td>
      <td>2.605967</td>
      <td>0.683509</td>
    </tr>
  </tbody>
</table>
</div>




```python
# what does a dataFrame contain - a Series
type(df['W'])
```




    pandas.core.series.Series




```python
# add new column to existing df
df['new'] = df['W'] + df['X']
df
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
      <th>W</th>
      <th>X</th>
      <th>Y</th>
      <th>Z</th>
      <th>new</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>2.706850</td>
      <td>0.628133</td>
      <td>0.907969</td>
      <td>0.503826</td>
      <td>3.334983</td>
    </tr>
    <tr>
      <th>B</th>
      <td>0.651118</td>
      <td>-0.319318</td>
      <td>-0.848077</td>
      <td>0.605965</td>
      <td>0.331800</td>
    </tr>
    <tr>
      <th>C</th>
      <td>-2.018168</td>
      <td>0.740122</td>
      <td>0.528813</td>
      <td>-0.589001</td>
      <td>-1.278046</td>
    </tr>
    <tr>
      <th>D</th>
      <td>0.188695</td>
      <td>-0.758872</td>
      <td>-0.933237</td>
      <td>0.955057</td>
      <td>-0.570177</td>
    </tr>
    <tr>
      <th>E</th>
      <td>0.190794</td>
      <td>1.978757</td>
      <td>2.605967</td>
      <td>0.683509</td>
      <td>2.169552</td>
    </tr>
  </tbody>
</table>
</div>




```python
# drop existing column
df.drop('new', axis = 1) # by default axis=0, that checks for rows. If we want to give col names then axis = 1
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
      <th>W</th>
      <th>X</th>
      <th>Y</th>
      <th>Z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>2.706850</td>
      <td>0.628133</td>
      <td>0.907969</td>
      <td>0.503826</td>
    </tr>
    <tr>
      <th>B</th>
      <td>0.651118</td>
      <td>-0.319318</td>
      <td>-0.848077</td>
      <td>0.605965</td>
    </tr>
    <tr>
      <th>C</th>
      <td>-2.018168</td>
      <td>0.740122</td>
      <td>0.528813</td>
      <td>-0.589001</td>
    </tr>
    <tr>
      <th>D</th>
      <td>0.188695</td>
      <td>-0.758872</td>
      <td>-0.933237</td>
      <td>0.955057</td>
    </tr>
    <tr>
      <th>E</th>
      <td>0.190794</td>
      <td>1.978757</td>
      <td>2.605967</td>
      <td>0.683509</td>
    </tr>
  </tbody>
</table>
</div>




```python
# but 'new' column is still existing in df after dropping
df
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
      <th>W</th>
      <th>X</th>
      <th>Y</th>
      <th>Z</th>
      <th>new</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>2.706850</td>
      <td>0.628133</td>
      <td>0.907969</td>
      <td>0.503826</td>
      <td>3.334983</td>
    </tr>
    <tr>
      <th>B</th>
      <td>0.651118</td>
      <td>-0.319318</td>
      <td>-0.848077</td>
      <td>0.605965</td>
      <td>0.331800</td>
    </tr>
    <tr>
      <th>C</th>
      <td>-2.018168</td>
      <td>0.740122</td>
      <td>0.528813</td>
      <td>-0.589001</td>
      <td>-1.278046</td>
    </tr>
    <tr>
      <th>D</th>
      <td>0.188695</td>
      <td>-0.758872</td>
      <td>-0.933237</td>
      <td>0.955057</td>
      <td>-0.570177</td>
    </tr>
    <tr>
      <th>E</th>
      <td>0.190794</td>
      <td>1.978757</td>
      <td>2.605967</td>
      <td>0.683509</td>
      <td>2.169552</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Pandas make sure that we do not lose any data accidentally. If we want to drop any columns/rows from a df then we've to provide inplace=true.
df.drop('new', axis=1, inplace=True)
df
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
      <th>W</th>
      <th>X</th>
      <th>Y</th>
      <th>Z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>2.706850</td>
      <td>0.628133</td>
      <td>0.907969</td>
      <td>0.503826</td>
    </tr>
    <tr>
      <th>B</th>
      <td>0.651118</td>
      <td>-0.319318</td>
      <td>-0.848077</td>
      <td>0.605965</td>
    </tr>
    <tr>
      <th>C</th>
      <td>-2.018168</td>
      <td>0.740122</td>
      <td>0.528813</td>
      <td>-0.589001</td>
    </tr>
    <tr>
      <th>D</th>
      <td>0.188695</td>
      <td>-0.758872</td>
      <td>-0.933237</td>
      <td>0.955057</td>
    </tr>
    <tr>
      <th>E</th>
      <td>0.190794</td>
      <td>1.978757</td>
      <td>2.605967</td>
      <td>0.683509</td>
    </tr>
  </tbody>
</table>
</div>




```python
# shape of df
df.shape # rows x columns
```




    (5, 4)




```python
# access df columns
df['W']
```




    A    2.706850
    B    0.651118
    C   -2.018168
    D    0.188695
    E    0.190794
    Name: W, dtype: float64




```python
# access df rows
df.loc['B']
```




    W    0.651118
    X   -0.319318
    Y   -0.848077
    Z    0.605965
    Name: B, dtype: float64




```python
# access df rows by its position
df.iloc[1]
```




    W    0.651118
    X   -0.319318
    Y   -0.848077
    Z    0.605965
    Name: B, dtype: float64




```python
# get elements of df (col, row)
df['W']['A']
```




    2.706849839399938




```python
# get elements of df (row, col)
df.loc['A']['W']
```




    2.706849839399938




```python
# subset of df
df.loc[['A', 'B'], ['X', 'Y']]
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
      <th>X</th>
      <th>Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>0.628133</td>
      <td>0.907969</td>
    </tr>
    <tr>
      <th>B</th>
      <td>-0.319318</td>
      <td>-0.848077</td>
    </tr>
  </tbody>
</table>
</div>



### DataFrame comparison

* Comparing df with a scalar creates a boolean df
* Passing the boolean df to original df filters only required elements (others: nan)
* Can be compared by particular columns


```python
# Dataframe
df
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
      <th>W</th>
      <th>X</th>
      <th>Y</th>
      <th>Z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>2.706850</td>
      <td>0.628133</td>
      <td>0.907969</td>
      <td>0.503826</td>
    </tr>
    <tr>
      <th>B</th>
      <td>0.651118</td>
      <td>-0.319318</td>
      <td>-0.848077</td>
      <td>0.605965</td>
    </tr>
    <tr>
      <th>C</th>
      <td>-2.018168</td>
      <td>0.740122</td>
      <td>0.528813</td>
      <td>-0.589001</td>
    </tr>
    <tr>
      <th>D</th>
      <td>0.188695</td>
      <td>-0.758872</td>
      <td>-0.933237</td>
      <td>0.955057</td>
    </tr>
    <tr>
      <th>E</th>
      <td>0.190794</td>
      <td>1.978757</td>
      <td>2.605967</td>
      <td>0.683509</td>
    </tr>
  </tbody>
</table>
</div>




```python
# create a boolean df that staisfies the condition df elements greater than zero
boolean_df = df > 0
boolean_df
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
      <th>W</th>
      <th>X</th>
      <th>Y</th>
      <th>Z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>B</th>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>C</th>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>D</th>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>E</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
# get elements greater than zero - boolean false = nan
df_greater_0 = df[boolean_df]
df_greater_0
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
      <th>W</th>
      <th>X</th>
      <th>Y</th>
      <th>Z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>2.706850</td>
      <td>0.628133</td>
      <td>0.907969</td>
      <td>0.503826</td>
    </tr>
    <tr>
      <th>B</th>
      <td>0.651118</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.605965</td>
    </tr>
    <tr>
      <th>C</th>
      <td>NaN</td>
      <td>0.740122</td>
      <td>0.528813</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>D</th>
      <td>0.188695</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.955057</td>
    </tr>
    <tr>
      <th>E</th>
      <td>0.190794</td>
      <td>1.978757</td>
      <td>2.605967</td>
      <td>0.683509</td>
    </tr>
  </tbody>
</table>
</div>




```python
# get elements where of column 'Y' > 1
df[df['Y'] > 1]
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
      <th>W</th>
      <th>X</th>
      <th>Y</th>
      <th>Z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>E</th>
      <td>0.190794</td>
      <td>1.978757</td>
      <td>2.605967</td>
      <td>0.683509</td>
    </tr>
  </tbody>
</table>
</div>




```python
# multiple conditions
df[(df['W'] > 0) & (df['X'] > 1)] 
# df[(df['W'] > 0) and (df['X'] > 1)] -> this gives error because 'and' can compare only 2 boolean values, not a series of them. We've to use '&'
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
      <th>W</th>
      <th>X</th>
      <th>Y</th>
      <th>Z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>E</th>
      <td>0.190794</td>
      <td>1.978757</td>
      <td>2.605967</td>
      <td>0.683509</td>
    </tr>
  </tbody>
</table>
</div>



### DataFrame Index reset

* Index will be reset to 0, 1, 2..... 
* Previous index becomes a new column
* Set index to a particular column. Unlike reset, this will overwrite the index
* If we want to set index to a particular list then that list has to be added as a column first


```python
new_df = df.reset_index()
new_df # previous index A, B, C... has become a new column 'index'
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
      <th>index</th>
      <th>W</th>
      <th>X</th>
      <th>Y</th>
      <th>Z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>2.706850</td>
      <td>0.628133</td>
      <td>0.907969</td>
      <td>0.503826</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B</td>
      <td>0.651118</td>
      <td>-0.319318</td>
      <td>-0.848077</td>
      <td>0.605965</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C</td>
      <td>-2.018168</td>
      <td>0.740122</td>
      <td>0.528813</td>
      <td>-0.589001</td>
    </tr>
    <tr>
      <th>3</th>
      <td>D</td>
      <td>0.188695</td>
      <td>-0.758872</td>
      <td>-0.933237</td>
      <td>0.955057</td>
    </tr>
    <tr>
      <th>4</th>
      <td>E</td>
      <td>0.190794</td>
      <td>1.978757</td>
      <td>2.605967</td>
      <td>0.683509</td>
    </tr>
  </tbody>
</table>
</div>




```python
# index is changed
new_df['W']
```




    0    2.706850
    1    0.651118
    2   -2.018168
    3    0.188695
    4    0.190794
    Name: W, dtype: float64




```python
# add a new column to df
new_index = 'CA NY WY OR CO'.split() # instead of writing many quotes to create a list, this is a quick way.
df['States'] = new_index
df
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
      <th>W</th>
      <th>X</th>
      <th>Y</th>
      <th>Z</th>
      <th>States</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>2.706850</td>
      <td>0.628133</td>
      <td>0.907969</td>
      <td>0.503826</td>
      <td>CA</td>
    </tr>
    <tr>
      <th>B</th>
      <td>0.651118</td>
      <td>-0.319318</td>
      <td>-0.848077</td>
      <td>0.605965</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>C</th>
      <td>-2.018168</td>
      <td>0.740122</td>
      <td>0.528813</td>
      <td>-0.589001</td>
      <td>WY</td>
    </tr>
    <tr>
      <th>D</th>
      <td>0.188695</td>
      <td>-0.758872</td>
      <td>-0.933237</td>
      <td>0.955057</td>
      <td>OR</td>
    </tr>
    <tr>
      <th>E</th>
      <td>0.190794</td>
      <td>1.978757</td>
      <td>2.605967</td>
      <td>0.683509</td>
      <td>CO</td>
    </tr>
  </tbody>
</table>
</div>




```python
# set index to a particular column
df.set_index('States') # unlike df_reset(), this will overwrite the index
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
      <th>W</th>
      <th>X</th>
      <th>Y</th>
      <th>Z</th>
    </tr>
    <tr>
      <th>States</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CA</th>
      <td>2.706850</td>
      <td>0.628133</td>
      <td>0.907969</td>
      <td>0.503826</td>
    </tr>
    <tr>
      <th>NY</th>
      <td>0.651118</td>
      <td>-0.319318</td>
      <td>-0.848077</td>
      <td>0.605965</td>
    </tr>
    <tr>
      <th>WY</th>
      <td>-2.018168</td>
      <td>0.740122</td>
      <td>0.528813</td>
      <td>-0.589001</td>
    </tr>
    <tr>
      <th>OR</th>
      <td>0.188695</td>
      <td>-0.758872</td>
      <td>-0.933237</td>
      <td>0.955057</td>
    </tr>
    <tr>
      <th>CO</th>
      <td>0.190794</td>
      <td>1.978757</td>
      <td>2.605967</td>
      <td>0.683509</td>
    </tr>
  </tbody>
</table>
</div>



### Grouping DataFrame

* Create two lists for two index (outside and inside) of dataframe
* Combine lists to create a tuple
* Create multiindex using the tuple
* Create a dataframe using the multiindex
* Name the index
* Fetch elements


```python
# create an outside list
outside = ['G1', 'G1', 'G1', 'G2', 'G2', 'G2']
outside
```




    ['G1', 'G1', 'G1', 'G2', 'G2', 'G2']




```python
# create an outside list
inside = [1, 2, 3, 1, 2, 3]
inside
```




    [1, 2, 3, 1, 2, 3]




```python
# create a tuple by combining two lists
heir_index = list(zip(outside, inside)) # zip combines 2 lists into a tuple
heir_index
```




    [('G1', 1), ('G1', 2), ('G1', 3), ('G2', 1), ('G2', 2), ('G2', 3)]




```python
# create a multiindex using tupple
heir_index = pd.MultiIndex.from_tuples(heir_index)
heir_index
```




    MultiIndex([('G1', 1),
                ('G1', 2),
                ('G1', 3),
                ('G2', 1),
                ('G2', 2),
                ('G2', 3)],
               )




```python
# create a dataframe with multiindex
df = pd.DataFrame(randn(6, 2), heir_index, ['A', 'B'])
df
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
      <th></th>
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">G1</th>
      <th>1</th>
      <td>0.302665</td>
      <td>1.693723</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.706086</td>
      <td>-1.159119</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.134841</td>
      <td>0.390528</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">G2</th>
      <th>1</th>
      <td>0.166905</td>
      <td>0.184502</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.807706</td>
      <td>0.072960</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.638787</td>
      <td>0.329646</td>
    </tr>
  </tbody>
</table>
</div>




```python
# name the index
df.index.names = ['Groups', 'Numbers']
df
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
      <th></th>
      <th>A</th>
      <th>B</th>
    </tr>
    <tr>
      <th>Groups</th>
      <th>Numbers</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">G1</th>
      <th>1</th>
      <td>0.302665</td>
      <td>1.693723</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.706086</td>
      <td>-1.159119</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.134841</td>
      <td>0.390528</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">G2</th>
      <th>1</th>
      <td>0.166905</td>
      <td>0.184502</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.807706</td>
      <td>0.072960</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.638787</td>
      <td>0.329646</td>
    </tr>
  </tbody>
</table>
</div>




```python
# get elements of G1
df.loc['G1']
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
      <th>A</th>
      <th>B</th>
    </tr>
    <tr>
      <th>Numbers</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.302665</td>
      <td>1.693723</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.706086</td>
      <td>-1.159119</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.134841</td>
      <td>0.390528</td>
    </tr>
  </tbody>
</table>
</div>




```python
# to fetch a particular element
df.loc['G1'].loc[2]['B']
```




    -1.1591194155484297



### Cross section

* df.loc cannot directly access inner elements of a dataframe
* Cross section can be used to directly fetch any element of a dataframe


```python
df
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
      <th></th>
      <th>A</th>
      <th>B</th>
    </tr>
    <tr>
      <th>Groups</th>
      <th>Numbers</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">G1</th>
      <th>1</th>
      <td>0.302665</td>
      <td>1.693723</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.706086</td>
      <td>-1.159119</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.134841</td>
      <td>0.390528</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">G2</th>
      <th>1</th>
      <td>0.166905</td>
      <td>0.184502</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.807706</td>
      <td>0.072960</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.638787</td>
      <td>0.329646</td>
    </tr>
  </tbody>
</table>
</div>




```python
# get elements of index Numbers = 1
df.xs(1, level = 'Numbers')
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
      <th>A</th>
      <th>B</th>
    </tr>
    <tr>
      <th>Groups</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>G1</th>
      <td>0.302665</td>
      <td>1.693723</td>
    </tr>
    <tr>
      <th>G2</th>
      <td>0.166905</td>
      <td>0.184502</td>
    </tr>
  </tbody>
</table>
</div>



### Missing Data

* Create a dictionary with missing values (np.nan)
* Create dataframe using dictionary
* Drop NAN with or without threshold
* Fill NAN with desired value


```python
# create a dictionary with missing values (np.nan)
d = {'A' : [1, 2, np.nan], 'B' : [5, np.nan, np.nan], 'C' : [1, 2, 3]}
d
```




    {'A': [1, 2, nan], 'B': [5, nan, nan], 'C': [1, 2, 3]}




```python
# create dataframe using dictionary
df = pd.DataFrame(d)
df
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>5.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>NaN</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
# drop missing values
df.dropna() # drops rows with at least one nan - by default axis = 0
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>5.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# drop columns with nan
df.dropna(axis = 1)
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
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
# drop with threshold
df.dropna(thresh = 2) # drop rows that do not have at least 2 non-nan values
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>5.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>NaN</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
# fill nan with desired value
df.fillna(value = 'FILL VALUE')
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>FILL VALUE</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>FILL VALUE</td>
      <td>FILL VALUE</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
# fill nan in a particular column
df['A'].fillna(value = df['A'].mean())
```




    0    1.0
    1    2.0
    2    1.5
    Name: A, dtype: float64



### Groupby

* It allows us to group rows together based on a column and perform an aggregate function on them
* After grouping we can call aggregate functions such as mean(), sum(), std(), max(), min(), count() etc
* Function 'describe' on DataFrameGroupBy object gives a bunch of information


```python
# create a dataframe
data = {'Company':['GOOG','GOOG','MSFT','MSFT','FB','FB'],
       'Person':['Sam','Charlie','Amy','Vanessa','Carl','Sarah'],
       'Sales':[200,120,340,124,243,350]}
df = pd.DataFrame(data)
df
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
      <th>Company</th>
      <th>Person</th>
      <th>Sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GOOG</td>
      <td>Sam</td>
      <td>200</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GOOG</td>
      <td>Charlie</td>
      <td>120</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MSFT</td>
      <td>Amy</td>
      <td>340</td>
    </tr>
    <tr>
      <th>3</th>
      <td>MSFT</td>
      <td>Vanessa</td>
      <td>124</td>
    </tr>
    <tr>
      <th>4</th>
      <td>FB</td>
      <td>Carl</td>
      <td>243</td>
    </tr>
    <tr>
      <th>5</th>
      <td>FB</td>
      <td>Sarah</td>
      <td>350</td>
    </tr>
  </tbody>
</table>
</div>




```python
# group by company name
by_company = df.groupby('Company') # this is DataFrameGroupBy object. We can call aggregate functions on this object
by_company
```




    <pandas.core.groupby.generic.DataFrameGroupBy object at 0x00000150A99804C0>




```python
# aggregate function mean(), sum(), std(), max(), min(), count() etc
by_company.mean() # gets mean sales for each company. It ignores 'Person' column because it's string
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
      <th>Sales</th>
    </tr>
    <tr>
      <th>Company</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>FB</th>
      <td>296.5</td>
    </tr>
    <tr>
      <th>GOOG</th>
      <td>160.0</td>
    </tr>
    <tr>
      <th>MSFT</th>
      <td>232.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# get sum of sales of MSFT
df.groupby('Company').sum().loc['MSFT']
```




    Sales    464
    Name: MSFT, dtype: int64




```python
# describe function
df.groupby('Company').describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="8" halign="left">Sales</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>Company</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>FB</th>
      <td>2.0</td>
      <td>296.5</td>
      <td>75.660426</td>
      <td>243.0</td>
      <td>269.75</td>
      <td>296.5</td>
      <td>323.25</td>
      <td>350.0</td>
    </tr>
    <tr>
      <th>GOOG</th>
      <td>2.0</td>
      <td>160.0</td>
      <td>56.568542</td>
      <td>120.0</td>
      <td>140.00</td>
      <td>160.0</td>
      <td>180.00</td>
      <td>200.0</td>
    </tr>
    <tr>
      <th>MSFT</th>
      <td>2.0</td>
      <td>232.0</td>
      <td>152.735065</td>
      <td>124.0</td>
      <td>178.00</td>
      <td>232.0</td>
      <td>286.00</td>
      <td>340.0</td>
    </tr>
  </tbody>
</table>
</div>



### Concatenation

* Glues together DataFrames. Dimensions should match along the axis



```python
# create dataframes
df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                        'B': ['B0', 'B1', 'B2', 'B3'],
                        'C': ['C0', 'C1', 'C2', 'C3'],
                        'D': ['D0', 'D1', 'D2', 'D3']},
                        index=[0, 1, 2, 3])
df1
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A0</td>
      <td>B0</td>
      <td>C0</td>
      <td>D0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A1</td>
      <td>B1</td>
      <td>C1</td>
      <td>D1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A2</td>
      <td>B2</td>
      <td>C2</td>
      <td>D2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A3</td>
      <td>B3</td>
      <td>C3</td>
      <td>D3</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                        'B': ['B4', 'B5', 'B6', 'B7'],
                        'C': ['C4', 'C5', 'C6', 'C7'],
                        'D': ['D4', 'D5', 'D6', 'D7']},
                         index=[4, 5, 6, 7]) 
df2
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>A4</td>
      <td>B4</td>
      <td>C4</td>
      <td>D4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>A5</td>
      <td>B5</td>
      <td>C5</td>
      <td>D5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>A6</td>
      <td>B6</td>
      <td>C6</td>
      <td>D6</td>
    </tr>
    <tr>
      <th>7</th>
      <td>A7</td>
      <td>B7</td>
      <td>C7</td>
      <td>D7</td>
    </tr>
  </tbody>
</table>
</div>




```python
df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],
                        'B': ['B8', 'B9', 'B10', 'B11'],
                        'C': ['C8', 'C9', 'C10', 'C11'],
                        'D': ['D8', 'D9', 'D10', 'D11']},
                        index=[8, 9, 10, 11])
df3
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>A8</td>
      <td>B8</td>
      <td>C8</td>
      <td>D8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>A9</td>
      <td>B9</td>
      <td>C9</td>
      <td>D9</td>
    </tr>
    <tr>
      <th>10</th>
      <td>A10</td>
      <td>B10</td>
      <td>C10</td>
      <td>D10</td>
    </tr>
    <tr>
      <th>11</th>
      <td>A11</td>
      <td>B11</td>
      <td>C11</td>
      <td>D11</td>
    </tr>
  </tbody>
</table>
</div>




```python
# concatenation on rows
pd.concat([df1, df2, df3]).transpose()
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>A0</td>
      <td>A1</td>
      <td>A2</td>
      <td>A3</td>
      <td>A4</td>
      <td>A5</td>
      <td>A6</td>
      <td>A7</td>
      <td>A8</td>
      <td>A9</td>
      <td>A10</td>
      <td>A11</td>
    </tr>
    <tr>
      <th>B</th>
      <td>B0</td>
      <td>B1</td>
      <td>B2</td>
      <td>B3</td>
      <td>B4</td>
      <td>B5</td>
      <td>B6</td>
      <td>B7</td>
      <td>B8</td>
      <td>B9</td>
      <td>B10</td>
      <td>B11</td>
    </tr>
    <tr>
      <th>C</th>
      <td>C0</td>
      <td>C1</td>
      <td>C2</td>
      <td>C3</td>
      <td>C4</td>
      <td>C5</td>
      <td>C6</td>
      <td>C7</td>
      <td>C8</td>
      <td>C9</td>
      <td>C10</td>
      <td>C11</td>
    </tr>
    <tr>
      <th>D</th>
      <td>D0</td>
      <td>D1</td>
      <td>D2</td>
      <td>D3</td>
      <td>D4</td>
      <td>D5</td>
      <td>D6</td>
      <td>D7</td>
      <td>D8</td>
      <td>D9</td>
      <td>D10</td>
      <td>D11</td>
    </tr>
  </tbody>
</table>
</div>




```python
# concatenation on columns
pd.concat([df1, df2, df3], axis = 1) # a bunch of missing values
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A0</td>
      <td>B0</td>
      <td>C0</td>
      <td>D0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A1</td>
      <td>B1</td>
      <td>C1</td>
      <td>D1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A2</td>
      <td>B2</td>
      <td>C2</td>
      <td>D2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A3</td>
      <td>B3</td>
      <td>C3</td>
      <td>D3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>A4</td>
      <td>B4</td>
      <td>C4</td>
      <td>D4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>A5</td>
      <td>B5</td>
      <td>C5</td>
      <td>D5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>A6</td>
      <td>B6</td>
      <td>C6</td>
      <td>D6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>A7</td>
      <td>B7</td>
      <td>C7</td>
      <td>D7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>A8</td>
      <td>B8</td>
      <td>C8</td>
      <td>D8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>A9</td>
      <td>B9</td>
      <td>C9</td>
      <td>D9</td>
    </tr>
    <tr>
      <th>10</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>A10</td>
      <td>B10</td>
      <td>C10</td>
      <td>D10</td>
    </tr>
    <tr>
      <th>11</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>A11</td>
      <td>B11</td>
      <td>C11</td>
      <td>D11</td>
    </tr>
  </tbody>
</table>
</div>



### Merging
 
* Merges dataframes together. This is similar to SQL joins
* Inner: Only common elements in both the dataframes based on given columns
* Left: All elements in left and common elements in both dataframes based on given columns
* Right: All elements in right and common elements in both dataframes based on given columns
* Outer: All elements in both dataframes based on given columns


```python
# create 2 dataframes
left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                     'key2': ['K0', 'K1', 'K0', 'K1'],
                        'A': ['A0', 'A1', 'A2', 'A3'],
                        'B': ['B0', 'B1', 'B2', 'B3']})
    
right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                               'key2': ['K0', 'K0', 'K0', 'K0'],
                                  'C': ['C0', 'C1', 'C2', 'C3'],
                                  'D': ['D0', 'D1', 'D2', 'D3']})
```


```python
left
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
      <th>key1</th>
      <th>key2</th>
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>K0</td>
      <td>K0</td>
      <td>A0</td>
      <td>B0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>K0</td>
      <td>K1</td>
      <td>A1</td>
      <td>B1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>K1</td>
      <td>K0</td>
      <td>A2</td>
      <td>B2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>K2</td>
      <td>K1</td>
      <td>A3</td>
      <td>B3</td>
    </tr>
  </tbody>
</table>
</div>




```python
right
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
      <th>key1</th>
      <th>key2</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>K0</td>
      <td>K0</td>
      <td>C0</td>
      <td>D0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>K1</td>
      <td>K0</td>
      <td>C1</td>
      <td>D1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>K1</td>
      <td>K0</td>
      <td>C2</td>
      <td>D2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>K2</td>
      <td>K0</td>
      <td>C3</td>
      <td>D3</td>
    </tr>
  </tbody>
</table>
</div>




```python
# inner join on key1 and key2 - only common elements (key1 and key2) in both dataframes
pd.merge(left, right, how='inner', on=['key1', 'key2'])
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
      <th>key1</th>
      <th>key2</th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>K0</td>
      <td>K0</td>
      <td>A0</td>
      <td>B0</td>
      <td>C0</td>
      <td>D0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>K1</td>
      <td>K0</td>
      <td>A2</td>
      <td>B2</td>
      <td>C1</td>
      <td>D1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>K1</td>
      <td>K0</td>
      <td>A2</td>
      <td>B2</td>
      <td>C2</td>
      <td>D2</td>
    </tr>
  </tbody>
</table>
</div>




```python
# left join on key1 and key2 - all elements in left and common elements in both dataframes
pd.merge(left, right, how='left', on=['key1', 'key2'])
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
      <th>key1</th>
      <th>key2</th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>K0</td>
      <td>K0</td>
      <td>A0</td>
      <td>B0</td>
      <td>C0</td>
      <td>D0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>K0</td>
      <td>K1</td>
      <td>A1</td>
      <td>B1</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>K1</td>
      <td>K0</td>
      <td>A2</td>
      <td>B2</td>
      <td>C1</td>
      <td>D1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>K1</td>
      <td>K0</td>
      <td>A2</td>
      <td>B2</td>
      <td>C2</td>
      <td>D2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>K2</td>
      <td>K1</td>
      <td>A3</td>
      <td>B3</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# right join on key1 and key2 - all elements in right and common elements in both dataframes
pd.merge(left, right, how='right', on=['key1', 'key2'])
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
      <th>key1</th>
      <th>key2</th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>K0</td>
      <td>K0</td>
      <td>A0</td>
      <td>B0</td>
      <td>C0</td>
      <td>D0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>K1</td>
      <td>K0</td>
      <td>A2</td>
      <td>B2</td>
      <td>C1</td>
      <td>D1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>K1</td>
      <td>K0</td>
      <td>A2</td>
      <td>B2</td>
      <td>C2</td>
      <td>D2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>K2</td>
      <td>K0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>C3</td>
      <td>D3</td>
    </tr>
  </tbody>
</table>
</div>




```python
# outer join on key1 and key2 - all elements in both dataframes
pd.merge(left, right, how='outer', on=['key1', 'key2'])
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
      <th>key1</th>
      <th>key2</th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>K0</td>
      <td>K0</td>
      <td>A0</td>
      <td>B0</td>
      <td>C0</td>
      <td>D0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>K0</td>
      <td>K1</td>
      <td>A1</td>
      <td>B1</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>K1</td>
      <td>K0</td>
      <td>A2</td>
      <td>B2</td>
      <td>C1</td>
      <td>D1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>K1</td>
      <td>K0</td>
      <td>A2</td>
      <td>B2</td>
      <td>C2</td>
      <td>D2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>K2</td>
      <td>K1</td>
      <td>A3</td>
      <td>B3</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>K2</td>
      <td>K0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>C3</td>
      <td>D3</td>
    </tr>
  </tbody>
</table>
</div>



### Joining

* It is a method for combining columns of two differently-indexed dataframes
* It's similar to merge but here we'll be combining based on index


```python
# create dataframes with different index
left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                     'B': ['B0', 'B1', 'B2']},
                      index=['K0', 'K1', 'K2']) 

right = pd.DataFrame({'C': ['C0', 'C2', 'C3'],
                    'D': ['D0', 'D2', 'D3']},
                      index=['K0', 'K2', 'K3'])
```


```python
left
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
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>K0</th>
      <td>A0</td>
      <td>B0</td>
    </tr>
    <tr>
      <th>K1</th>
      <td>A1</td>
      <td>B1</td>
    </tr>
    <tr>
      <th>K2</th>
      <td>A2</td>
      <td>B2</td>
    </tr>
  </tbody>
</table>
</div>




```python
right
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
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>K0</th>
      <td>C0</td>
      <td>D0</td>
    </tr>
    <tr>
      <th>K2</th>
      <td>C2</td>
      <td>D2</td>
    </tr>
    <tr>
      <th>K3</th>
      <td>C3</td>
      <td>D3</td>
    </tr>
  </tbody>
</table>
</div>




```python
# left join
left.join(right)
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>K0</th>
      <td>A0</td>
      <td>B0</td>
      <td>C0</td>
      <td>D0</td>
    </tr>
    <tr>
      <th>K1</th>
      <td>A1</td>
      <td>B1</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>K2</th>
      <td>A2</td>
      <td>B2</td>
      <td>C2</td>
      <td>D2</td>
    </tr>
  </tbody>
</table>
</div>




```python
# right join
right.join(left)
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
      <th>C</th>
      <th>D</th>
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>K0</th>
      <td>C0</td>
      <td>D0</td>
      <td>A0</td>
      <td>B0</td>
    </tr>
    <tr>
      <th>K2</th>
      <td>C2</td>
      <td>D2</td>
      <td>A2</td>
      <td>B2</td>
    </tr>
    <tr>
      <th>K3</th>
      <td>C3</td>
      <td>D3</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# inner join
right.join(left, how='inner')
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
      <th>C</th>
      <th>D</th>
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>K0</th>
      <td>C0</td>
      <td>D0</td>
      <td>A0</td>
      <td>B0</td>
    </tr>
    <tr>
      <th>K2</th>
      <td>C2</td>
      <td>D2</td>
      <td>A2</td>
      <td>B2</td>
    </tr>
  </tbody>
</table>
</div>




```python
# outer join
right.join(left, how='outer')
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
      <th>C</th>
      <th>D</th>
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>K0</th>
      <td>C0</td>
      <td>D0</td>
      <td>A0</td>
      <td>B0</td>
    </tr>
    <tr>
      <th>K1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>A1</td>
      <td>B1</td>
    </tr>
    <tr>
      <th>K2</th>
      <td>C2</td>
      <td>D2</td>
      <td>A2</td>
      <td>B2</td>
    </tr>
    <tr>
      <th>K3</th>
      <td>C3</td>
      <td>D3</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



### Operations

* Get unique elements in a dataframe
* Get unique elements count
* Apply our custom functions to dataframe
* Sort dataframes by column
* Store dataframe as a multiindex table using pivot_table function


```python
# create dataframe
df = pd.DataFrame({'col1':[1,2,3,4],'col2':[444,555,666,444],'col3':['abc','def','ghi','xyz']})
df
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
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>444</td>
      <td>abc</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>555</td>
      <td>def</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>666</td>
      <td>ghi</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>444</td>
      <td>xyz</td>
    </tr>
  </tbody>
</table>
</div>




```python
# returns first n rows
df.head(2)
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
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>444</td>
      <td>abc</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>555</td>
      <td>def</td>
    </tr>
  </tbody>
</table>
</div>




```python
# unique elements
df['col2'].unique()
```




    array([444, 555, 666], dtype=int64)




```python
# number of unique elements
df['col2'].nunique() # OR len(df['col2'].unique())
```




    3




```python
# to know how many times a value has occured in the dataframe
df['col2'].value_counts()
```




    444    2
    555    1
    666    1
    Name: col2, dtype: int64




```python
# to apply our custom functions to dataframe
def times2(num):
    return num * 2

df['col1'].apply(times2)
```




    0    2
    1    4
    2    6
    3    8
    Name: col1, dtype: int64




```python
# using lambda function
df['col1'].apply(lambda num : num * 2)
```




    0    2
    1    4
    2    6
    3    8
    Name: col1, dtype: int64




```python
# using apply function we can also pass built-in functions
df['col3'].apply(len)
```




    0    3
    1    3
    2    3
    3    3
    Name: col3, dtype: int64




```python
# sort by column
df.sort_values('col2') # by default ascending
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
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>444</td>
      <td>abc</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>444</td>
      <td>xyz</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>555</td>
      <td>def</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>666</td>
      <td>ghi</td>
    </tr>
  </tbody>
</table>
</div>




```python
# check any null values
df.isnull()
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
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
data = {'A':['foo','foo','foo','bar','bar','bar'],
     'B':['one','one','two','two','one','one'],
       'C':['x','y','x','y','x','y'],
       'D':[1,3,2,5,4,1]}
df = pd.DataFrame(data)
df
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>foo</td>
      <td>one</td>
      <td>x</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>foo</td>
      <td>one</td>
      <td>y</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>foo</td>
      <td>two</td>
      <td>x</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bar</td>
      <td>two</td>
      <td>y</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>bar</td>
      <td>one</td>
      <td>x</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>bar</td>
      <td>one</td>
      <td>y</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# to store dataframe as a multi-index table
df.pivot_table(values = 'D', index = ['A', 'B'], columns = ['C'])
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
      <th>C</th>
      <th>x</th>
      <th>y</th>
    </tr>
    <tr>
      <th>A</th>
      <th>B</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">bar</th>
      <th>one</th>
      <td>4.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>two</th>
      <td>NaN</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">foo</th>
      <th>one</th>
      <td>1.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>two</th>
      <td>2.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



### Data Input and Output

* CSV
* Excel
* HTML
* SQL


```python
# read from csv file
df = pd.read_csv('data/example_csv')
df
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
# write dataframe to csv file when index = True (by default)
df.to_csv('data/output_csv') # if index = True then index of dataframe will be saved as a new unnamed column
pd.read_csv('data/output_csv')
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
      <th>Unnamed: 0</th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
# write dataframe to csv file with index = False (do not consider index or ignore index of the datafram)
df.to_csv('data/output_csv', index = False) 
pd.read_csv('data/output_csv')
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
# read from excel file
df = pd.read_excel('data/example_excel.xlsx', sheet_name = 'Sheet1')
df
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
# write to excel file
df.to_excel('data/output_excel.xlsx', sheet_name = 'Sheet2', index = False)
pd.read_excel('data/output_excel.xlsx', sheet_name = 'Sheet2')
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
# read html
data = pd.read_html('https://www.fdic.gov/resources/resolutions/bank-failures/failed-bank-list/banklist.html')
type(data) # a list that contains a dataframe
```




    list




```python
df = data[0]
df
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
      <th>Bank Name</th>
      <th>City</th>
      <th>ST</th>
      <th>CERT</th>
      <th>Acquiring Institution</th>
      <th>Closing Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The First State Bank</td>
      <td>Barboursville</td>
      <td>WV</td>
      <td>14361</td>
      <td>MVB Bank, Inc.</td>
      <td>April 3, 2020</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ericson State Bank</td>
      <td>Ericson</td>
      <td>NE</td>
      <td>18265</td>
      <td>Farmers and Merchants Bank</td>
      <td>February 14, 2020</td>
    </tr>
    <tr>
      <th>2</th>
      <td>City National Bank of New Jersey</td>
      <td>Newark</td>
      <td>NJ</td>
      <td>21111</td>
      <td>Industrial Bank</td>
      <td>November 1, 2019</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Resolute Bank</td>
      <td>Maumee</td>
      <td>OH</td>
      <td>58317</td>
      <td>Buckeye State Bank</td>
      <td>October 25, 2019</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Louisa Community Bank</td>
      <td>Louisa</td>
      <td>KY</td>
      <td>58112</td>
      <td>Kentucky Farmers Bank Corporation</td>
      <td>October 25, 2019</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>556</th>
      <td>Superior Bank, FSB</td>
      <td>Hinsdale</td>
      <td>IL</td>
      <td>32646</td>
      <td>Superior Federal, FSB</td>
      <td>July 27, 2001</td>
    </tr>
    <tr>
      <th>557</th>
      <td>Malta National Bank</td>
      <td>Malta</td>
      <td>OH</td>
      <td>6629</td>
      <td>North Valley Bank</td>
      <td>May 3, 2001</td>
    </tr>
    <tr>
      <th>558</th>
      <td>First Alliance Bank &amp; Trust Co.</td>
      <td>Manchester</td>
      <td>NH</td>
      <td>34264</td>
      <td>Southern New Hampshire Bank &amp; Trust</td>
      <td>February 2, 2001</td>
    </tr>
    <tr>
      <th>559</th>
      <td>National State Bank of Metropolis</td>
      <td>Metropolis</td>
      <td>IL</td>
      <td>3815</td>
      <td>Banterra Bank of Marion</td>
      <td>December 14, 2000</td>
    </tr>
    <tr>
      <th>560</th>
      <td>Bank of Honolulu</td>
      <td>Honolulu</td>
      <td>HI</td>
      <td>21029</td>
      <td>Bank of the Orient</td>
      <td>October 13, 2000</td>
    </tr>
  </tbody>
</table>
<p>561 rows × 6 columns</p>
</div>




```python
df['City']
```




    0      Barboursville
    1            Ericson
    2             Newark
    3             Maumee
    4             Louisa
               ...      
    556         Hinsdale
    557            Malta
    558       Manchester
    559       Metropolis
    560         Honolulu
    Name: City, Length: 561, dtype: object




```python
df.loc[1]['Bank Name']
```




    'Ericson State Bank'




```python
# to read sql first create sql engine in memory
from sqlalchemy import create_engine
sql_engine = create_engine('sqlite:///:memory:')

# take original df
df = pd.read_csv('data/example_csv')
df
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
# populate data to sql table
df.to_sql('my_table', sql_engine) # we can run this only once because we get table already exist error
sqldf = pd.read_sql('my_table', con = sql_engine) # sql_engine is the connection to sql
sqldf
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
      <th>index</th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>


