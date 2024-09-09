---
marp: true
theme: default
paginate: true
---

# PyEarth: A Python Introduction to Earth Science
## Lecture 2: NumPy and Pandas

---

# Introduction to NumPy

- NumPy: Numerical Python
- Fundamental package for scientific computing in Python
- Provides support for large, multi-dimensional arrays and matrices
- Offers a wide range of mathematical functions

---

# Why NumPy?

- Efficient: Optimized for performance
- Versatile: Supports various data types
- Integrates well with other libraries
- Essential for data analysis and scientific computing

---

# Creating NumPy Arrays

```python
import numpy as np

# From a list
arr1 = np.array([1, 2, 3, 4, 5])

# Using NumPy functions
arr2 = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
arr3 = np.linspace(0, 1, 5)  # [0, 0.25, 0.5, 0.75, 1]
arr4 = np.zeros((3, 3))  # 3x3 array of zeros
arr5 = np.ones((2, 4))  # 2x4 array of ones
arr6 = np.random.rand(3, 3)  # 3x3 array of random values
```

---

# Useful NumPy Functions

1. Array operations:
   - `np.reshape()`: Reshape an array
   - `np.concatenate()`: Join arrays
   - `np.split()`: Split an array

2. Mathematical operations:
   - `np.sum()`, `np.mean()`, `np.std()`: Basic statistics
   - `np.min()`, `np.max()`: Find minimum and maximum values
   - `np.argmin()`, `np.argmax()`: Find indices of min/max values

---

# Useful NumPy Functions (cont.)

3. Linear algebra:
   - `np.dot()`: Matrix multiplication
   - `np.linalg.inv()`: Matrix inverse
   - `np.linalg.eig()`: Eigenvalues and eigenvectors

4. Array manipulation:
   - `np.transpose()`: Transpose an array
   - `np.sort()`: Sort an array
   - `np.unique()`: Find unique elements

---

# How to Find NumPy Functions

1. GPT, Claude, and other AI assistants
2. Use Python's built-in help function:
   ```python
   import numpy as np
   help(np.array)
   ```
3. Use IPython/Jupyter Notebook's tab completion and `?` operator:
   ```python
   np.array?
   ```

---

# NumPy vs. Basic Python: Speed Comparison

Let's compare the speed of calculating the mean of a large array:

```python
import numpy as np
import time

# Create large arrays
size = 10000000
data = list(range(size))
np_data = np.array(data)

# Python list comprehension
start = time.time()
result_py = [x**2 + 2*x + 1 for x in data]
end = time.time()
print(f"Python time: {end - start:.6f} seconds")

# NumPy vectorized operation
start = time.time()
result_np = np_data**2 + 2*np_data + 1
end = time.time()
print(f"NumPy time: {end - start:.6f} seconds")
```

NumPy is significantly faster due to its optimized C implementation.

---

# Real-world Example: Analyzing Earthquake Data

We'll use NumPy to analyze earthquake data:

```python
import numpy as np

# Load earthquake data (magnitude and depth)
# the first coloumn is utc datetime
earthquakes = np.loadtxt("data/earthquakes.csv", delimiter=",", skiprows=1, usecols=(1, 2, 3, 4), dtype=float)

# Calculate average magnitude and depth
avg_depth = np.mean(earthquakes[:, 2])
avg_magnitude = np.mean(earthquakes[:, 3])

# Find the strongest earthquake
strongest_idx = np.argmax(earthquakes[:, 3])
strongest_magnitude = earthquakes[strongest_idx, 3]
strongest_depth = earthquakes[strongest_idx, 2]

print(f"Average magnitude: M{avg_magnitude:.2f}")
print(f"Average depth: {avg_depth:.2f} km")
print(f"Strongest earthquake: Magnitude {strongest_magnitude:.2f} at depth {strongest_depth:.2f} km")
```

---

# Introduction to Pandas

- Pandas: Python Data Analysis Library
- Built on top of NumPy
- Provides high-performance, easy-to-use data structures and tools
- Essential for data manipulation and analysis

---

# Why Pandas?

- Handles structured data efficiently
- Powerful data alignment and merging capabilities
- Integrates well with other libraries
- Excellent for handling time series data
- Built-in tools for reading/writing various file formats

---

# Pandas Data Structures

1. Series: 1D labeled array
2. DataFrame: 2D labeled data structure with columns of potentially different types

```python
import pandas as pd

# Create a Series
s = pd.Series([1, 3, 5, np.nan, 6, 8])

# Create a DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': pd.date_range('20230101', periods=4),
    'C': pd.Series(1, index=range(4), dtype='float32'),
    'D': np.array([3] * 4, dtype='int32'),
    'E': pd.Categorical(["test", "train", "test", "train"]),
    'F': 'foo'
})
```

---

# Useful Pandas Functions

1. Data loading and saving:
   - `pd.read_csv()`, `pd.read_excel()`, `pd.read_sql()`
   - `df.to_csv()`, `df.to_excel()`, `df.to_sql()`

2. Data inspection:
   - `df.head()`, `df.tail()`: View first/last rows
   - `df.info()`: Summary of DataFrame
   - `df.describe()`: Statistical summary

3. Data selection:
   - `df['column']`: Select a column
   - `df.loc[]`: Label-based indexing
   - `df.iloc[]`: Integer-based indexing

---

# Useful Pandas Functions (cont.)

4. Data manipulation:
   - `df.groupby()`: Group data
   - `df.merge()`: Merge DataFrames
   - `df.pivot()`: Reshape data

5. Data cleaning:
   - `df.dropna()`: Drop missing values
   - `df.fillna()`: Fill missing values
   - `df.drop_duplicates()`: Remove duplicate rows

6. Time series functionality:
   - `pd.date_range()`: Create date ranges
   - `df.resample()`: Resample time series data

---

# How to Find Pandas Functions

1. GPT, Claude, and other AI assistants
2. Use Python's built-in help function:
   ```python
   import pandas as pd
   help(pd.DataFrame)
   ```
3. Use IPython/Jupyter Notebook's tab completion and `?` operator:
   ```python
   pd.DataFrame?
   ```

---

# Pandas vs. NumPy

- Pandas is built on top of NumPy
- Pandas adds functionality for handling structured data
- Pandas excels at:
  - Handling missing data
  - Data alignment
  - Merging and joining datasets
  - Time series functionality
- NumPy is better for:
  - Large numerical computations
  - Linear algebra operations
  - When you need ultimate performance

---

# Real-world Example: Revisit the Earthquake Data

We'll use Pandas to analyze earthquake data this time:

```python
import pandas as pd

# Load earthquake data
df = pd.read_csv("data/earthquakes.csv")

# Calculate average magnitude and depth
avg_depth = df['depth'].mean()
avg_magnitude = df['magnitude'].mean()

# Find the strongest earthquake
strongest_idx = df['magnitude'].idxmax()
strongest_magnitude = df.loc[strongest_idx, 'magnitude']
strongest_depth = df.loc[strongest_idx, 'depth']

print(f"Average magnitude: M{avg_magnitude:.2f}")
print(f"Average depth: {avg_depth:.2f} km")
print(f"Strongest earthquake: Magnitude {strongest_magnitude:.2f} at depth {strongest_depth:.2f} km")
```

---

# Real-world Example: Analyzing Temperature Data

We'll use Pandas to analyze temperature data:

```python
import pandas as pd

# Load temperature data
df = pd.read_csv("data/global_temperature.csv")

# Convert date column to datetime
df["date"] = pd.to_datetime(df["date"])

# Set date as index
df.set_index("date", inplace=True)

# Find the hottest and coldest days
hottest_day = df["temperature"].idxmax()
coldest_day = df["temperature"].idxmin()

print(f"Hottest day: {hottest_day.date()} ({df.loc[hottest_day, 'temperature']:.1f}°C)")
print(f"Coldest day: {coldest_day.date()} ({df.loc[coldest_day, 'temperature']:.1f}°C)")

# Calculate monthly average temperatures
yearly_avg = df.resample("Y").mean()

# Plot monthly average temperatures
yearly_avg["temperature"].plot(figsize=(12, 6))

plt.title("Yearly Average Temperatures")
plt.ylabel("Temperature (°C)")
plt.show()
```

---

# Conclusion

- NumPy and Pandas are essential tools for data analysis in Python
- NumPy excels at numerical computations and array operations
- Pandas is great for structured data manipulation and analysis
- Both libraries integrate well with other scientific Python tools
- Practice and explore these libraries to become proficient in data analysis!

