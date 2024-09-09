---
marp: true
theme: default
paginate: true
---

# PyEarth: A Python Introduction to Earth Science
## Lecture 3: Matplotlib, Cartopy, and PyGMT

---

# Introduction to Matplotlib

- Matplotlib is a powerful plotting library for Python
- It allows you to create a wide variety of static, animated, and interactive visualizations
- Used extensively in scientific computing and data analysis

---

# Basic Matplotlib Structure

```python
import matplotlib.pyplot as plt

# Create a figure and axis
fig, ax = plt.subplots()

x, y = [0, 1, 2, 3, 4], [0, 1, 4, 9, 16]

# Plot data
ax.plot(x, y, "o-", color="blue", label="Data")

# Add labels and title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.legend()
ax.set_title('My Plot')

# Show the plot
plt.show()
```

---

# Common Matplotlib Functions

1. Line Plot: `plt.plot(x, y)`
2. Scatter Plot: `plt.scatter(x, y)`
3. Bar Plot: `plt.bar(x, y)`
4. Histogram: `plt.hist(data)`
5. Box Plot: `plt.boxplot(data)`
6. Pie Chart: `plt.pie(sizes)`

---

# Line Plot Example

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Sine Wave')
plt.show()
```


---

# Scatter Plot Example

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.random.rand(50)
y = np.random.rand(50)
colors = np.random.rand(50)
sizes = 1000 * np.random.rand(50)

plt.scatter(x, y, c=colors, s=sizes, alpha=0.5)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot')
plt.show()
```


---

# Bar Plot Example

```python
import matplotlib.pyplot as plt

categories = ['A', 'B', 'C', 'D']
values = [3, 7, 2, 5]

plt.bar(categories, values)
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Bar Plot')
plt.show()
```


---

# Histogram Example

```python
import matplotlib.pyplot as plt
import numpy as np

data = np.random.randn(1000)

plt.hist(data, bins=30)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.show()
```


---

# Box Plot Example

```python
import matplotlib.pyplot as plt
import numpy as np

data = [np.random.normal(0, std, 100) for std in range(1, 4)]

plt.boxplot(data)
plt.xlabel('Group')
plt.ylabel('Value')
plt.title('Box Plot')
plt.show()
```


---

# Pie Chart Example

```python
import matplotlib.pyplot as plt

sizes = [30, 20, 25, 15, 10]
labels = ['A', 'B', 'C', 'D', 'E']

plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.title('Pie Chart')
plt.axis('equal')
plt.show()
```


---

# Global Earthquake Dataset Example

```python
import matplotlib.pyplot as plt
import pandas as pd

# Read the earthquake data
df = pd.read_csv('data/earthquakes.csv')

plt.figure(figsize=(12, 8))
plt.scatter(df['longitude'], df['latitude'], 
            s=df['magnitude']*10, alpha=0.5, c="red")
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Global Earthquakes')
plt.show()
```


---

# Introduction to Cartopy

- Cartopy is a library for cartographic projections and geospatial data visualization
- It provides object-oriented map projection definitions
- Allows easy manipulation of geospatial data and creation of map plots

---

# Cartopy: Plotting Earthquakes

```python
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd

df = pd.read_csv('data/earthquakes.csv')

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)

scatter = ax.scatter(df['longitude'], df['latitude'], 
                     s=df['magnitude']*10, alpha=0.5, c="red",
                     transform=ccrs.PlateCarree())

ax.set_global()
plt.title('Global Earthquakes (Cartopy)')
plt.show()
```


---

# Common Map Projections

1. Plate Carrée (Equirectangular)
2. Mercator
3. Lambert Conformal Conic
4. Orthographic

---

# Plate Carrée Projection

- Simplest projection
- Latitude and longitude lines are equally spaced
- Distorts shape and size, especially near poles

```python
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines()
ax.gridlines()
plt.title("PlateCarree Projection")
plt.show()
```



---

# Mercator Projection

- Conformal projection (preserves angles)
- Used for navigation
- Severely distorts size near poles

```python
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator())
ax.coastlines()
ax.gridlines()
plt.title("Mercator Projection")
plt.show()
```

---

# Lambert Conformal Conic Projection

- Conformal projection
- Good for mid-latitude regions
- Used for aeronautical charts

```python
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal())
ax.coastlines()
ax.gridlines()
plt.title("Lambert Conformal Conic Projection")
plt.show()
```

---

# Orthographic Projection

- Perspective projection (as seen from space)
- Shows one hemisphere at a time
- Useful for visualizing global phenomena

```python
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic())
ax.coastlines()
ax.gridlines()
plt.title("Orthographic Projection")
plt.show()
```

---

# When to Use Each Projection

- Plate Carrée: Simple global views, data in lat/lon coordinates
- Mercator: Navigation, web maps (e.g., Google Maps)
- Lambert Conformal Conic: Regional maps, especially mid-latitudes
- Orthographic: Visualizing Earth as a globe, planetary science

---

# Example: Plate Carrée

```python
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd

df = pd.read_csv('data/earthquakes.csv')
region = [-125, -114, 32, 42]
df = df[(df['longitude'] >= region[0]) & (df['longitude'] <= region[1]) &
        (df['latitude'] >= region[2]) & (df['latitude'] <= region[3])]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

ax.set_extent(region)
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.gridlines(draw_labels=True)

ax.scatter(df['longitude'], df['latitude'], 
           s=df['magnitude']*10, alpha=0.5, c="red", 
           transform=ccrs.PlateCarree())

plt.title('California Earthquakes (Plate Carrée)')
plt.show()
```

---

# Adding Topography with Cartopy

```python
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd

df = pd.read_csv('data/earthquakes.csv')
region = [-125, -114, 32, 42]
df = df[(df['longitude'] >= region[0]) & (df['longitude'] <= region[1]) &
        (df['latitude'] >= region[2]) & (df['latitude'] <= region[3])]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

ax.set_extent(region)
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.gridlines(draw_labels=True)

# Add topography
ax.stock_img()

ax.scatter(df['longitude'], df['latitude'], 
           s=df['magnitude']*10, alpha=0.5, c="red", 
           transform=ccrs.PlateCarree())

plt.title('California Earthquakes (Plate Carrée)')
plt.show()
```

---

# Introduction to PyGMT

- PyGMT is a Python interface for the Generic Mapping Tools (GMT)
- Provides access to GMT's powerful mapping and data processing capabilities
- Can handle high-resolution topography data

---

# PyGMT: Downloading Topography Data

```python
import pygmt

# Download topography data for California
region = [-125, -114, 32, 42]
grid = pygmt.datasets.load_earth_relief(resolution="30s", region=region)

# Plot the topography
fig = pygmt.Figure()
fig.grdimage(grid=grid, projection="M15c", frame=True, cmap="geo")
fig.colorbar(frame=["a1000", "x+lElevation", "y+lm"])
fig.coast(shorelines="1/0.5p")
fig.show()
```

---

# Combining PyGMT and Cartopy

```python
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pygmt
import numpy as np
import pandas as pd

# Download topography data
region = [-125, -114, 32, 42]
df = pd.read_csv('data/earthquakes.csv')
df = df[(df['longitude'] >= region[0]) & (df['longitude'] <= region[1]) &
        (df['latitude'] >= region[2]) & (df['latitude'] <= region[3])]

grid = pygmt.datasets.load_earth_relief(resolution="30s", region=region)

# Create a Cartopy plot
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_extent(region)

# Plot the topography
img_extent = (grid.lon.min(), grid.lon.max(), grid.lat.min(), grid.lat.max())
ax.set_extent(region, crs=ccrs.PlateCarree())
ax.scatter(df['longitude'], df['latitude'], 
           s=df['magnitude']*10, alpha=0.5, c="red", 
           transform=ccrs.PlateCarree())

# Plot the earthquake data
ax.imshow(grid.data, extent=img_extent, transform=ccrs.PlateCarree(), 
          cmap='terrain', origin='lower')

# Add coastlines and borders
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.STATES)
ax.gridlines(draw_labels=True)

plt.colorbar(ax.images[0], label='Elevation (m)')
plt.title('California Topography (PyGMT + Cartopy)')
plt.show()
```

---

# Summary

- Matplotlib: Versatile plotting library for various chart types
- Cartopy: Geospatial data visualization with map projections
- PyGMT: Access to high-resolution topography data and GMT capabilities
- Combining these tools allows for powerful Earth science visualizations
