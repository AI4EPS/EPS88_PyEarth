# PyEarth: A Python Introduction to Earth Science
## Class 1 Exercises

### Exercise 1: Variables and Data Types

Create variables for the following Earth-related information:

- Name of our planet
- Earth's radius in kilometers
- Earth's average surface temperature in Celsius
- Whether Earth has a moon (use a boolean)
- A list of Earth's layers (inner core, outer core, mantle, crust)

Print all these variables.


### Exercise 2: Arithmetic Operations

Calculate the following:
- The circumference of Earth (use the radius from Exercise 1 and the formula 2 * π * r)
- The difference between the boiling point of water (100°C) and Earth's average surface temperature
- The number of times Earth's diameter (use the radius from Exercise 1) can fit between Earth and the Moon (average distance: 384,400 km)

Print the results.


### Exercise 3: Control Flow

Create a function that takes a temperature in Celsius and returns a description of Earth's temperature:
- If temp < 0: "Earth is in an ice age"
- If 0 <= temp < 15: "Earth is cool"
- If 15 <= temp < 25: "Earth is moderate"
- If temp >= 25: "Earth is warm"

Test your function with different temperatures.


### Exercise 4: Lists and Loops

Given the list of planets: `["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]`

Write a loop that prints each planet's name and its position from the Sun.
Example output: "Mercury is the 1st planet from the Sun"


### Exercise 5: Functions

Write a function that converts kilometers to miles (1 km = 0.621371 miles).
Use this function to convert Earth's radius to miles.


### Bonus Exercise: Dictionaries

Create a dictionary for Earth with the following keys and values:
- name: "Earth"
- radius_km: (use the value from Exercise 1)
- has_moon: (use the value from Exercise 1)
- atmosphere_composition: {"nitrogen": 78, "oxygen": 21, "other": 1}

Write a function that takes this dictionary and prints a summary of Earth's properties.

Example output:
"Earth has a radius of X km and has/doesn't have a moon. Its atmosphere is composed of 78% nitrogen, 21% oxygen, and 1% other gases."
