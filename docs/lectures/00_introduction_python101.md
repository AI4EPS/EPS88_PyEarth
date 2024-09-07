---
marp: true
theme: invert
class: compact
paginate: true
# size: 4:3
---

# PyEarth: A Python Introduction to Earth Science
## Class 1: Introduction && Python 101

---

# Course Overview

- Focus on Python programming for Earth Science applications
- Topics include:
  - Python basics
  - Data analysis with NumPy and Pandas
  - Data visualization with Matplotlib and Cartopy
  - Machine learning with Scikit-learn
  - Deep learning with PyTorch
- Project-based learning

---

# Grading

- Attendance: 10%
- In-class and homework exercises: 50%
- Project presentation: 20%
- Project report: 20%

Bonus points:

- Assisting classmates: 10%
- Contributing course materials: 10%
- Achieving success in a Kaggle competition (top 10% rank): 10%

---

# Class Tools

## GitHub
- Version control system
- Collaboration platform

## Codespaces
- Cloud-based development environment
- Pre-configured with necessary tools and libraries

## Copilot
- AI-powered coding assistant
- Helps with code completion and generation

---

# How to Use Copilot

1. Install Copilot extension
2. Authenticate with your GitHub account
3. Start coding and watch for suggestions
4. Accept suggestions with Tab or continue typing

Tips:
- Write clear comments to guide Copilot
- Review and understand the suggested code

---

# Using Chatbots for Programming and Earth Science Questions

- AI-powered language models, such as GPT, Claude, Gemini, Grok, LLama, etc.
- Can assist with:
  - Explaining concepts
  - Debugging code
  - Answering Earth Science questions
  - Providing coding examples

Tips:
- Be specific in your questions
- Verify information with reliable sources
- Use as a learning aid, not a substitute for understanding

---

# 4. Introduction to Python

## What is Python?
- High-level, interpreted programming language
- Known for its simplicity and readability
- Widely used in scientific computing and data analysis

---

# Python Basics: Variables and Data Types

```python
# Integer
age = 25

# Float
temperature = 98.6

# String
name = "Earth"

# Boolean
is_planet = True

# List
planets = ["Mercury", "Venus", "Earth", "Mars"]

# Dictionary
planet_info = {
    "name": "Earth",
    "diameter": 12742,
    "has_atmosphere": True
}

# Print variables
print(f"Age: {age}")
print(f"Temperature: {temperature}")
print(f"Name: {name}")
print(f"Is planet: {is_planet}")
print(f"Planets: {planets}")
print(f"Planet info: {planet_info}")
```

---

# Python Basics: Arithmetic Operations

```python
# Addition
sum = 5 + 3
print(f"5 + 3 = {sum}")

# Subtraction
difference = 10 - 4
print(f"10 - 4 = {difference}")

# Multiplication
product = 6 * 7
print(f"6 * 7 = {product}")

# Division
quotient = 20 / 4
print(f"20 / 4 = {quotient}")

# Integer division
int_quotient = 20 // 3
print(f"20 // 3 = {int_quotient}")

# Modulo (remainder)
remainder = 20 % 3
print(f"20 % 3 = {remainder}")

# Exponentiation
power = 2 ** 3
print(f"2 ** 3 = {power}")
```

---

# Python Basics: Control Flow

## If-Else Statements

```python
temperature = 25

if temperature > 30:
    print("It's hot outside!")
elif temperature > 20:
    print("It's warm outside.")
else:
    print("It's cool outside.")
```

## For Loops

```python
planets = ["Mercury", "Venus", "Earth", "Mars"]

for planet in planets:
    print(f"{planet} is a planet in our solar system.")
```

---

# Python Basics: Functions

```python
def celsius_to_fahrenheit(celsius):
    fahrenheit = (celsius * 9/5) + 32
    return fahrenheit

# Using the function
temp_c = 25
temp_f = celsius_to_fahrenheit(temp_c)
print(f"{temp_c}°C is equal to {temp_f}°F")
```

---

# Conclusion

- We've covered the course overview and tools
- Introduced basic Python concepts
- Next class: Numpy & Pandas for data analysis

Questions?

