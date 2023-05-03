import matplotlib.pyplot as plt
import numpy as np

# Generate some data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create a figure and axis object
fig, ax = plt.subplots()

# Plot the data
ax.plot(x, y)

# Set the x and y axis labels and title
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Sin(x)')

# Show the plot
plt.show()