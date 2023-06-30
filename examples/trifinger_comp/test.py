import matplotlib.pyplot as plt
import numpy as np

# Create some data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create the scatter plot
plt.scatter(x, y, c=y, cmap='cool')

# Add a colorbar
plt.colorbar()

# Show the plot
plt.show()
