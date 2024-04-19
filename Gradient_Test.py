import matplotlib.pyplot as plt
from perlin_noise import PerlinNoise

noise = PerlinNoise(octaves=10, seed=1)
xpix, ypix = 100, 100
pic = [[noise([i/xpix, j/ypix]) for j in range(xpix)] for i in range(ypix)]

plt.imshow(pic, cmap='gray')
#plt.show()

# Create data for the grid
x = []
y = []
for i in range(10):
    for j in range(10):
        x.append(i)
        y.append(j)

# Plot the grid of points
plt.figure(figsize=(8, 6))  # Adjust figure size as needed
plt.scatter(x, y)
plt.title('10x10 Grid of Points')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()
