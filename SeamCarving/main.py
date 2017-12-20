#!/usr/bin/env python3

from scipy.misc import imread, imsave
from time import time
import numpy as np
import matplotlib.pyplot as plt

image = imread("data/image.png", mode="RGB")

def gradient(img):
  height,width,_ = img.shape
  result = np.zeros(shape=(height, width))
  for i in range(1, height-1):
    for j in range(1, width-1):
      for c in range(3):
        vi = float(img[i+1,j,c]) - float(img[i-1,j,c])
        vj = float(img[i,j+1,c]) - float(img[i,j-1,c])
        result[i,j] = result[i,j] + abs(vi) + abs(vj)
  return result

def dp(cost):
  height,width = cost.shape
  dyn = np.zeros(shape=(height, width))
  for i in range(1, height):
    dyn[i,0] = float("inf")
    dyn[i,width-1] = float("inf")
    for j in range(1, width-1):
      dyn[i,j] = cost[i,j] + min([dyn[i-1,j-1], dyn[i-1,j], dyn[i-1,j+1]])
  remove = [0]
  for j in range(1, height):
    if dyn[height-1][j] < dyn[height-1][remove[0]]:
      remove[0] = j
  for i in reversed(range(height-1)):
    next = 0
    for j in [-1,0,1]:
      if dyn[i+1][remove[-1]] == dyn[i][remove[-1]+j] + cost[i+1][remove[-1]]:
        next = remove[-1]+j
    remove.append(next)
  return list(reversed(remove))

plt.imshow(image)
plt.show()

print("Compute cost function")
starting_time = time()
cost = gradient(image)
ending_time = time()
print("Done in %.3lfs" % (ending_time - starting_time))
imsave("results/cost_init.png", cost)
plt.imshow(cost, cmap='gray')
plt.show()

print("Dynamic programming")
starting_time = time()
remove = []
max_value = cost.max()
for repeat in range(10):
  height,width = cost.shape
  remove.append(dp(cost))
  for i in range(height):
    cost[i][remove[-1][i]] = float("inf")
cost = np.minimum(max_value, cost)
ending_time = time()
print("Done in %.3lfs" % (ending_time - starting_time))
imsave("results/cost_remove.png", cost)
plt.imshow(cost, cmap='gray')
plt.show()

remove = list(map(list, zip(*remove)))

new_image = []
for i in range(len(remove)):
  r = np.delete(image[i,:,0], remove[i])
  g = np.delete(image[i,:,1], remove[i])
  b = np.delete(image[i,:,2], remove[i])
  new_image.append(np.transpose([r,g,b]))
new_image = np.array(new_image)

imsave("results/image_input.png", image)
imsave("results/image_output.png", new_image)
plt.imshow(new_image)
plt.show()

