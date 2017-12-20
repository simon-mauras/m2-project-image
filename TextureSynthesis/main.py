#!/usr/bin/env python3

from scipy.misc import imread, imsave
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from time import time
import numpy as np
import matplotlib.pyplot as plt
import sys

clusters_n = 1
clusters_init = "random"

clusters_n = 4
clusters_init = np.array([(0,0,0),(255,0,0),(0,255,0),(0,0,255)])

def smaller(img):
  shape = list(img.shape)
  shape[0] = shape[0] // 2
  shape[1] = shape[1] // 2
  result = np.ndarray(shape, dtype=img.dtype)
  for i in range(shape[0]):
    for j in range(shape[1]):
      result[i][j] = (0. + img[2*i][2*j] + img[2*i+1][2*j]  \
                         + img[2*i][2*j+1] + img[2*i+1][2*j+1]) / 4
  return result

def normalize(mask):
  height,width = mask.shape
  for i in range(height):
    for j in range(width):
      if mask[i][j] > 0:
        mask[i][j] = 255

delta = [ (i,j) for i in [-1,0,1] for j in [-1,0,1] if (i,j) != (0,0) ]

def solve(image, hint, mask):
  height,width = mask.shape
  
  if height < 10 or width < 10:
    imsave("results/%dx%d_input.png" % (height,width), image)
    imsave("results/%dx%d_mask.png" % (height,width), mask)
    imsave("results/%dx%d_hint.png" % (height,width), hint)
    imsave("results/%dx%d_output.png" % (height,width), image)
    #plt.imshow(image, interpolation="none")
    #plt.show()
    return image
  
  image2 = smaller(image)
  hint2 = smaller(hint)
  mask2 = smaller(mask)
  normalize(mask2)
  result2 = solve(image2, hint2, mask2)
  
  print("--------------------------------")
  print("Solving %d x %d" % (height,width))
  
  imsave("results/%dx%d_input.png" % (height,width), image)
  plt.imshow(image, interpolation="none")
  plt.show()
  
  imsave("results/%dx%d_mask.png" % (height,width), mask)
  plt.imshow(mask, interpolation="none")
  plt.show()
  
  print("Computing best guess")
  starting_time = time()
  result = image.copy()
  for i in range(height):
    for j in range(width):
      if mask[i,j] > 0:
        result[i,j] = result2[i//2,j//2]
  ending_time = time()
  print("Done in %.3lfs" % (ending_time - starting_time))
  
  imsave("results/%dx%d_guess.png" % (height,width), result)
  plt.imshow(result, interpolation="none")
  plt.show()
  
  print("Clustering hint")
  starting_time = time()
  hint_id = np.zeros(shape=(height, width), dtype=np.uint32)
  hint_vector = []
  for i in range(0,height):
    for j in range(0,width):
      hint_id[i,j] = len(hint_vector)
      hint_vector.append(hint[i,j])
  kmeans = KMeans(n_clusters=clusters_n,init=clusters_init).fit(hint_vector)
  for i in range(0,height):
    for j in range(0,width):
      hint_id[i,j] = kmeans.labels_[hint_id[i,j]]
  ending_time = time()
  print("Done in %.3lfs" % (ending_time - starting_time))
  
  imsave("results/%dx%d_hint.png" % (height,width), hint)
  imsave("results/%dx%d_clusters.png" % (height,width), hint_id)
  plt.imshow(hint, interpolation="none")
  plt.show()
  plt.imshow(hint_id, interpolation="none")
  plt.show()
  
  print("Building dictionary")
  starting_time = time()
  dict_coord = [ [] for i in range(clusters_n) ]
  dict_vector = [ [] for i in range(clusters_n) ]
  dict_kdtree = [ None for i in range(clusters_n) ]
  for i in range(1,height-1):
    for j in range(1,width-1):
      vector = []
      valid = mask[i,j] == 0
      for di,dj in delta:
        if mask[i+di,j+dj] > 0:
          valid = False
        vector.extend(image[i+di,j+dj])
      vector = tuple(map(float, vector))
      if valid:
        dict_coord[hint_id[i,j]].append((i,j))
        dict_vector[hint_id[i,j]].append(vector)
  for i in range(clusters_n):
    dict_coord[i] = np.array(dict_coord[i])
    dict_vector[i] = np.array(dict_vector[i])
    dict_kdtree[i] = cKDTree(dict_vector[i], leafsize=100)
  ending_time = time()
  print("Done in %.3lfs" % (ending_time - starting_time))

  print("Computing solution")
  starting_time = time()
  for i in range(1,height-1):
    for j in range(1,width-1):
      if mask[i,j] > 0:
        vector = []
        for di,dj in delta:
          vector.extend(result[i+di,j+dj])
        vector = tuple(map(float, vector))
        best_id = dict_kdtree[hint_id[i,j]].query([vector])[1][0]
        i2,j2 = dict_coord[hint_id[i,j]][best_id]
        result[i,j] = image[i2,j2]
  ending_time = time()
  print("Done in %.3lfs" % (ending_time - starting_time))
  
  imsave("results/%dx%d_output.png" % (height,width), result)
  plt.imshow(result, interpolation="none")
  plt.show()
  
  return result

image = imread("data/image.png", mode="RGB")
hint = imread("data/hint.png", mode="RGB")
mask = imread("data/mask.png", "L")

result = solve(image, hint, mask)
#plt.imshow(result)
#plt.show()

