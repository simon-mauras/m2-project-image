#!/usr/bin/env python3

from scipy.misc import imread, imsave
from scipy.sparse import coo_matrix as matrix
from scipy.sparse.linalg import cg as solve
from time import time
import numpy as np
import matplotlib.pyplot as plt


background = imread("data/background.png", mode="RGB")
foreground = imread("data/foreground.png", mode="RGB")
mask = imread("data/foreground-mask.png", mode="L")
delta = (0,200)

border = []
interior = []
for i in range(mask.shape[0]):
  for j in range(mask.shape[1]):
    if mask[i][j] > 128:
      inside = True
      for (di,dj) in [(-1,0),(1,0),(0,-1),(0,1)]:
        if mask[i+di][j+dj] <= 128 : inside = False
      if inside:
        interior.append((i,j))
      else:
        border.append((i,j))

id_to_coord = border + interior
size = len(id_to_coord)

coord_to_id = dict()
for i in range(size):
  coord_to_id[id_to_coord[i]] = i

A = []
b = [[],[],[]]
for c in range(3):
  row = []
  col = []
  data = []
  id_row = 0
  for (i,j) in border:
    b[c].append(background[i+delta[0]][j+delta[1]][c])
    data.append(1)
    row.append(id_row)
    col.append(coord_to_id[(i,j)])
    id_row = id_row + 1
  for (i,j) in interior:
    data.extend([4,-1,-1,-1,-1])
    row.extend([id_row] * 5)
    col.append(coord_to_id[(i,j)])
    value = 4 * foreground[i][j][c]
    for (di,dj) in [(-1,0),(1,0),(0,-1),(0,1)]:
      col.append(coord_to_id[(i+di,j+dj)])
      value = value - foreground[i+di][j+dj][c]
    b[c].append(value)
    id_row = id_row + 1
  A.append(matrix((data,(row,col)), shape=(id_row,size)))

def display(img, colors=[0,1,2]):
  tmp = img.copy()
  for c in range(3):
    if c not in colors:
      tmp[:,:,c] = 0
  plt.imshow(tmp)
  plt.show()
  

for nb_iter in range(0,5):
  print("Let's solve for maxiter=%d" % (10**nb_iter))
  for c in range(3):
    starting_time = time()
    x = solve(A[c], b[c], maxiter=10**nb_iter)
    ending_time = time()
    print("Done in %.3lfs" % (ending_time - starting_time))
    for (i,j) in interior:
      v = max(0, min(x[0][coord_to_id[(i,j)]], 255))
      background[i+delta[0]][j+delta[1]][c] = v
  display(background)
  #imsave("results/%d.png" % (10**nb_iter), background)

