
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

cart_coord = pd.read_csv('3d_data_0.csv').to_numpy()[1:,1:]



fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(projection='3d')

ax.scatter3D(cart_coord[:,0],cart_coord[:,1],cart_coord[:,2],s=0.7,c='red')
# ax.scatter3D(cart_coord_2[:,0],cart_coord_2[:,1],cart_coord_2[:,2],s=0.7,c='blue')

plt.show()
print(cart_coord.shape)





