
# coding: utf-8

# In[1]:

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from scipy import misc

# In[2]:


f = misc.imread('m67v.png')
X = 146
Y = 40
target=(Y,X)
subf = f[target[0]-15:target[0]+15,target[1]-15:target[1]+15]
h,w = subf.shape


# In[3]:


def onclick(event):
    target=(int(event.ydata),int(event.xdata))
    subf = f[target[0]-15:target[0]+15,target[1]-15:target[1]+15]
    ax.imshow(subf, cmap='gray')
    ax2.cla()
    ax2.plot(subf[:,15],np.arange(h,0,-1))
    ax3.cla()
    ax3.plot(subf[15,:])
    ax4.cla()
    ax4.imshow(f, cmap='gray')
    ax4.axvline(target[1], color='red')
    ax4.axhline(target[0], color='red')
    fig.canvas.draw()


# In[5]:
# matplotlib.rcParams["figure.facecolor"]='0.1'
# matplotlib.rcParams["figure.edgecolor"]='0.9'

fig = plt.figure(figsize=(10,10))
gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[3,1]) 
ax=plt.subplot(gs[0])
plt.imshow(subf, cmap='gray')
ax.axvline(15,color='lightsteelblue')
ax.axhline(15,color='lightsteelblue')
sig = plt.Circle((15,15),5,color='cyan', alpha=0.3)
back = plt.Circle((15,15),10, color='red', fill=False)
ax.add_artist(sig)
ax.add_artist(back)

ax2=plt.subplot(gs[1])
ax2.plot(subf[:,15],np.arange(h,0,-1))
plt.yticks([])

ax3=plt.subplot(gs[2])
ax3.plot(subf[15,:])
plt.gca().invert_yaxis()
plt.xticks([])

ax4=plt.subplot(gs[3])
ax4.imshow(f, cmap='gray')
ax4.axvline(target[1], color='red')
ax4.axhline(target[0], color='red')

plt.tight_layout()

for i in range(1):
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()
