import numpy as np
import skimage
from PIL import Image

im = Image.open("dc_img.jpg")
im = im.convert("L")
im = im.save("Gray.jpg")

im = Image.open("Gray.jpg")
im = im.resize([224,224])
pixels = im.load()
x = np.asarray(im)

img_blocks = skimage.util.view_as_blocks(x, block_shape=(8, 8))

H1 = [[0.5,0,0,0,0.5,0,0,0],
      [0.5,0,0,0,-0.5,0,0,0],
      [0,0.5,0,0,0,0.5,0,0],
      [0,0.5,0,0,0,-0.5,0,0],
      [0,0,0.5,0,0,0,0.5,0],
      [0,0,0.5,0,0,0,-0.5,0],
      [0,0,0,0.5,0,0,0,0.5],
      [0,0,0,0.5,0,0,0,-0.5]
      ]

H2 = [
      [0.5,0,0.5,0,0,0,0,0],
      [0.5,0,-0.5,0,0,0,0,0],
      [0,0.5,0,0.5,0,0,0,0],
      [0,0.5,0,-0.5,0,0,0,0],
      [0,0,0,0,1,0,0,0],
      [0,0,0,0,0,1,0,0],
      [0,0,0,0,0,0,1,0],
      [0,0,0,0,0,0,0,1]
      ]

H3 = [[0.5,0.5,0,0,0,0,0,0],
      [0.5,-0.5,0,0,0,0,0,0],
      [0,0,1,0,0,0,0,0],
      [0,0,0,1,0,0,0,0],
      [0,0,0,0,1,0,0,0],
      [0,0,0,0,0,1,0,0],
      [0,0,0,0,0,0,1,0],
      [0,0,0,0,0,0,0,1]
      ]
H1 = np.matrix(H1)
H2 = np.matrix(H2)
H3 = np.matrix(H3)
H = np.matmul(np.matmul(H1,H2),H3)
new_img_blocks = [[0 for j in range(8*8)] for i in range(28*28)]
for i in range(len(img_blocks)): 
    for j in range(len(img_blocks)):
        new_img_blocks[i][j] = np.matmul(np.matmul(H.T,img_blocks[i][j]),H)
        
decomp_blocks = skimage.util.view_as_blocks(x, block_shape=(1, 1))
dec = [[] for i in range(224)]
for i in range(len(decomp_blocks)):
    d = [0 for i in range(224)]
    for j in range(len(decomp_blocks)):
        d[j]=decomp_blocks[i][j][0][0].tolist()
    dec[i] = d
dec = np.asarray(dec)
im = Image.fromarray(dec)
im = im.convert("L")
#dup_img = Image.new(im.mode,im.size)
#pixel_new = dup_img.load()
#pixel_new = im.load()
im.show()
im.save('Compressed.jpg')