import numpy as np


"""Sample konvolucija: 
Kanali ulazne slike (input_depth) : 64
Feature mape (output_depth) : 64
Broj slika (batchsize) : 64 
Velicina slike : 24,24
Velicina kernela : 3,3
Broj operacija (forward_backward_operations) : 3 -> Forward propagation koristi 1 conv, Backward koristi 2

number of raw images : Broj 2d slika koje moraju proc kroz konvoluciju (24x24)"""

batchsize = 64
input_depth = 64 
output_depth = 64 
forward_backward_operations = 3
numberofrawimages = input_depth*output_depth*forward_backward_operations*batchsize

images = np.random.randn(batchsize,input_depth,24,24)
kernels = np.random.randn(input_depth,output_depth,3,3)
output = np.zeros((batchsize,output_depth,24,24))
print("\n")


#d monster
for operation in range(forward_backward_operations):
    for idx,image_channels in enumerate(images):
        for depthidx,image in enumerate(image_channels):
            for j in range(output_depth):
                for k in range(image.shape[0]):
                    for l in range(image.shape[1]):
                        for m in range(kernels.shape[2]):
                            for n in range(kernels.shape[3]):
                                output[idx,j,k,l] += image[k,l] * kernels[depthidx,j,m,n]

            print(f"\rOperation {depthidx}\{numberofrawimages} done , {depthidx/numberofrawimages:.4f}%",end="")
                            

                    
    