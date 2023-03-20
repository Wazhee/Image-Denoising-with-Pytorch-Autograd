import torch
import numpy as np
import torch.nn.functional as F


"""
Returns the x and y gradient images for input image I. 
Input:
I: (Tensor) Image of shape (H, W, 3)

Output:
(Ix, Iy): (Tensor) Gradient images each of shape (H, W, 3)
"""
def get_spatial_gradients(I):
  I = I.permute(2, 0, 1).unsqueeze(0) # Change I's shape from (H, W, 3) to (1, 3, H, W)
  kx = torch.zeros(I.shape[1], I.shape[1], 3, 3).to(I.device)
  ky = torch.zeros(I.shape[1], I.shape[1], 3, 3).to(I.device)

  for i in range(3):
    kx[i, i, 1, 1] = -1
    kx[i, i, 1, 2] = 1
    ky[i, i, 1, 1] = -1
    ky[i, i, 2, 1] = 1

  Ix = F.conv2d(I, kx, padding=1)
  Iy = F.conv2d(I, ky, padding=1)
  return Ix[0,...].permute(1,2,0), Iy[0,...].permute(1,2,0)


"""
Loss with L2 norm 

Denoising objective function.
Input:
I, J: (Tensor) Images of shape (H, W, 3)
alpha: (float) Regularization hyperparameter

Output:
loss: (float)
"""
def denoising_L2_loss(I, J, alpha):
  djdx, djdy = get_spatial_gradients(J) # code for 1.2a.
  ij = torch.square(I-J).sum()
  dj = torch.square(djdx).sum() + torch.square(djdy).sum() 
  loss = ij + alpha*dj
  return loss


def main():
  img = imageio.imread('/content/drive/MyDrive/Comp 546: Computer Vision/parrot_noisy.png')/255.0
  I = torch.Tensor(img)
  I = I.to('cuda')

  lr = 1 # Learning rate
  alpha = 2 # alpha
  n_iter = 3000 # Number of iterations
  J = I.clone()

  for i in tqdm(range(n_iter)):
    # Your code for 1.2b goes here
    J.requires_grad_(True)
    loss = denoising_L2_loss(I, J, alpha)
    loss.requires_grad_(True)
    loss.backward()
    with torch.no_grad():
      J_plus = J - (alpha*F.normalize(J.grad))
      J.grad.zero_()
      J = J_plus
      
 
if __name__ == "__main__":
  print("Denoising image with L2 Norm...")
  main()
