# Diffusion-Models
https://lilianweng.github.io/posts/2021-07-11-diffusion-models/



# Training process:
"""
Corresponds to Algorithm 1 from (Ho et al., 2020).
"""

def get_loss(self batch, batch_idx):

  # Get a random time step for each image in the batch
  ts = torch.randint(0, self.t_range, [batch.shape[0]], device=self.device)
  noise_imgs = []
  # Generate noise, one for each image in the batch
  epsilons = torch.randn(batch.shape, device=self.device)
  
  for i in range(len(ts)):
    a_hat = self.alpha_bar(ts[i])
    noise_imgs.append(
      (math.sqrt(a_hat) * batch[i]) + (math.sqrt(1 - a_hat) * epsilons[i])
    )
  
  noise_imgs = torch.stack(noise_imgs, dim=0)
  # Run the noisy images through the U-Net, to get the predicted noise
  e_hat = self.forward(noise_imgs, ts)
  
  # Calculate the loss, that is, the MSE between the predicted noise and the actual noise
  loss = nn. functional.mse_loss(
    e_hat.reshape(-1, self.in_size), epsilons.reshape(-1, self.in_size)
  )
  return loss
