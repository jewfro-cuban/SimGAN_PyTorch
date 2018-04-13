# VisdomPortal
Portal for Visdom and PyTorch

# Requirements
+ PyTorch
+ Visdom
+ Numpy

# Usage
```
vis = VisdomPortal(env_name='test')
```
parameters:
+ env_name=some string
+ image_norm=[(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)] # similar to torchvision.transforms.normalize

# Draw curve
```
vis.draw_curve(value=loss, step=index, title='loss')
```
parameters:
+ value: torch.autograd.variable.Variable, int, float, or np.ndarray
+ step: int or numpy

# Draw multiple images
```
vis.draw_images(value=images, title='Images', unnormalize=True)
```
parameters:
+ value: 4-D torch.autograd.variable.Variable, torch._TensorBase, or np.ndarray
+ title: string
+ unnormalize: if un-normalize input values