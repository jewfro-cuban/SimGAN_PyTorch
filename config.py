import torch

# Is the PC has cuda
cuda_use = torch.cuda.is_available()
# which cuda to use
cuda_num = 0

init_lr = 0.001
# lambda in paper, the author of the paper said it's 0.01
delta = 0.0001
img_width = 55
img_height = 35
img_channels = 1

# synthetic image path
syn_path = 'data/SynthEyes'
# real image path
real_path = 'data/MPIIGaze'


# =================== training params ======================
# pre-train R times
g_pretrain = 1000
# pre-train D times
d_pretrain = 100
# train steps
train_steps = 10000

batch_size = 128
# the history buffer size
buffer_size = 12800

k_d = 1  # number of discriminator updates per step
k_g = 10  # number of generative network updates per step, the author of the paper said it's 50

# output R pre-training result per times
r_pre_per = 50
# output D pre-training result per times
d_pre_per = 20
# save model dictionary and training dataset output result per train times
save_per = 1000

# file root
save_path = 'models'

# dictionary saving path
D_path = 'D_%d.pkl'
R_path = 'R_%d.pkl'

optimizer_path = 'optimizer_status.pkl'