# Please install super-image dataset: pip install datasets super-image

from datasets import load_dataset

num_proc = 8 # Number of CPU Cores // 2
#out_dir = 

dataset = load_dataset('eugenesiow/Div2k', num_proc=4)