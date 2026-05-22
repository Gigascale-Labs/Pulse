import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

import torch 
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# Load data - using observatory dataset
from datasets import load_dataset
df = load_dataset("SimulaMet/moltbook-observatory-archive", "posts")
print(f"Loaded {len(df)} posts")
  # see what columns exist

