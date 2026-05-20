# First we need to check if the GPU is available. Will be very difficult for you to load all of this data
import torch
print("GPU Available: ", torch.cuda.is_available())

import pandas as pd
from datasets import load_dataset

ds = load_dataset("SimulaMet/moltbook-observatory-archive", "posts")

# Take a look at first few rows of the dataset for verification: should look like "Hello Moltobook from HF"
print(ds["title"][0:5])

# Now we need to create the embeddings from the "content" column of the dataset. 
# From the original paper/methodology, We will use the Hugging Face Transformers library to create the embeddings. We will use the "sentence-transformers/all-MiniLM-L6-v2" model for this purpose.
import transformers
from transformers import AutoTokenizer, AutoModel

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Function to create embeddings
def create_embeddings(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # Get the model's output
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Mean pooling to get a single vector representation
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
    
    return embeddings

# Now we will create embeddings for the "content" column of the dataset and store them in a new column called "embeddings"
ds = ds.map(lambda x: {"embeddings": create_embeddings(x["content"]).numpy()}, batched=True)
# Finally, we will save the dataset with the new embeddings column to a new file for later use
ds.save_to_disk("moltbook_observatory_with_embeddings")
# %%


