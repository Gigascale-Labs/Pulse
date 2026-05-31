import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import time

## CREATING embeddingz section

import torch 
print("Using GPU?", torch.cuda.is_available())
print("GPU used (if at all):", torch.cuda.get_device_name(0))
device = "cuda" 

# Load dataset 
from datasets import load_dataset

print('...')
print('')

dataset = load_dataset("SimulaMet/moltbook-observatory-archive", "posts")
print(f"Loaded data structure from HF: {dataset}")

# Extract the archive split - this is just for because I found that the 'dataset' originally is a dictionary because of data structure.
df = dataset["archive"].to_pandas()
df = df.iloc[:2894804].reset_index(drop=True) # Manually updated so that it just keeps the posts that have saved embeddings

df["text"] = (df["title"].fillna("") + " " + df["content"].fillna("")).str.strip().tolist() # Etracting content
texts = df["text"].tolist()

print('')
print('...')
print('')

print("First entry of input to embedding model:", texts[0])
print("Number of posts (and therefore expected embeddingz):", len(texts))
print("I fix the number of loaded posts at 2894804 as the dataset continuously updates.")

print('')
print('...')
print('')


## Embed with MiniLM on GPU
## NOTE - UNCOMMENT THIS SECTION IF NEW EMBEDDINGS NEEDING TO BE GENERATED. OTHERWISE, THE EMBEDDINGS ARE SAVED UNDER moltbook_embeddings.py on oscar@system-1.

# Load model - NOTE that this is NOT training the model, it is pretrained ! The encode() function essentially just converts the 
#model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
#print(f"\nLoading embeddings with model f{model}. Make sure CUDA is bein utilised!!")

# Generate embeddingz
#embeddings = model.encode(texts, batch_size=1024, show_progress_bar=True, normalize_embeddings=True, device=device)

# Save 
#print(f"We expect embeddings shape in 384-D space. Shape of embeddingz is: {embeddings.shape}")
#print("Writing matrix to disk: moltbook_embeddings.npy...")
#np.save("moltbook_embeddings.npy", embeddings)
#print("Saved successfully!")

print('')
print('...')
print('')

## KMeansCluster section - take centroids of subset of data with MiniBatch class and minimise distances to these centroids to figure out where these clusters take place.
## We expect from the Personas paper (Amin et al.) for the clusters to be k=5 but don't trust it fully so are replicating via the 'Silhoutte' sweep below.

from sklearn.cluster import MiniBatchKMeans

embeddings = np.load("moltbook_embeddings.npy") # CHANGE relative to where embeddings are stored.
print('Embeddings loaded successfully, so we can Now moving on to clustering and persona extraction...')
print(f"Shape of loaded embeddingz: {embeddings.shape}")

print('')
print('...')
print('')

time.sleep(2)

## ADDED - data cleaning. We could see that from silhoutte scores/clustering that we need to remove some noise:
# Removing short posts (eg less than 10 words) did not work - doesn't seem to contain any short posts.
# Other cleaning methods could be removing stop words, but this is not necessarily ideal for the embedding model. 
#Finally tried to remove non-english posts using LANGDETECT (slow), but I'm just cuatious again this may remove some of the nuance that the embedding model can pick up on. 

# One method that worked, once clusters were examined, was filtering out the spam from the dataset:
import re

def is_spam(text):
    patterns = [
        r'"op"\s*:\s*"mint"',
        r'"op"\s*:\s*"link"',
        r'"p"\s*:\s*"mbc-20"',
        r'0x[a-fA-F0-9]{40}',
        r'mbc20\.xyz',
        r'"tick"\s*:\s*"(GPT|CLAW|MOLT)"',
    ]
    return any(re.search(p, text) for p in patterns)

spam_mask = ~df["text"].apply(is_spam)
clean_indices = np.where(spam_mask.values)[0]

df = df[spam_mask].reset_index(drop=True)
embeddings = embeddings[clean_indices]

print(f"After spam filter: {len(df):,} posts")
print(f"Embeddings shape: {embeddings.shape}")
#From now, we will proceed with the raw text and see how it performs in the clustering step - FUTURE RESEARCH SHOULD COME BACK.


print('')
print('...')
print('')


# Run over the 3-8 kmeans cluster parameter of k and optimise the SILHOUTTE SCORE - search up, but essentially this is the quality of the clustering/separation:
from sklearn.metrics import silhouette_score
SAMPLE_SIZE = 100000 # This was upscaled from 50000 to 250000, but still didn't significantly increase silhoutte scores.
# NB SAMPLE_SIZE should be changed again if the runtime is sufficient, but this allows for fast comparisons with representativeness. 
# Random sampling:
idx = np.random.choice(len(embeddings), SAMPLE_SIZE, replace=False)
sample = embeddings[idx]

# Added component to do away w silhoutte score problems; apply a UMAP = dimensionality reduction.
# This could significantly reduce embedding collapse:
import umap
reducer = umap.UMAP(n_components=50, metric='cosine')
embeddings_50d = reducer.fit_transform(sample) # fit on to NUM_SAMPLES
print("Applied UMAP dimensionality reduction to 50D space for better clustering performance.")
time.sleep(2)
print("we expect the shape of the reduced embeddingz to be (SAMPLE_SIZE, 50). Shape is:", embeddings_50d.shape)

print('')
print('...')
print('')

# sweep k=3 to k=8
scores = {}
for k in range(3, 9):
    print(f"Fitting k={k}...")
    km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings_50d) # Fit to the REDUCED embeddingz
    score = silhouette_score(embeddings_50d, labels, metric="cosine")
    scores[k] = round(score, 3)
    print(f"  k={k} silhouette={score:.3f}") 

# find the optimal
best_k = max(scores, key=scores.get)
print(f"\nOptimal k={best_k} (score={scores[best_k]})")
print(f"Full scores OF EACH : {scores}")


print('Now optimal k has been found, we can fit the model to the FULL DATASET')
print('')
print('...')
print('')

# fit k-means to the full dataset with the optimal k
n_init = 10 # Depending on runtime on system-1 or other, this can be increased to take more random states of minibatches.
embeddings_50d_full = reducer.transform(embeddings) # Apply the same UMAP transformation to the FULL embeddingz
embeddings = embeddings_50d_full # For the rest of the code, we will work with ze reduced embeddingz. This has been shown in lit to still preserve local neighbourhoods and therefore should be fine for the clustering step.

# then fit k-means on the reduced version
kmeans = MiniBatchKMeans(n_clusters=best_k, random_state=42, n_init=n_init) #Fit to the swept best k
labels = kmeans.fit_predict(embeddings) # Fit the MiniBatch to embeddings
centroids = kmeans.cluster_centers_ # For similarity calculations later on ...
print(f"Fitted MiniBatchKMeans with k={best_k} to the FULL dataset.")

print(' ')
print('...')
print('')


## FINAL STEP - for input into Moltbook, we want to extract a certain number of samples (again this parameter can be optimised relative to runtime / what is needed for MiroFish)
# Take a 'n_posts' number of samples for each cluster. We could inspect manually, or put them into an LLM as with prior papers.

n_posts = 500 # Change if needed - for now just want to get this running and into a .txt file per persona.

from sklearn.metrics.pairwise import cosine_similarity # Our cosine similarity here works to calculate our distances from centroidz, and extract most representative post.
# In order to do this, we need to map it back onto the original 'texts':


for k in range(kmeans.n_clusters):
    print(f"\nCluster {k}:") # Each cluster being representative of a persona
    
    # get indices of all posts in this cluster
    cluster_mask = labels == k # Boolean 'masking' method - this allows us to map onto k < clusters
    cluster_indices = np.where(cluster_mask)[0]
    cluster_embeddings = embeddings[cluster_indices] 
    # from numpy docs - https://numpy.org/doc/stable/user/basics.indexing.html#boolean-array-indexing
    
    # compute cosine similarity of each post to its cluster centroid
    centroid = centroids[k].reshape(1, -1)
    similarities = cosine_similarity(cluster_embeddings, centroid).flatten() # simple 'dot-product'-like method to calculate the similarity of each post to the centroid. The higher the similarity, the more representative the post is of the cluster/persona.
    
    # get top n_posts most similar to centroid - based on the same cosine similarity
    top_local_idx = np.argsort(similarities)[::-1][:n_posts] # Argsort for the most similar posts within each of the clusters.
    top_global_idx = cluster_indices[top_local_idx]
    
    top_posts = df.iloc[top_global_idx][["title", "content"]].copy()
    top_posts["similarity"] = similarities[top_local_idx]
    
    print(f"  Total posts in cluster: {cluster_mask.sum():,}")
    print(f"  MININUM cos-similarity score: {similarities[top_local_idx[-1]]:.3f}")
    for title in top_posts["title"].head(5).tolist():
        print(f"    - {title[:80]}")
    
    # write to txt file - FOR STEPHEN'S MiroFish sim
    out_path = f"test_cluster_{k}_posts.txt"
    with open(out_path, "w") as f:
        for i, row in enumerate(top_posts.itertuples()):
            f.write(f"POST {i+1}:\n")
            f.write(f"Title: {row.title}\n")
            f.write(f"Content: {row.content}\n\n")

    time.sleep(3)
    
    print(f"  Saved {out_path}")
