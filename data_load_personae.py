import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

## Creating embeddingz section

# GPU check
print(f"Using GPU? {torch.cuda.is_available()}")
print(f"GPU used (if at all): {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Load data - OBSERVATORY (highest sample size, but requires TailScale system-1 access)
dataset = load_dataset("SimulaMet/moltbook-observatory-archive", "posts")
print(f"Dataset length: {len(dataset)}")
print(f"Loaded data structure from HF: {dataset}")

# Extract the archive split - this is just for because I found that the 'dataset' originally is a dictionary because of data structure.
df = dataset["archive"].to_pandas()

texts = (df["title"].fillna("") + " " + df["content"].fillna("")).str.strip().tolist() # Etracting content
print(f"\nFirst entry of input to embedding model: {texts[0]}")
print(f"Number of embeddingz: {len(texts)}")


## NOTE - UNCOMMENT THIS SECTION IF NEW EMBEDDINGS NEEDING TO BE GENERATED. OTHERWISE, THE EMBEDDINGS ARE SAVED UNDER moltbook_embeddings.py on oscar@system-1.
# Load model - NOTE that this is NOT training the model, it is pretrained ! The encode() function essentially just converts the 
#model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
#print(f"\nLoading embeddings with model f{model}. Make sure CUDA is bein utilised!!")

# Generate embeddingz
#embeddings = model.encode(texts, batch_size=1024, show_progress_bar=True, normalize_embeddings=True, device=device)

# Save 
#print(f"We expect embeddings shape in 384-D space. Shape of embeddings is: {embeddings.shape}")
#print("Writing matrix to disk: moltbook_embeddings.npy...")
#np.save("moltbook_embeddings.npy", embeddings)
#print("Saved successfully!")

## KMeansCluster section - take centroids of subset of data with MiniBatch class and minimise distances to these centroids to figure out where these clusters take place.

# We expect from the Personas paper (Amin et al.) for the clusters to be k=5 but don't trust it fully so are replicating via the 'Silhoutte' sweep below.

from sklearn.cluster import MiniBatchKMeans

embeddings = np.load("moltbook_embeddings.npy") # CHANGE relative to where embeddings are stored.

# fit k-means at k=5 to start (matching Amin et al.)
n_init = 10 # Depending on runtime on system-1 or other, this can be increased to take more random states of minibatches.

kmeans = MiniBatchKMeans(n_clusters=5, random_state=42, n_init=n_init)
labels = kmeans.fit_predict(embeddings) # Fit the MiniBatch to embeddings
centroids = kmeans.cluster_centers_

# Run over the 3-8 kmeans cluster parameter of k and optimise the SILHOUTTE SCORE - search up, but essentially this is the quality of the clustering/separation:

from sklearn.metrics import silhouette_score
SAMPLE_SIZE = 50_000 # NB this should be changed again if the runtime is sufficient, but this allows for fast comparisons with representativeness. 

# Random sampling:

idx = np.random.choice(len(embeddings), SAMPLE_SIZE, replace=False)
sample = embeddings[idx]

# sweep k=3 to k=8
scores = {}
for k in range(3, 9):
    print(f"Fitting k={k}...")
    km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(sample)
    score = silhouette_score(sample, labels, metric="cosine")
    scores[k] = round(score, 3)
    print(f"  k={k} silhouette={score:.3f}") 

print("NOTE that our Amin et al. paper has cosine = 0.624 with k=5 as the max value. Refer to the paper and plot the k vs cosine if you want to fully compare.")

# find the optimal
best_k = max(scores, key=scores.get)
print(f"\nOptimal k={best_k} (score={scores[best_k]})")
print(f"Full scores: {scores}")

## FINAL STEP - for input into Moltbook, we want to extract a certain number of samples (again this parameter can be optimised relative to runtime / what is needed for MiroFish)

# Take a 'n_posts' number of samples for each cluster. We could inspect manually, or put them into an LLM as with prior papers.

