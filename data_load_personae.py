import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import time # This is just used to make the output more interpretable and easier to track in the terminal.
import torch 

import random

from datasets import load_dataset # To Load dataset  

from sklearn.cluster import MiniBatchKMeans # For clustering - this is a faster version of KMeans, which is important given the size of the dataset.

import re # Filtering

from sklearn.metrics import silhouette_score # For evaluating clustering performance

#import umap # CHANGED to PCA:
from sklearn.decomposition import PCA

from sklearn.metrics.pairwise import cosine_similarity # Cosine sim for distance from centroids

import uuid # For generating unique IDs for each post, which can be useful for tracking and referencing posts in the dataset.

#import cuml - I had issues importing this myself, but it would be necessary for code/runtime optimisation
#import cudf

print("Using GPU?", torch.cuda.is_available())
print("GPU used (if at all):", torch.cuda.get_device_name(0))
print("List of devices being used (CHECK THIS CAREFULLY):", torch.cuda.device_count(), "GPUs being used:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

# BE CAREFUL to limit device usage if needed - use torch.cuda.set_device() if you want to specify a particular GPU, or set device to "cpu" if you want to force CPU usage.
device = "cuda" 

# I've kept these print statements so that they can be easily copied back to Python file - they are to make it easier to track the progress of the code in the terminal.

print('\n...\n')

response = input("Generate new embeddings? (T/F): ").strip().upper()
EMBED = response == "T"
print(f"EMBED set to: {EMBED}")
print("Make sure you have the correct T/F or it could either take way too long or overwrite your existing embeddings!")

response_embeddings = input("Enter a REASONABLE name + TODAY's DATE of the file to save/load embeddings. If you are loading, ensure to include the whole file name: ").strip()
print(f"Embeddings will be saved to/loaded from: {response_embeddings}")

# on the observatory dataset
dataset = load_dataset("SimulaMet/moltbook-observatory-archive", "posts")
df = dataset["archive"].to_pandas()
print(f"Dataset loaded! Dataset structure is: {' ,'.join(df.columns)}")

df["created_at"] = pd.to_datetime(df["created_at"])

# filter closer to AIcell date range - include a few more days
start = "2026-01-28"
end   = "2026-02-15"
df = df[(df["created_at"] >= start) & (df["created_at"] <= end)].reset_index(drop=True) # This is to change the date range, but also to prevent mismatches between embedding size and post size. (Observatory is always being updated.)

print(f"Posts in date window: {len(df):,}")

df["text"] = (df["title"].fillna("") + " " + df["content"].fillna("")).str.strip().tolist() # Etracting content
texts = df["text"].tolist()

print('\n...\n')


print("Head of the dataset (just to check it looks right):")
print(df.head(5)[["created_at", "title", "content"]])
print("Number of posts (and therefore expected embeddingz):", len(texts))
print('\n...\n')

if EMBED:
        ## Embed with MiniLM on GPU - if untrue and you just want to save embeddings, please put use the 'output' file name in the config, and then this is added to the system-1 datapool dir.
        # Load model - NOTE that this is NOT training the model, it is pretrained ! The encode() function essentially just converts the 
        model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        print(f"\nLoading embeddings with model MINI-LM Sentence transformer.")

        # Generate embeddingz
        embeddings = model.encode(texts, batch_size=2048, show_progress_bar=True, normalize_embeddings=True, device=device) # NOTE made - batch_size could be increased, but increasing to 5096 exceeds memory; the max value can be calculated but it's not exactly clear: https://www.reddit.com/r/LocalLLaMA/comments/1fkuuds/what_is_exactly_is_the_formula_to_calculate_vram/

        # Save 
        print(f"We expect embeddings shape in 384-D space. Shape of embeddingz is: {embeddings.shape}")

        # Generate unique suffix - this is so there's GUARANTEED to be no overwriting.
        unique_id = str(uuid.uuid4())[:8]

        # Target sys-1 dir
        target_dir = "/datapool/operational_data/dumps_personas" # CHANGE THIS TO TARGET DIRECTORY IN SYSTEM-1 - this is where I save the current sprint.

        # Combine everything into a single, clean path
        file_path = f"{target_dir}/{response_embeddings}_{unique_id}.npy" # From config

        # Save using the USER-DEFINED + unique path
        print(f"Writing matrix to disk: {file_path}...")

        # Hint: Pass your new file_path variable directly into np.save()
        np.save(file_path, embeddings)

        print('\n Embeddings saved. Moving on to dimensionality reduction on embeddings...\n')
else:
        file_path = f"/datapool/operational_data/dumps_personas/{response_embeddings}" # From config
        print(f"Loading embeddings from file: {file_path}...")
        embeddings = np.load(file_path)
        print("Loaded successfully!")
        print(f"Shape of loaded embeddingz: {embeddings.shape}")


SAMPLE_SIZE = 100000 # Should be more than enough to be representative, but also to run in a reasonable time frame. 
# NB SAMPLE_SIZE should be changed again if the runtime is sufficient, but this allows for fast comparisons with representativeness. 
# Random sampling:
random.seed(67)
idx = np.random.choice(len(embeddings), SAMPLE_SIZE, replace=False)
embeddings = embeddings[idx] # Work with this SAMPLE embeddings. but this should be changed if we are moving beyond a prototype.
# Apply the same idx filter to the dataframe to ensure they match - this is important for the later stages when we want to extract the most representative post from each cluster.
df = df.iloc[idx].reset_index(drop=True)


## ADDED, but optional - NOT USED IN FINAL PERSONAS.
# This appears after embeddings so that we don't have to regenerate every time.

# Log of attempted methods:
# Removing short posts (eg less than 10 words) did not work - doesn't seem to contain any short posts.
# Other cleaning methods could be removing stop words, but this is not necessarily ideal for the embedding model. 
#Finally tried to remove non-english posts using LANGDETECT (quite slow). 

# I've left this after embeddings are loaded so that additional data cleaning can be tried - the only working method I found was filtering out the spam, which is what I have included here. 
#This is also a method that can be easily adapted by devs if they want to try different cleaning methods -  can just change the function and keep the rest of the code the same.
# The following spam method tends to filter out 75% (!) of the dataset, but we really don't want any crypto agents if they are going to be interacting in MiroFish 
# - how could something that is outputting...
#  'POST 1: Title: Minting GPT - #00bj89cg Content: {"p":"mbc-20","op":"mint","tick":"GPT","amt":"100"}  mbc20.xyz' or 'POST 1:Title: wallet link #75869Content: {"p":"mbc-20","op":"link","wallet":"0x47895e645094ABf6aC6A622F557760AfE59C9Bf3"}
# ... have any place in opinion dynamics and GD?

# One method that made for cleaner separation between clusters was filtering out the spam from the dataset:
## ADDED, optional since there is less spam earlier in dataset
# - data cleaning - this appears after embeddings so that we don't have to regenerate every time.
# Removing short posts (eg less than 10 words) did not work - doesn't seem to contain any short posts.
# Other cleaning methods could be removing stop words, but this is not necessarily ideal for the embedding model. 
#Finally tried to remove non-english posts using LANGDETECT (quite slow). 

# I've left this after embeddings are loaded so that additional data cleaning can be tried - the only working method I found was filtering out the spam, which is what I have included here. 
#This is also a method that can be easily adapted by devs if they want to try different cleaning methods -  can just change the function and keep the rest of the code the same.
# The following spam method tends to filter out 75% (!) of the dataset, but we really don't want any crypto agents if they are going to be interacting in MiroFish 
# - how could something that is outputting...
#  'POST 1: Title: Minting GPT - #00bj89cg Content: {"p":"mbc-20","op":"mint","tick":"GPT","amt":"100"}  mbc20.xyz' or 'POST 1:Title: wallet link #75869Content: {"p":"mbc-20","op":"link","wallet":"0x47895e645094ABf6aC6A622F557760AfE59C9Bf3"}
# ... have any place in opinion dynamics and GD?

# One method that made for cleaner separation between clusters was filtering out the spam from the dataset:
import re

def is_spam(text):
    patterns = [
        r'"op"\s*:\s*"mint"',
        r'"op"\s*:\s*"link"',
        r'"p"\s*:\s*"mbc-20"',
    ]
    return any(re.search(p, text) for p in patterns)

spam_mask = ~df["text"].apply(is_spam)
clean_indices = np.where(spam_mask.values)[0]

df = df[spam_mask].reset_index(drop=True)
embeddings = embeddings[clean_indices]

print(f"After spam filter...")
print(f"Sanity check - we expect the number of posts in the dataframe and the number of embeddings to match. Number of posts: {len(df):,}, number of embeddings: {embeddings.shape[0]:,}")
#From now, we will proceed with the raw text and see how it performs in the clustering step - FUTURE RESEARCH SHOULD COME BACK.

print('\nFilter has been applied - we now move on to dimensionality reduction to avoid curse of dimensionality...\n')


## Added component to do away w the unclean clustering problems, reflected by a low silhoutte score: apply ONE dimensionality reduction (simplicity), and make sure the embedding quality is retained:
# This might allow us to reduce the curse of dimensionality (NOT to be confused with embedding collapse).

''' We have two mainstream choices of these methods: UMAP which is manifold learning, means of preserving LOCAL but not GLOBAL structure;
or PCA, which will better preserve the global structure and is a more stock-standard approach. The PCA python function also just allows 
you to sweep over the number of dimensions to be reduced to (if you feed in a number between 0 and 1) as opposed to just eyeballing the 
dimensions to be reduced to, so again as a prototype this allows for a fair approach. 
For reference, though, could also extend this approach: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
'''

percentage_variance = 0.90 # Tweaked this parameter manually, balancing embedding quality with silhoutte score.
pca = PCA(n_components=percentage_variance, random_state=67) #Or use cuml.PCA(...)
embeddings = pca.fit_transform(embeddings) # Dim reduction until we dip below 95% of variance being explained by the sum of the PCs. The inclusion of this allows to verify embedding quality. 

print(f"Dimensions retained at {percentage_variance*100:.0f}% variance: {pca.n_components_}, down from 384!")
print("we expect the shape of the reduced embeddingz to be (SAMPLE_SIZE, n_components). Shape is:", embeddings.shape)

print('...\nEmbeddings have now been fully cleaned/reduced. Moving on to clustering\n...')

# KMeansCluster section - take centroids of subset of data with MiniBatch class and minimise distances to these centroids to figure out where these clusters take place.
# We expect from the Personas paper (Amin et al.) for the clusters to be k=5 but don't trust it fully so are replicating via the 'Silhoutte' sweep below.

# Run over the 3-8 kmeans cluster parameter of k and optimise the SILHOUTTE SCORE - search up, but essentially this is the quality of the clustering/separation:
n_init = 10 # Depending on runtime on system-1 or other, this can be increased to intialise more random states for taking minibatches.

# sweep k=3 to k=8
scores = {}
for k in range(3, 9):
    print(f"Fitting k={k}...")
    km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=n_init)
    labels = km.fit_predict(embeddings) # Fit to the REDUCED embeddingz
    score = silhouette_score(embeddings, labels, metric="cosine")
    scores[k] = round(score, 3)
    print(f"  k={k} silhouette={score:.3f}") 

# find the optimal
best_k = max(scores, key=scores.get)
print(f"\nOptimal k={best_k} (score={scores[best_k]})")
print(f"Full scores OF EACH : {scores}")

print('...\nNow optimal k has been found, we can fit the model to the FULL DATASET.\n...')

# IF MORE REPRESENTATIVE SAMPLES ARE NEEDED, FIT K-MEANS TO THE FULL DATASET WITH THE OPTIMAL K:
# embeddings = reducer.transform(embeddings) # Apply the same PCA transformation to the FULL embeddingz

# then fit k-means on the reduced version
kmeans = MiniBatchKMeans(n_clusters=best_k, random_state=42, n_init=n_init) #Fit to the swept best k
labels = kmeans.fit_predict(embeddings) # Fit the MiniBatch to embeddings
centroids = kmeans.cluster_centers_ # For similarity calculations later on ...

print(' ')
print(f"Fitted MiniBatchKMeans with k={best_k} to the FULL dataset, and extracted centroids for similarity calculations. Moving tograbbing representative posts...")
print('')

## FINAL STEP - for input into Moltbook, we want to extract a certain number of samples.
# Take a 'n_posts' number of samples for each cluster. 
#  We could inspect manually, or put them into an LLM as with prior papers, to extract persona summaries.

N_POSTS = 500 # Hyperparameter to change - 500 is representative for prototype.

for k in range(kmeans.n_clusters):
    print(f"\nCluster {k}:") # Each cluster being representative of a persona
    
    # get indices of all posts in this cluster
    cluster_mask = labels == k # Boolean 'masking' method - this allows us to map onto the k clusters
    cluster_indices = np.where(cluster_mask)[0]
    cluster_embeddings = embeddings[cluster_indices] 
    # from numpy docs - https://numpy.org/doc/stable/user/basics.indexing.html#boolean-array-indexing
    
    # compute cosine similarity of each post to its cluster centroid
    centroid = centroids[k].reshape(1, -1)
    similarities = cosine_similarity(cluster_embeddings, centroid).flatten() # simple 'dot-product'-like method to calculate the similarity of each post to the centroid. The higher the similarity, the more representative the post is of the cluster/persona.
    
    # get top N_POSTS most similar to centroid - based on the same cosine similarity
    top_local_idx = np.argsort(similarities)[::-1][:N_POSTS] # Argsort for the most similar posts within each of the clusters.
    top_global_idx = cluster_indices[top_local_idx]
    
    top_posts = df.iloc[top_global_idx][["title", "content"]].copy()
    top_posts["similarity"] = similarities[top_local_idx]
    
    print(f"  Total posts in cluster: {cluster_mask.sum():,}")
    print(f"  MININUM cos-similarity score: {similarities[top_local_idx[-1]]:.3f}")
    for title in top_posts["title"].head(5).tolist():
        print(f"    - {title[:80]}")
    
    # write to txt file - FOR STEPHEN'S MiroFish sim
    out_path = f"test_cluster_{k}_posts.txt"
    target_dir = "/datapool/operational_data/dumps_personas"
    file_path = f"{target_dir}/{out_path}"
    
    print(f"Writing text data to disk: {file_path}...")
    
    with open(file_path, "w") as f:
        for i, row in enumerate(top_posts.itertuples()):
            f.write(f"POST {i+1}:\n")
            f.write(f"Title: {row.title}\n")
            # Note: Double check if your DataFrame column is named 'content' or 'text' 
            # based on your previous spam filtering logic!
            f.write(f"Content: {row.content}\n\n")

    time.sleep(3)
    
    print(f"  Saved successfully to {file_path}")

print('\nAll done!! We have now got the data from Moltbook, embedded, clustered, and extracted the rep posts from the embeddings. ' \
'This data is now ready for use in MiroFish. Note that only 3 personas have been generated, but there are many posts that don\'t have perfect clustering.' \
'When you are prompting MiroFish, therefore, it is very important to think about how to initialise the agents so that they are drawn from a wide range of these posts,' \
'But also not just one of the posts, such that within the OASIS sim, there is both inter- and intra-persona diversity. (ref: https://www.betterhelp.com/advice/psychology/why-diversity-in-politics-plays-a-role-in-society/' \
', noting that this same diveristy should also be reflected between the HUMAN and AI personas if the sim is on a hybrid group.)')
