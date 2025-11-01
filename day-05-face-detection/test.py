import pickle
data = pickle.load(open("/Users/ariopurba37/Documents/dream/coding/50-days-of-code/day-05/embeddings/face_embeddings.pkl", "rb"))
print(len(data["embeddings"]), len(data["labels"]))
print(data["labels"][:5])
