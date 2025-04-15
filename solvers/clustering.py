
# Calcola la matrice delle distanze euclidee
distanze = cdist(scenari, scenari, metric='euclidean')

# Esegui il clustering gerarchico
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
etichette = cluster.fit_predict(scenari)

print("Etichette dei cluster:", etichette)
