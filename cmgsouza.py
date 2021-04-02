import random
import numpy as np
from scipy.spatial import distance

class KMeansMahalanobisClassifier:
    def __init__(self, k, initial_iterations=5, iterations=100):
        self.k = k
        self.initial_iterations = initial_iterations
        self.iterations = iterations
        self.seed = 100
    
    def mahalanobis_distance(self, u, v, cov_inv):
        return np.sqrt(np.dot(np.dot(u - v, cov_inv), (u - v)))
    
    def fit(self, dataset):
        #initialization step
        random.seed(self.seed)
        
        #assign initial centroids
        self.centroids = list(random.sample(list(dataset), 1))
        
        for i in range(self.k - 1):
            dist = np.array([min([np.linalg.norm(sample - centroid)**2 for centroid in self.centroids]) for sample in                                                                                                                   dataset])
            index = np.random.choice(range(len(dataset)), p=(dist/dist.sum()))
            self.centroids.append(dataset[index])
        
        #assign initial clusters 
        for i in range(self.initial_iterations):
            
            self.partitions = {i:[] for i in range(self.k)}
        
            for sample in dataset:
                distances = [np.sqrt(distance.euclidean(sample, centroid)) for centroid in self.centroids]
                self.partitions[distances.index(min(distances))].append(sample)
            
            #update the centroids
            for cluster in self.partitions:
                self.centroids[cluster] = np.mean(self.partitions[cluster], axis=0)
        
        #iteration step
        for i in range(self.iterations):            
            
            old_partitions = dict(self.partitions)
            
            self.partitions = {i:[] for i in range(self.k)}
            
            #assign points to a cluster given the smallest mahalanobis distance            
            for sample in dataset:
                distances = [self.mahalanobis_distance(np.mean(cluster, axis=0), sample, np.linalg.inv(np.cov(cluster,                                                                  rowvar=False))) for cluster in old_partitions.values()]
                self.partitions[distances.index(min(distances))].append(sample)
            
            
            #compare current cluster with the previous one  
            clusters_comparison = [np.allclose(np.mean(self.partitions[cluster]), np.mean(old_partitions[cluster])) for                                                                                             cluster in self.partitions]
            
            if False not in clusters_comparison:
                return self
                break
           
        return self
    
    def predict(self, dataset):
 
        self.labels = []
        for sample in dataset:
            distances = [self.mahalanobis_distance(np.mean(cluster, axis=0), sample, np.linalg.inv(np.cov(cluster, rowvar=False))) for cluster in self.partitions.values()]
            self.labels.append(distances.index(min(distances)))
        return np.array(self.labels)


