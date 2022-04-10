from .base import ClientSelection
import numpy as np


# Clustered Sampling
class ClusteredSampling1(ClientSelection):
    def __init__(self, n_samples, num_clients):
        super().__init__(n_samples, num_clients)
        '''
        Since clustering is performed according to the clients sample size n_i,
        unless n_i changes during the learning process,
        Algo 1 needs to be run only once at the beginning of the learning process.
        '''
        epsilon = int(10 ** 10)
        client_ids = sorted(n_samples.keys())
        n_samples = np.array([n_samples[i] for i in client_ids])
        weights = n_samples / np.sum(n_samples)
        total = len(client_ids)
        n_cluster = num_clients
        
        # associate each client to a cluster
        augmented_weights = np.array([w * n_cluster * epsilon for w in weights])
        ordered_client_idx = np.flip(np.argsort(augmented_weights))

        print(augmented_weights)

        distri_clusters = np.zeros((n_cluster, total)).astype(int)
        k = 0
        for client_idx in ordered_client_idx:
            while augmented_weights[client_idx] > 0:
                sum_proba_in_k = np.sum(distri_clusters[k])
                u_i = min(epsilon - sum_proba_in_k, augmented_weights[client_idx])
                print(u_i)
                distri_clusters[k, client_idx] = u_i
                augmented_weights[client_idx] += -u_i
                sum_proba_in_k = np.sum(distri_clusters[k])
                if sum_proba_in_k == 1 * epsilon:
                    k += 1

        distri_clusters = distri_clusters.astype(float)
        for l in range(n_cluster):
            distri_clusters[l] /= np.sum(distri_clusters[l])
            print(np.sum(distri_clusters[l]))

        self.distri_clusters = distri_clusters
        

    def select(self, round, possible_clients, num_clients):
        num_clients = min(num_clients, len(possible_clients))
        selected_clients = []
        for k in range(num_clients):
            weight = self.distri_clusters[k]  #all clients are online.  ##np.take(self.distri_clusters[k], possible_clients)
            selected_clients.append(np.random.choice(possible_clients, 1, p=weight/sum(weight)))
        return selected_clients