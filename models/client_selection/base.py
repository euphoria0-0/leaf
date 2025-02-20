import numpy as np


# base
class ClientSelection(object):
    def __init__(self, n_samples, num_clients, client_ids, args) -> None:
        pass

    def select(self, round, possible_clients, num_clients):
        pass
    
    def close_file(self):
        pass


# Random Client Selection
class RandomSelection(ClientSelection):
    def __init__(self, n_samples, num_clients, client_ids, args) -> None:
        super().__init__(n_samples, num_clients, client_ids, args)
        
    def select(self, round, possible_clients, num_clients):
        num_clients = min(num_clients, len(possible_clients))
        selected_clients = np.random.choice(possible_clients, num_clients, replace=False)
        return selected_clients


# Number of Local Data based Client Selection
class NumDataSampling(ClientSelection):
    def __init__(self, n_samples, num_clients, client_ids, args) -> None:
        super().__init__(n_samples, num_clients, client_ids, args)
        
    def select(self, round, possible_clients, num_clients):
        num_clients = min(num_clients, len(possible_clients))
        
        n_samples = [c.num_samples for c in possible_clients]
        weights = n_samples / np.sum(n_samples)
        
        selected_clients = np.random.choice(possible_clients, num_clients, p=weights/sum(weights), replace=False)
        return selected_clients
