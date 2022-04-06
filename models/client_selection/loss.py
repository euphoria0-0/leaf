from .base import ClientSelection
import numpy as np
import os

# Random Client Selection
class LossSampling(ClientSelection):
    def __init__(self, n_samples, num_clients) -> None:
        super().__init__(n_samples, num_clients)

    def set_hyperparams(self, args):
        self.alpha = args.alpha
        self.save_probs = args.save_probs
        save_path = f'{args.save_path}/{args.dataset}'
        os.makedirs(save_path, exist_ok=True)
        self.result_file = open(f'{save_path}/{args.metrics_name}_values.txt', 'w')
        
    def select(self, round, possible_clients, num_clients, metric):
        num_clients = min(num_clients, len(possible_clients))
        np.random.seed(round)
        # value
        values = np.exp(np.array(metric) * self.alpha)
        probs = values / sum(values)
        selected_clients = np.random.choice(possible_clients, num_clients, p=probs, replace=False)
        # save
        if self.save_probs:
            self.save_results(values)

        return selected_clients
    
    def preselect(self, round, possible_clients, num_clients):
        num_clients = min(num_clients, len(possible_clients))
        np.random.seed(round)
        selected_clients = np.random.choice(possible_clients, num_clients, replace=False)
        return selected_clients
    
    def save_results(self, arr):
        arr.astype(np.float32).tofile(self.result_file, sep=',')
        self.result_file.write("\n")


