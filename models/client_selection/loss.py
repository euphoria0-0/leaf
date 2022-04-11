from .base import ClientSelection
import numpy as np

# Loss-based Client Selection
class LossSampling(ClientSelection):
    def __init__(self, n_samples, num_clients) -> None:
        super().__init__(n_samples, num_clients)

    def set_hyperparams(self, args):
        # alpha value for value function
        # alpha > 0: sampling clients with high loss
        # alpha < 0: sampling clients with low loss
        self.alpha = args.alpha
        self.save_probs = args.save_probs
        if self.save_probs:
            self.result_file = open(f'{args.save_path}/values.txt', 'w')
        
    def select(self, round, possible_clients, num_clients, metric):
        num_clients = min(num_clients, len(possible_clients))
        # value
        values = np.exp(np.array(metric) * self.alpha)
        probs = values / sum(values)
        selected_clients = np.random.choice(possible_clients, num_clients, p=probs, replace=False)
        # save
        if self.save_probs:
            self.save_results(values)

        return selected_clients
    
    def save_results(self, arr):
        arr.astype(np.float32).tofile(self.result_file, sep=',')
        self.result_file.write("\n")
    
    def close_file(self):
        self.result_file.close()



# Loss-based Client Selection
class LossRank(ClientSelection):
    def __init__(self, n_samples, num_clients) -> None:
        super().__init__(n_samples, num_clients)

    def set_hyperparams(self, args):
        # alpha value for value function
        # alpha > 0: sampling clients with high loss
        # alpha < 0: sampling clients with low loss
        self.alpha = args.alpha
        self.save_probs = args.save_probs
        if self.save_probs:
            self.result_file = open(f'{args.save_path}/values.txt', 'w')
        
    def select(self, round, possible_clients, num_clients, metric):
        num_clients = min(num_clients, len(possible_clients))
        # rank-value
        arg = np.argsort(metric)
        rank = np.empty(len(arg), dtype=np.int)
        for i in range(len(arg)):
            rank[arg[i]] = i+1
        probs = rank / sum(rank)
        selected_clients = np.random.choice(possible_clients, num_clients, p=probs, replace=False)

        # save
        if self.save_probs:
            self.save_results(probs)

        return selected_clients
    
    def save_results(self, arr, round):
        arr.astype(np.float32).tofile(self.result_file, sep=',')
        self.result_file.write("\n")
