from .base import ClientSelection
import numpy as np
from copy import deepcopy


# Loss-based Client Selection
class LossSampling(ClientSelection):
    def __init__(self, n_samples, num_clients, client_ids, args) -> None:
        super().__init__(n_samples, num_clients, client_ids, args)
        # alpha value for value function
        # alpha > 0: sampling clients with high loss
        # alpha < 0: sampling clients with low loss
        self.alpha = args.alpha
        self.loss = args.loss
        self.save_probs = True
        if self.save_probs:
            self.result_file = open(f'{args.save_path}/values.txt', 'w')
        
    def select(self, round, possible_clients, num_clients, metric):
        num_clients = min(num_clients, len(possible_clients))
        # create value
        if self.loss == 'total':
            metric *= np.array([c.num_samples for c in possible_clients])
        elif self.loss == 'sqrt':
            metric *= np.array([np.sqrt(c.num_samples) for c in possible_clients])
        values = np.exp(np.array(metric) * self.alpha)
        probs = values / sum(values)
        selected_clients = np.random.choice(possible_clients, num_clients, p=probs, replace=False)
        # save
        if self.save_probs:
            self.save_results(values)

        return selected_clients
        
    def save_results(self, arr):
        np.round(arr, 12).tofile(self.result_file, sep=',')
        self.result_file.write("\n")
    
    def close_file(self):
        if self.save_probs:
            self.result_file.close()


# Loss-based Client Selection
class LossRankSampling(ClientSelection):
    def __init__(self, n_samples, num_clients, client_ids, args) -> None:
        super().__init__(n_samples, num_clients, client_ids, args)
        self.save_probs = True
        if self.save_probs:
            self.result_file = open(f'{args.save_path}/values.txt', 'w')
        
    def select(self, round, possible_clients, num_clients, metric):
        num_clients = min(num_clients, len(possible_clients))
        # create rank value
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
        
    def save_results(self, arr):
        np.round(arr,8).tofile(self.result_file, sep=',')
        self.result_file.write("\n")
    
    def close_file(self):
        if self.save_probs:
            self.result_file.close()



# Loss-based Client Selection
class LossRankSelection(ClientSelection):
    def __init__(self, n_samples, num_clients, client_ids, args) -> None:
        super().__init__(n_samples, num_clients, client_ids, args)
        
    def select(self, round, possible_clients, num_clients, metric):
        num_clients = min(num_clients, len(possible_clients))
        # select high rank value
        selected_client_idxs = np.argsort(metric)[-num_clients:]
        selected_clients = np.take(possible_clients, selected_client_idxs)
        return selected_clients



# Power-of-d-Choice
class PowerOfChoice(ClientSelection):
    def __init__(self, n_samples, num_clients, client_ids, args) -> None:
        super().__init__(n_samples, num_clients, client_ids, args)
    
    def select_candidates(self, possible_clients, d):
        buffer_size = min(d, len(possible_clients))
        n_samples = [c.num_samples for c in possible_clients]
        weights = n_samples / np.sum(n_samples)
        candidate_clients = np.random.choice(possible_clients, buffer_size, p=weights/sum(weights), replace=False)
        return candidate_clients
        
    def select(self, round, possible_clients, num_clients, metric):
        num_clients = min(num_clients, len(possible_clients))
        
        # select high loss value
        selected_client_idxs = np.argsort(metric)[-num_clients:]
        selected_clients = np.take(possible_clients, selected_client_idxs)
        return selected_clients



# Active Federated Learning
class ActiveFederatedLearning(ClientSelection):
    def __init__(self, n_samples, num_clients, client_ids, args) -> None:
        super().__init__(n_samples, num_clients, client_ids, args)
        self.alpha1 = 0.75  # args.alpha1 #0.75
        self.alpha2 = args.alpha  #0.01
        self.alpha3 = 0.1   # args.alpha3 #0.1

        self.save_probs = True
        if self.save_probs:
            self.result_file = open(f'{args.save_path}/values.txt', 'w')

    def select(self, round, possible_clients, num_clients, metric):
        num_clients = min(num_clients, len(possible_clients))
        # set sampling distribution
        values = np.exp(np.array(metric) * self.alpha2)
        # 1) select 75% of K(total) users
        drop_client_idxs = np.argsort(metric)[:int(self.alpha1 * len(possible_clients))]
        probs = deepcopy(values)
        probs[drop_client_idxs] = 0
        probs /= sum(probs)
        #probs = np.nan_to_num(probs, nan=max(probs))
        # 2) select 99% of m users using prob.
        num_select = int((1 - self.alpha3) * num_clients)
        #np.random.seed(round)
        selected = np.random.choice(len(metric), num_select, p=probs, replace=False)
        # 3) select 1% of m users randomly
        not_selected = np.array(list(set(np.arange(len(metric))) - set(selected)))
        selected2 = np.random.choice(not_selected, num_clients - num_select, replace=False)
        selected_client_idxs = np.append(selected, selected2, axis=0)
        print(f'{len(selected_client_idxs)} selected users: {selected_client_idxs}')
        
        selected_clients = np.take(possible_clients, selected_client_idxs)
        
        # save result file
        if self.save_probs:
            self.save_results(metric)
            self.save_results(values)
            self.save_results(probs)
        
        return selected_clients
    
    def save_results(self, arr):
        np.round(arr,8).tofile(self.result_file, sep=',')
        self.result_file.write("\n")
    
    def close_file(self):
        if self.save_probs:
            self.result_file.close()