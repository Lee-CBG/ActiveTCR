from math import log
import numpy as np

def entropy_per_sample(x):
    if x == 0 or x == 1: # to prevent math domain error
        x = 0.0000001
    return -(x*log(x) + (1-x)*log(1-x))



class query_strategies:
    '''
    Query name mappings. function names (left) : query strategies in paper (right)
    
    random_sampling                                    : random sampling
    entropy_sampling                                   : global entropy sampling
    entropy_and_random_pooling_sampling                : local entropy
    entropy_and_random_pooling_and_dropout_sampling    : local entropy + dropout committee sampling
    leastY_sampling                                    : misclassification sampling
    mostY_sampling                                     : misclassification sampling
    '''
    def __init__(self, model, n_samples=12003):
        self.model = model
        self.n_samples = n_samples
        self.pooling_size = n_samples * 2
        
    def random_sampling(self, X1, X2):
        return np.random.choice(len(X1), self.n_samples, replace=False)
    
    def entropy_sampling(self, X1, X2):
        yhat = self.model.predict([X1, X2])
        entropy_res = [0 for _ in range(len(yhat))]
        for i, n in enumerate(yhat.flatten()):
            entropy_res[i] = entropy_per_sample(n)
        return np.argpartition(entropy_res, -self.n_samples)[-self.n_samples:] # biggest indexes
    
    def entropy_and_random_pooling_sampling(self, X1, X2):
        random_pooling_idx = np.random.choice(len(X1), self.pooling_size, replace=False)
        
        # use a hashmap to keep track of the index {0, 1, 2...:random_sampling_ind}
        hashmap = defaultdict(int)
        for i, n in enumerate(random_pooling_idx):
            hashmap[i] = n

        query_idx_within_pool = self.entropy_sampling(X1[random_pooling_idx], X2[random_pooling_idx])
        query_idx = [hashmap.get(key) for key in query_idx_within_pool]
        return query_idx
    
    def entropy_and_dropout_sampling(self, X1, X2):
        # Entropy
        yhat = self.model.predict([X1, X2])
        entropy_res = [0 for _ in range(len(yhat))]
        for i, n in enumerate(yhat.flatten()):
            entropy_res[i] = entropy_per_sample(n)
            
        # KL divergence
        vote_proba = np.stack([self.model([X1, X2], training=True) for _ in range(10)])
        vote_proba = np.concatenate((vote_proba, 1 - vote_proba), axis = 2)

        # first compute the average of the class probabilities of each time dropout prediction
        # this is called the consensus probablity
        consensus_proba = vote_proba.mean(axis=0)

        # KL_div
        from scipy.stats import entropy
        KL_div = [[0 for _ in range(2)] for _ in range(len(X1))]
        for i in range(len(X1)):
            for j in range(2):
                KL_div[i][j] = entropy(vote_proba[j, i], qk=consensus_proba[i])

        # max_disagreement
        max_disagreement = np.array(KL_div).max(axis=1)
        
        entropy_plus_KL = entropy_res + max_disagreement
        query_idx = np.argpartition(entropy_plus_KL, -self.n_samples)[-self.n_samples:]
        return query_idx 

        
    def dropout_sampling_and_random_pooling_sampling(self, X1, X2):
        
        random_pooling_idx = np.random.choice(len(X1), self.pooling_size, replace=False)
        
        # use a hashmap to keep track of the index {0, 1, 2...:random_sampling_ind}
        hashmap = defaultdict(int)
        for i, n in enumerate(random_pooling_idx):
            hashmap[i] = n
        
        # KL divergence
        vote_proba = np.stack([self.model([X1[random_pooling_idx], X2[random_pooling_idx]], training=True) for _ in range(10)])
        vote_proba = np.concatenate((vote_proba, 1 - vote_proba), axis = 2)

        # first compute the average of the class probabilities of each time dropout prediction
        # this is called the consensus probablity
        consensus_proba = vote_proba.mean(axis=0)

        # KL_div
        from scipy.stats import entropy
        KL_div = [[0 for _ in range(2)] for _ in range(len(X1[random_pooling_idx]))]
        for i in range(len(X1[random_pooling_idx])):
            for j in range(2):
                KL_div[i][j] = entropy(vote_proba[j, i], qk=consensus_proba[i])

        # max_disagreement
        max_disagreement = np.array(KL_div).max(axis=1)
        
        query_idx_within_pool = np.argpartition(max_disagreement, -self.n_samples)[-self.n_samples:]
        query_idx = [hashmap.get(key) for key in query_idx_within_pool]
        return query_idx
        
    
    def entropy_and_random_pooling_and_dropout_sampling(self, X1, X2):
        random_pooling_idx = np.random.choice(len(X1), self.pooling_size, replace=False)
        hashmap = defaultdict(int)
        for i, n in enumerate(random_pooling_idx):
            hashmap[i] = n
            
        query_idx_within_pool = self.entropy_and_dropout_sampling(X1[random_pooling_idx], X2[random_pooling_idx])
        query_idx = [hashmap.get(key) for key in query_idx_within_pool]
        return query_idx
    
    def leastY_sampling(self, X1, X2):
        """
        Rationale example:
        
        Recall that the goal of this project is to reduce annotation cost for positively bind TCR-epitope pairs. 
        Below is a yhat for from 10 pool_pos. If we use entropy_sampling, we will spot samples that has a prediction
        value close to 0.5, representing the informatives. 
        
        A problem is entropy_sampling will treat the prediction value 0.99 and 0.01 same, because they share same
        entropy score. This may not be what we want as 0.99 means the classifier confidently and correctly judged
        that TCR-epitope pair is binding, and 0.01 means the classifier confidently and WRONGLY judged that 
        TCR-epitope is NOT binding. 
        
        array([[0.9997564 ],
               [0.97860193],
               [0.9998944 ],
               [0.05928884],
               [0.03913847],
               [0.42862335],
               [0.9795568 ],
               [0.9995041 ],
               [0.9999989 ],
               [0.8535571 ]], dtype=float32)
        """
        yhat = self.model.predict([X1, X2])
        return np.argpartition(yhat.flatten(), self.n_samples)[:self.n_samples] # smallest indexes
    
    def mostY_sampling(self, X1, X2):
        yhat = self.model.predict([X1, X2])
        return np.argpartition(yhat.flatten(), -self.n_samples)[-self.n_samples:] # smallest indexes