import numpy as np
import math
import random

class Feature:
    def __init__(self,id,data):
        self.id = id
        self.data = np.array(data, dtype=np.float32)
    def to_numpy(self):
        return self.data
    
class Point4D:
    def __init__(self, id, x, y, z, w):
        self.data = np.array([x, y, z, w], dtype=np.float32)
        self.id = id
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.w = float(w)
         
    def to_tuple(self):
        return (self.x, self.y, self.z, self.w)
    
    def to_numpy(self):
        return self.data
    
    def distance(self, other: "Point4D") -> float:
        return math.sqrt(
            (self.x - other.x)**2 +
            (self.y - other.y)**2 +
            (self.z - other.z)**2 +
            (self.w - other.w)**2
        )
    def __repr__(self):
        return f"Point4D({self.x}, {self.y}, {self.z}, {self.w})"
    
class IsolationTree:
    def __init__(self, height_limit, current_height=0):
        self.left = None
        self.right = None
        self.split_feature = None
        self.split_value = None
        self.height_limit = height_limit
        self.current_height = current_height
        self.size = None  # number of points in this node

    def fit(self, X):
        self.size = len(X)

        # Stop if reached height limit or node small
        if self.current_height >= self.height_limit or len(X) <= 1:
            return self

        # Pick random feature
        self.split_feature = random.randint(0, X.shape[1] - 1)

        # Pick random split within feature range
        col = X[:, self.split_feature]
        min_val, max_val = np.min(col), np.max(col)
        if min_val == max_val:
            return self   # no further split possible

        self.split_value = random.uniform(min_val, max_val)

        left_mask = col < self.split_value
        right_mask = ~left_mask

        self.left = IsolationTree(self.height_limit, self.current_height + 1)
        self.right = IsolationTree(self.height_limit, self.current_height + 1)

        self.left.fit(X[left_mask])
        self.right.fit(X[right_mask])

        return self

    def path_length(self, x):
        # If leaf
        if self.left is None or self.right is None:
            # average path length for n points (c(n))
            if self.size <= 1:
                return self.current_height
            return self.current_height + c(self.size)

        # Go left or right
        if x[self.split_feature] < self.split_value:
            return self.left.path_length(x)
        else:
            return self.right.path_length(x)


# c(n) = average path length for an unsuccessful search in a binary tree
def c(n):
    if n <= 1:
        return 0
    return 2 * (math.log(n - 1) + 0.5772156649) - (2 * (n - 1) / n)

class IsolationForest:
    def __init__(self, n_trees=100, sample_size=256, contamination=0.05):
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.contamination = contamination
        self.trees = []
        self.c_avg = c(sample_size)

    def fit(self, X):
        self.trees = []
        X = np.array(X, dtype=np.float32)

        for _ in range(self.n_trees):
            # Random subsample
            if len(X) > self.sample_size:
                ix = np.random.choice(len(X), self.sample_size, replace=False)
                sample = X[ix]
            else:
                sample = X

            height_limit = int(math.ceil(math.log2(self.sample_size)))
            tree = IsolationTree(height_limit)
            tree.fit(sample)
            self.trees.append(tree)

    def anomaly_score(self, x):
        # Average path length of x across all trees
        paths = [tree.path_length(x) for tree in self.trees]
        avg_path = np.mean(paths)
        # s(x) = 2^{-E(h(x))/c(n)}
        return 2 ** (-avg_path / self.c_avg)

    def predict(self, X):
        preds = []
        scores = []
        X = np.array(X, dtype=np.float32)

        # Compute scores
        for x in X:
            s = self.anomaly_score(x)
            scores.append(s)

        # Threshold from contamination
        threshold = np.percentile(scores, 100 * (1 - self.contamination))

        for s in scores:
            preds.append(-1 if s > threshold else 1)

        return np.array(preds), np.array(scores)
    
def remove_junk_isoforest(points,
                          n_trees=100,
                          sample_size=256,
                          contamination=0.05):    

    X = np.array([p.to_numpy() for p in points], dtype=np.float32)

    model = IsolationForest(
        n_trees=n_trees,
        sample_size=sample_size,
        contamination=contamination,
    )

    model.fit(X)
    preds, scores = model.predict(X)

    good = []
    junk = []

    for p, label in zip(points, preds):
        if label == -1:
            junk.append(p)
        else:
            good.append(p)

    return good, junk, preds, scores

 
