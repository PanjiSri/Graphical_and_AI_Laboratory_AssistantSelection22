import numpy as np

class CARTScratch:
    def __init__(self, min_samples=2, max_height=5):
        self.min_samples = min_samples
        self.max_height = max_height
        self.tree = None
    
    def fit(self, X, y):
        print("Memulai proses fitting data...")
        self.tree = self.grow_tree(X, y)
        print("Proses fitting selesai.")
    
    def grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        print(f"Menumbuhkan pohon pada depth {depth} dengan {n_samples} sampel...")
        
        if n_samples >= self.min_samples and depth < self.max_height:
            best_split = self.find_split(X, y, n_features)
            
            # Jika split ditemukan dengan gain positif
            if best_split and best_split['gain'] > 0:
                print(f"Pembagian terbaik ditemukan di fitur {best_split['feature']} dengan threshold {best_split['threshold']}.")
                left_subtree = self.grow_tree(best_split['X_left'], best_split['y_left'], depth + 1)
                right_subtree = self.grow_tree(best_split['X_right'], best_split['y_right'], depth + 1)
                return TreeNode(best_split['feature'], best_split['threshold'], 
                                left_subtree, right_subtree, gain=best_split['gain'])
        
        # Node daun jika kondisi tidak terpenuhi
        leaf_value =self.determine_leaf_value(y)
        print(f"Leaf node terbentuk dengan nilai {leaf_value}.")
        return TreeNode(value=leaf_value)
    
    def find_split(self, X, y, n_features):
        best_split = None
        max_gain = -float("inf")
        
        for feature_index in range(n_features):
            feature_values = X[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            for threshold in possible_thresholds:
                X_left, y_left, X_right, y_right = self.partition(X, y, feature_index, threshold)
                
                # Jika ada sampel di setiap sisi split
                if len(y_left) > 0 and len(y_right) > 0:
                    gain = self.calculate_gain(y, y_left, y_right)
                    if gain > max_gain:
                        best_split = {
                            'feature': feature_index,
                            'threshold': threshold,
                            'X_left': X_left, 'y_left': y_left,
                            'X_right': X_right, 'y_right': y_right,
                            'gain': gain
                        }
                        max_gain = gain
        
        return best_split
    
    def partition(self, X, y, feature_index, threshold):
        left_mask = X[:, feature_index] <= threshold
        right_mask = X[:, feature_index] > threshold
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]
        return X_left, y_left, X_right, y_right
    
    def calculate_gain(self, y, y_left, y_right):
        prop_left = len(y_left) / len(y)
        prop_right = len(y_right) / len(y)
        gain = self.calculate_gini(y) - (prop_left * self.calculate_gini(y_left) + prop_right * self.calculate_gini(y_right))
        return gain
    
    def calculate_gini(self, y):
        unique_classes = np.unique(y)
        gini_index = 1.0
        for cls in unique_classes:
            p_cls = len(y[y == cls]) / len(y)
            gini_index -= p_cls ** 2
        return gini_index
    
    def determine_leaf_value(self, y):
        leaf_value = np.bincount(y).argmax()
        return leaf_value
    
    def predict(self, X):
        if self.tree is None:
            raise ValueError("Model belum fit dengan data! Silakan fit model terlebih dahulu.")
        print("Mulai prediksi...")
        predictions = [self.make_prediction(inputs) for inputs in X]
        print("Prediksi selesai.")
        return np.array(predictions)
    
    def make_prediction(self, inputs):
        node = self.tree
        while node.left:
            if inputs[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None, gain=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.gain = gain
