import numpy as np

class GaussianNaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None

    def fit(self, X, y):
        print("Memulai proses fit...")
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self.mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self.var = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0)
            # menghindari varian nol
            self.var[idx, :] = np.where(self.var[idx, :] == 0, 1e-6, self.var[idx, :])  
            self.priors[idx] = X_c.shape[0] / float(n_samples)
            print(f"Memproses kelas {c}: mean={self.mean[idx, :]}, var={self.var[idx, :]}")

        print("Proses fit selesai.")

    def _calculate_likelihood(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def _calculate_posterior(self, x):
        posteriors = []
        for idx, _ in enumerate(self.classes):
            # menghindari log(0)
            prior = np.log(self.priors[idx] + 1e-9)  
            class_likelihood = np.sum(np.log(self._calculate_likelihood(idx, x) + 1e-9))
            posterior = prior + class_likelihood
            posteriors.append(posterior)
        print(f"Menghitung posterior: {posteriors}")
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        print("Memulai proses prediksi...")
        predictions = np.array([self._calculate_posterior(x) for x in X])
        print("Proses prediksi selesai.")
        return predictions
