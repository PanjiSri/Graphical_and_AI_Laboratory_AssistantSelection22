# Tugas Seleksi Asisten Lab GaIB ‘22

## Kreator

Panji Sri Kuncara Wisma - 13522028

## Deskripsi Proyek

Proyek ini merupakan bagian dari seleksi Asisten Lab GaIB ‘22, yang mencakup penerapan algoritma machine learning baik secara supervised maupun unsupervised. Setiap bagian telah diimplementasikan dari _scratch_ menggunakan Python, serta dibandingkan dengan implementasi menggunakan pustaka populer seperti `scikit-learn` dan `tensorflow`.

## Struktur Repository
```
.
├── answers/
│   ├── supervised-learning/
│   │   ├── ann-scratch.pdf
│   │   ├── cart-classification.pdf
│   │   ├── gaussian-naive-bayes.pdf
│   │   ├── knn.pdf
│   │   ├── logistic-regression.pdf
│   │   └── svm-scratch.pdf
│   └── unsupervised-learning/
│       ├── dbscan.pdf
│       ├── kmeans_scratch.pdf
│       └── pca.pdf
├── dataset/
│   ├── dataset1.csv
│   └── dataset2.csv
├── src/
│   ├── supervised-learning/
│   │   ├── ann-scratch.py
│   │   ├── cart-classification.py
│   │   ├── gaussian-naive-bayes.py
│   │   ├── knn.py
│   │   ├── logistic-regression.py
│   │   └── svm-scratch.py
│   └── unsupervised-learning/
│       ├── dbscan.py
│       ├── kmeans_scratch.py
│       └── pca.py
├── doe.py
├── unsuperv.py
├── .gitignore
├── README.md
└── requirements.txt

```

## Cara Penggunaan dan Informasi Penting Tambahan

Untuk menjalankan kode dalam repository ini, disarankan menggunakan Windows Subsystem for Linux (WSL) atau lingkungan Linux lainnya untuk memastikan kompatibilitas penuh. Seluruh file .py di dalam direktori src dianggap sebagai library, dan cara terbaik untuk menggunakannya adalah dengan mengimpornya ke dalam skrip atau notebook lain sesuai kebutuhan.

### Contoh Penggunaan

Untuk menggunakan model logistic regression dari scratch sebagai library, Anda bisa mengimpor modulnya seperti berikut:

```
import os
import importlib.util
import sys

module_name = "logistic-regression"
module_path = os.path.join('supervised-learning', module_name + '.py')

spec = importlib.util.spec_from_file_location(module_name, module_path)
logistic_module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = logistic_module
spec.loader.exec_module(logistic_module)

LogisticRegressionScratch = logistic_module.LogisticRegressionScratch

LogisticRegressionScratch
model = LogisticRegressionScratch()
```

### Informasi Khusus untuk File .ipynb dan .py

- **doe.ipynb**

    Notebook ini digunakan untuk seluruh bagian 2 (Supervised Learning). Di dalamnya sudah diurutkan penggunaan setiap algoritma dari scratch sesuai spesifikasi pada dataset1.csv.

- **unsuperv.ipynb**

    Notebook ini digunakan untuk seluruh bagian 3 (Unsupervised Learning). Di dalamnya juga sudah diurutkan penggunaan setiap algoritma dari scratch sesuai spesifikasi pada dataset2.csv.


## Checklist Algoritma yang Diimplementasikan

### Supervised Learning (Bagian 2)
- [v] KNN
- [v] Logistic Regression
- [v] Gaussian Naive Bayes
- [v] CART
- [v] SVM
- [v] ANN

**Bonus yang diimplementasikan:**
- Implementasi Newton’s Method untuk Logistic Regression
- Menerapkan kernel lain untuk SVM yaitu polinomial dan rbf

### Unsupervised Learning (Bagian 3)
- [v] K-MEANS
- [v] DBSCAN
- [v] PCA

**Bonus yang diimplementasikan:**
- Metode inisialisasi k-means++ 

### Reinforcement Learning (Bagian 4)
- [v] Q-LEARNING
- [v] SARSA


