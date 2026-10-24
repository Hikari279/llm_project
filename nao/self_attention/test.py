import numpy as np

# 1. Inisialisasi data contoh scaled_scores dengan shape (T, T)
T = 4
scaled_scores = np.array([
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0],
    [9.0, 10.0, 11.0, 12.0],
    [13.0, 14.0, 15.0, 16.0]
])

# 2. Buat causal mask berbentuk (T, T)
# np.triu(..., k=1) membuat matriks atas (upper triangle) di atas diagonal utama bernilai True
causal_mask = np.triu(np.ones((T, T), dtype=bool), k=1)

# 3. Ganti scores pada posisi yang dilarang (True pada mask) menjadi -np.inf
# Gunakan np.where: jika mask True isi -np.inf, jika False pertahankan nilai asli
masked_scores = np.where(causal_mask, -np.inf, scaled_scores)

# --- Tampilkan Hasil ---
print("Scaled Scores (Original):")
print(scaled_scores)
print("\nCausal Mask (Posisi True akan dilarang):")
print(causal_mask)
print("\nMasked Scores (Output):")
print(masked_scores)
