#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 1. Pengumpulan Data
data = pd.read_csv("https://raw.githubusercontent.com/rebekz/datascience_course/main/data/rollingsales/rollingsales_brooklyn_2016.20160830.csv")

data


# In[16]:


# Langkah 2: Pembersihan dan Eksplorasi Data

# Menampilkan informasi dataset
print("Jumlah baris dan kolom dalam dataset:", data.shape)
print("\nInformasi dataset:")
print(data.info())

# Menampilkan statistik ringkasan dataset
print("\nStatistik Ringkasan:")
print(data.describe())

# Menampilkan beberapa baris pertama dataset
print("\nBeberapa Baris Pertama Dataset:")
print(data.head())


# In[17]:


fitur = ['building.class.category', 'neighborhood', 'tax.class.at.present']
target = 'sale.price'

# 3. Transformasi dan Preprocessing Data
encoder = LabelEncoder()
data_encoded = data.copy()

# Melakukan encoding pada fitur
for feature in fitur:
    data_encoded[feature + '_encoded'] = encoder.fit_transform(data_encoded[feature])

# Menampilkan hasil encoding
print(data_encoded[fitur + [feature + '_encoded' for feature in fitur]].head())


# In[20]:


# 4. Eksplorasi Data

# Statistik deskriptif untuk kolom 'sale_price'
print("Statistik Deskriptif untuk Kolom 'sale price':")
print(data['sale.price'].describe())

# Distribusi frekuensi untuk kolom 'building_class_category'
print("\nDistribusi Frekuensi untuk Kolom 'building class category':")
print(data['building.class.category'].value_counts())

# Histogram untuk kolom 'sale_price'
plt.figure(figsize=(10, 6))
plt.hist(data['sale.price'], bins=20)
plt.title("Histogram of Sale Price")
plt.xlabel("Sale Price")
plt.ylabel("Frequency")
plt.show()


# In[59]:


print(data['sale.price'].dtypes)
print(data['building.class.category'].dtypes)
print(data['neighborhood'].dtypes)


# In[60]:


data['building.class.category'] = data['building.class.category'].astype('category')
data['neighborhood'] = data['neighborhood'].astype('category')


# In[62]:


encoder = LabelEncoder()
data['building.class.category_encoded'] = encoder.fit_transform(data['building.class.category'])
data['neighborhood_encoded'] = encoder.fit_transform(data['neighborhood'])
print(data)


# In[45]:


# 5. Analisis Statistik

# Korelasi antara 'sale_price' dan 'building_class_category'
correlation = data['sale.price'].corr(data['building.class.category_encoded'])
print("Korelasi antara 'sale.price' dan 'building.class.category_encoded':", correlation)

# Rata-rata harga penjualan berdasarkan lingkungan
mean_price_by_neighborhood = data.groupby('neighborhood')['sale.price'].mean()
print("\nRata-rata harga penjualan berdasarkan lingkungan:")
print(mean_price_by_neighborhood)


# In[65]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 6. Pembuatan Model dan Prediksi

# Memisahkan fitur dan target
X = data[['building.class.category_encoded', 'gross.square.feet', 'neighborhood_encoded']]
y = data['sale.price']

# Memisahkan data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat model regresi linear
model = LinearRegression()

# Melatih model menggunakan data latih
model.fit(X_train, y_train)

# Melakukan prediksi pada data uji
y_pred = model.predict(X_test)

# Menghitung mean squared error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Menampilkan hasil prediksi dan MSE
print('Prediksi Harga Penjualan Properti berdasarkan lingkungan:')
print(y_pred)
print('Mean Squared Error (MSE):', mse)


# In[68]:


# 7. Interpretasi dan Kesimpulan
coefficients = pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_})
print(coefficients)


# In[69]:


# Interpretasi
# Contoh: Berdasarkan model regresi linear, koefisien positif pada fitur "neighborhood_encoded" menunjukkan 
# bahwa terdapat hubungan positif antara lingkungan dengan harga properti di Brooklyn. Nilai koefisien yang 
# lebih tinggi (misalnya, 1000) menunjukkan pengaruh yang lebih besar terhadap harga properti dibandingkan 
# dengan fitur lainnya. Artinya, perbedaan 1 unit dalam nilai "neighborhood_encoded" dapat mengakibatkan 
# peningkatan sebesar $1000 dalam harga properti.

# Kesimpulan dan Rekomendasi
# Contoh: Berdasarkan analisis data rollingsales Brooklyn 2016-20160830, ditemukan bahwa rata-rata harga properti 
# di Brooklyn adalah $500,000, dengan median harga properti sebesar $450,000. Distribusi harga properti cenderung 
# normal dengan beberapa outliers yang signifikan di sisi kanan. Tren peningkatan harga properti terlihat dalam 
# rolling average, menunjukkan bahwa harga properti cenderung meningkat seiring waktu.

# Rekomendasi:
# 1. Perhatikan lingkungan (neighborhood) sebagai faktor penting dalam menentukan harga properti di Brooklyn. 
# Lingkungan yang lebih berkualitas atau populer dapat memiliki dampak positif pada harga properti.

# 2. Evaluasi secara cermat fitur-fitur lain yang berpengaruh dalam menentukan harga properti di Brooklyn. 
# Fitur seperti jumlah kamar, luas bangunan, dan fasilitas di sekitar properti dapat menjadi faktor penting 
# dalam menarik minat pembeli.

# 3. Perhatikan perubahan tren pasar properti di Brooklyn. Memantau dan memahami tren harga properti dapat 
# membantu dalam pengambilan keputusan yang lebih baik dalam jual beli properti.


# In[ ]:




