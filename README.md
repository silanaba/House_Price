# 🏠 Ev Fiyat Tahmin Modeli
Bu projede makine öğrenimi tekniklerini kullanarak ev fiyatları tahmin edilmeye çalışılmıştır. Projenin amacı, belirli ev özelliklerini (konum, metrekare, oda sayısı vb.) girdi olarak alarak evin potansiyel satış fiyatını yüksek doğrulukla öngörmektir.
Model, kullanıcıların kolayca tahmin yapabilmesi için Streamlit kullanılarak basit bir web uygulamasına dönüştürülmüştür.


# 📊 Veri Seti
Kaynak:https://www.kaggle.com/datasets/harlfoxem/housesalesprediction
Özellikler (Features):
* price: Hedef Değişken. Evin satış fiyatı (USD)
* bedrooms: Evdeki toplam yatak odası sayısı
* bathrooms: Evdeki toplam banyo sayısı
* sqft_living: Yaşam alanı metrekare cinsinden (iç mekan).
* sqft_lot: Arsa alanı metrekare cinsinden (dış mekan)
* floors: Evdeki toplam kat sayısı
* waterfront: Evin deniz/göl kenarında olup olmadığı. (0 = Hayır, 1 = Evet)
* view: Mülkün iyi bir manzaraya sahip olup olmadığına dair indeks (0'dan 4'e kadar)
* condition: Evin genel durumu/kondisyonu (1'den 5'e kadar, 5 en iyi)
* grade: Ev inşasının kalitesini yansıtan bir derecelendirme (1'den 13'e kadar)
* sqft_above: Zemin seviyesinin üzerindeki metrekare (üst katlar)
* sqft_basement: Bodrum katının metrekare cinsinden büyüklüğü
* yr_built: Evin inşa edildiği yıl
* yr_renovated: Evin en son yenilendiği yıl. (Yenilenmemişse 0)
* zipcode: Evin bulunduğu posta kodu.
* lat: Evin coğrafi enlemi
* long: Evin coğrafi boylamı
* sqft_living15: En yakın 15 komşunun ortalama yaşam alanı
* sqft_lot15: En yakın 15 komşunun ortalama arsa alanı

# 🛠️ Kullanılan Teknolojiler
Modeli geliştirmek ve çalıştırmak için aşağıdaki araçlar ve kütüphaneler kullanılmıştır:
Dil: Python
Temel Kütüphaneler: Pandas, NumPy (Veri işleme ve analizi)
Makine Öğrenimi: Scikit-learn (Model eğitimi, doğrulama ve değerlendirme)
Kullanılan Algoritmalar: LinearRegression, Lasso, Ridge, KNeighborsRegressor, DecisionTreeRegressor, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor ve XGBRegressor
Model Performansı : R2 Skoru, MAE, RMSE, MSE
Veri Görselleştirme: Matplotlib, Seaborn
Canlı Uygulama : Streamlit 

# ⚙️Proje Adımları 

Kütüphane ve Veri Seti Yükleme
Keşifçi Veri Analizi ( Exploratory Data Analysis EDA) 
Veri Ön İşleme (Data Preprocessing)
Model Geliştirme, Değerlendirme ve Optimizasyon 
Modelin Kaydedilmesi
Canlı Uygulama Geliştirme


# 🧠 Makine Öğrenimi Teorisi ve Uygulanan Metotlar

1. Linear Regression (Doğrusal Regresyon)
Doğrusal regresyon, bağımlı değişken (hedef değişken) ile bağımsız değişkenler (özellikler) arasındaki doğrusal ilişkiyi modellemek için kullanılır. 
2. Lasso (Least Absolute Shrinkage and Selection Operator)
Lasso, doğrusal regresyonun bir varyasyonudur ve modele ceza (penaltı) ekler. Bu ceza, regresyon katsayılarının mutlak değerlerinin toplamını minimize etmeye çalışır. Bu şekilde bazı katsayıları sıfıra indirir, böylece bazı özelliklerin seçilmesini sağlar (özellik seçimi). 
3. Ridge Regression (Ridge Regresyonu)
Ridge regresyonu da doğrusal regresyonun bir varyasyonudur ve Lasso gibi bir ceza terimi ekler, fakat Lasso'dan farkı, bu cezanın katsayıların karelerinin toplamı üzerine uygulandığıdır. Bu, özelliklerin değerlerini küçültmeye çalışır ancak sıfıra indirgenmezler.
4. K Neighbors Regressor (K En Yakın Komşu Regresyonu)
KNN regresyonu, tahmin için en yakın komşuların ortalamasını alır. Herhangi bir noktadaki tahmin, yakınındaki KKK komşunun hedef değişkenlerinin ortalamasına dayanır. Bu, non-parametrik bir modeldir, yani veri hakkında bir varsayım yapmaz.
5. Decision Tree (Karar Ağaçları)
Karar ağacı, veri kümesini sürekli olarak bölerek her bir bölümdeki verinin en iyi şekilde sınıflandırılmasını sağlar. Regresyon problemlerinde, her yaprak düğümü bir hedef değerinin ortalaması ile ilişkilidir. Model, her düğümde veriyi en iyi bölen özelliği seçer.
6. Random Forest Regressor (Rastgele Orman Regresyonu)
Random Forest, çok sayıda karar ağacının birleşimidir. Her bir ağaç, rastgele seçilen bir özellik alt kümesi üzerinde eğitilir ve tahminler, ağaçların ortalamasına dayanır. Bu, modelin overfitting 
7. AdaBoost Regressor
AdaBoost, zayıf öğrenicilerin birleşiminden güçlü bir model oluşturur. Her iterasyonda, bir önceki modelin hatalarını daha fazla vurgular ve bu hataların daha doğru tahmin edilmesini sağlar. Her yeni model, önceki modelin hatalarını düzelterek eğitilir.
8. Gradient Boosting Regressor (Gradyan Artışı Regresyonu)
Gradient Boosting, her bir yeni modelin, mevcut modelin hatalarını düzeltmeye odaklandığı bir tekniktir. Bu algoritma, önceki modelin hataları üzerine gradyan inişi yaparak öğrenir.
9. XGBoost Regressor (XGBoost Regresyonu)
XGBoost, Gradient Boosting'in optimize edilmiş ve düzenlenmiş bir versiyonudur. Ağaçları oluştururken, her bir iterasyonda, gradyan inişi kullanılarak modelin hataları düzeltilir. XGBoost, ağaçlar arasında daha iyi genelleme ve overfitting engelleme sağlar.

# 🎯 Model Değerlendirme Metrikleri
 Ortalama Mutlak Hata (Mean Absolute Error - MAE)
Tüm tahmin hatalarının mutlak değerlerinin ortalamasını verir. Tahminleriniz gerçek değerden ortalama olarak ne kadar sapıyor, onu gösterir. Birimi hedef değişkenle aynıdır.
Kök Ortalama Kare Hata (Root Mean Squared Error - RMSE)
Hataların karelerinin ortalamasının kareköküdür. Büyük hataları daha fazla cezalandırır. Birimi hedef değişkenle aynıdır.
R2 Skoru
  	Modelin bağımlı değişkenin varyansını ne kadar iyi açıkladığını gösterir. 0 ile 1 arasında bir değer alır. 1'e yakın değerler daha iyi uyumu temsil eder.


# ⚙️ Hiperparametre Optimizasyonu (Grid Search)
En iyi model olarak seçilen XGBoost Regressor'ın performansını maksimize etmek için GridSearchCV metodu kullanılmıştır.
Teorik Açıklama
Hiperparametreler, modelin öğrenme sürecinde veriden öğrenmediği, dışarıdan (yani veri bilimcisi tarafından) ayarlanan parametrelerdir (Örn: max_depth, learning_rate). Grid Search, tanımlanan hiperparametre değerlerinin tüm olası kombinasyonlarını sistematik olarak dener ve her kombinasyonu çapraz doğrulama (cross-validation) ile test ederek en iyi performansı veren kombinasyonu bulur.
