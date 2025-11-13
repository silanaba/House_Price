import streamlit as st
import pickle
import pandas as pd
from pandas.io.sas.sas_constants import col_count_p1_multiplier



st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"  # veya "centered"
)


st.markdown("""
    <style>
        .title {
            font-size:40px;
            font-weight:bold;
            color:#4CAF50;
            text-align:center;
            margin-bottom:10px;
        }
        .subtitle {
            text-align:center;
            color:gray;
            font-size:18px;
        }
    </style>
    <div class="title">🏡 House Price Prediction App</div>
    <div class="subtitle">Ev özelliklerine göre tahmini satış fiyatını hesaplayın 💰</div>
""", unsafe_allow_html=True)




st.markdown("""
    <style>
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(120deg, blue, red);
        }
        [data-testid="stSidebar"] {
            background-color: black;
        }
        .stButton>button {
            background-color:pink;
            color:white;
            border:none;
            padding:10px 24px;
            border-radius:10px;
            font-size:16px;
        }
        .stButton>button:hover {
            background-color:purple;
        }
    </style>
""", unsafe_allow_html=True)







model = pickle.load(open('model.pkl', 'rb'))

st.title("House Price Prediction App")

st.write("Bu uygulama evin özelliklerine göre tahmini satış fiyatını hesaplar.")

col1, col2 = st.columns(2)


with col1:
 bedrooms = st.number_input("Yatak Odası Sayısı", 0, 100)
 bathrooms = st.number_input("Banyo Sayısı", 1.0, 8.0, 2.0, step = 0.5)
 sqft_living = st.number_input("Yaşam Alanı (sqft)",0, 10000000)
 floors = st.number_input("Kat Sayısı", 1.0,4.0,1.0, step = 0.5)
 grade = st.number_input("Konut Kalite Puanı (Grade)", 1, 13)
 sqft_lot = st.number_input("Toplam Arazi (sqft)",1,1000)
 waterfront = st.selectbox("Etrafta göl, deniz ya da nehir var mı?", [1,2])
 view =  st.selectbox("Evin manzarasının kalitesi hakkında bir puanlama",[1,2,3,4,5])
 condition = st.selectbox("Evin genel kondisyonu ya da durumu – yapısal ve bakım açısından durumu değerlendiren bir puanlama",[1,2,3,4,5] )

with col2:
 sqft_above= st.number_input("Bodrum katı olmayan katlardaki yaşam alanının kare feet olarak ölçüsü", 0, 1000000)
 sqft_basement = st.number_input(" Bodrum katındaki yaşam alanının kare feet olarak ölçüsü", 0, 1000000)
 yr_build = st.number_input("İnşa Edilen Yıl", min_value=1900, step=1)
 yr_renovated = st.number_input("Evin son olarak yenilendiği yıl", min_value=1900, step =1)
 zipcode = st.number_input("Zip code", min_value=98001, max_value=98199,  step=1)
 lat = st.number_input("Evin bulunduğu konumun enlem koordinatı", 0, 100000000)
 lot = st.number_input("Evin bulunduğu konumun boylam koordinatı",1,10000000)
 sqft_living15 = st.number_input("Evin bulunduğu bölgedeki (yaklaşık 15 komşu konut) için yaşam alanı ortalaması",0,10000000)
 sqft_lot15 = st.number_input("Evin bulunduğu bölgedeki 15 komşu evin arsalarının kare feet olarak toplam ya da ortalama ölçüsü",0,10000000)

input_data = pd.DataFrame({
    'bedrooms': [bedrooms],
    'bathrooms': [bathrooms],
    'sqft_living': [sqft_living],
    'sqft_lot': [sqft_lot],
    'waterfront': [waterfront],
    'view': [view],
    'condition': [condition],
    'sqft_above': [sqft_above],
    'sqft_basement': [sqft_basement],
    'yr_build': [yr_build],
    'yr_renovated': [yr_renovated],
    'zipcode': [zipcode],
    'lat': [lat],
    'lot': [lot],
    'sqft_living15': [sqft_living15],
    'sqft_lot15': [sqft_lot15],
    'grade': [grade],
    'floors': [floors]

})


if st.button("🎯 Fiyatı Tahmin Et"):
    tahmin = model.predict(input_data)
    st.success(f"🏠 Tahmini Evin Fiyatı: ${tahmin[0]:,.2f}")

tahmin = model.predict(input_data)


st.markdown(f"""
        <div style="background-color:#e8f5e9;padding:20px;border-radius:10px;margin-top:20px;">
            <h3 style="color:red;">💰 Tahmini Fiyat:</h3>
            <h2 style="color:blue;">${tahmin[0]:,.2f}</h2>
        </div>
    """, unsafe_allow_html=True)



