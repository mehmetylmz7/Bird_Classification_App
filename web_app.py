import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import os
import pandas as pd
import zipfile
import shutil

# ==========================================
# 1. AYARLAR VE SINIF İSİMLERİ
# ==========================================
# Model dosyasının adı (Script ile aynı klasörde olmalı)
MODEL_PATH = 'model_son.pth'
IMG_SIZE = 380
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 200 Kuş Türü Listesi
CLASS_NAMES = [
    "Acadian_Flycatcher", "American_Crow", "American_Goldfinch", "American_Pipit", "American_Redstart",
    "American_Three_toed_Woodpecker", "Anna_Hummingbird", "Artic_Tern", "Baird_Sparrow", "Baltimore_Oriole",
    "Bank_Swallow", "Barn_Swallow", "Bay_breasted_Warbler", "Belted_Kingfisher", "Bewick_Wren",
    "Black_Tern", "Black_and_white_Warbler", "Black_billed_Cuckoo", "Black_capped_Vireo", "Black_footed_Albatross",
    "Black_throated_Blue_Warbler", "Black_throated_Sparrow", "Blue_Grosbeak", "Blue_Jay", "Blue_headed_Vireo",
    "Blue_winged_Warbler", "Boat_tailed_Grackle", "Bobolink", "Bohemian_Waxwing", "Brandt_Cormorant",
    "Brewer_Blackbird", "Brewer_Sparrow", "Bronzed_Cowbird", "Brown_Creeper", "Brown_Pelican",
    "Brown_Thrasher", "Cactus_Wren", "California_Gull", "Canada_Warbler", "Cape_Glossy_Starling",
    "Cape_May_Warbler", "Cardinal", "Carolina_Wren", "Caspian_Tern", "Cedar_Waxwing",
    "Cerulean_Warbler", "Chestnut_sided_Warbler", "Chipping_Sparrow", "Chuck_will_Widow", "Clark_Nutcracker",
    "Clay_colored_Sparrow", "Cliff_Swallow", "Common_Raven", "Common_Tern", "Common_Yellowthroat",
    "Crested_Auklet", "Dark_eyed_Junco", "Downy_Woodpecker", "Eared_Grebe", "Eastern_Towhee",
    "Elegant_Tern", "European_Goldfinch", "Evening_Grosbeak", "Field_Sparrow", "Fish_Crow",
    "Florida_Jay", "Forsters_Tern", "Fox_Sparrow", "Frigatebird", "Gadwall",
    "Geococcyx", "Glaucous_winged_Gull", "Golden_winged_Warbler", "Grasshopper_Sparrow", "Gray_Catbird",
    "Gray_Kingbird", "Gray_crowned_Rosy_Finch", "Great_Crested_Flycatcher", "Great_Grey_Shrike", "Green_Jay",
    "Green_Kingfisher", "Green_Violetear", "Green_tailed_Towhee", "Groove_billed_Ani", "Harris_Sparrow",
    "Heermann_Gull", "Henslow_Sparrow", "Herring_Gull", "Hooded_Merganser", "Hooded_Oriole",
    "Hooded_Warbler", "Horned_Grebe", "Horned_Lark", "Horned_Puffin", "House_Sparrow",
    "House_Wren", "Indigo_Bunting", "Ivory_Gull", "Kentucky_Warbler", "Laysan_Albatross",
    "Lazuli_Bunting", "Le_Conte_Sparrow", "Least_Auklet", "Least_Flycatcher", "Least_Tern",
    "Lincoln_Sparrow", "Loggerhead_Shrike", "Long_tailed_Jaeger", "Louisiana_Waterthrush", "Magnolia_Warbler",
    "Mallard", "Mangrove_Cuckoo", "Marsh_Wren", "Mockingbird", "Mourning_Warbler",
    "Myrtle_Warbler", "Nashville_Warbler", "Nelson_Sharp_tailed_Sparrow", "Nighthawk", "Northern_Flicker",
    "Northern_Fulmar", "Northern_Waterthrush", "Olive_sided_Flycatcher", "Orange_crowned_Warbler", "Orchard_Oriole",
    "Ovenbird", "Pacific_Loon", "Painted_Bunting", "Palm_Warbler", "Parakeet_Auklet",
    "Pelagic_Cormorant", "Philadelphia_Vireo", "Pied_Kingfisher", "Pied_billed_Grebe", "Pigeon_Guillemot",
    "Pileated_Woodpecker", "Pine_Grosbeak", "Pine_Warbler", "Pomarine_Jaeger", "Prairie_Warbler",
    "Prothonotary_Warbler", "Purple_Finch", "Red_bellied_Woodpecker", "Red_breasted_Merganser", "Red_cockaded_Woodpecker",
    "Red_eyed_Vireo", "Red_faced_Cormorant", "Red_headed_Woodpecker", "Red_legged_Kittiwake", "Red_winged_Blackbird",
    "Rhinoceros_Auklet", "Ring_billed_Gull", "Ringed_Kingfisher", "Rock_Wren", "Rose_breasted_Grosbeak",
    "Ruby_throated_Hummingbird", "Rufous_Hummingbird", "Rusty_Blackbird", "Sage_Thrasher", "Savannah_Sparrow",
    "Sayornis", "Scarlet_Tanager", "Scissor_tailed_Flycatcher", "Scott_Oriole", "Seaside_Sparrow",
    "Shiny_Cowbird", "Slaty_backed_Gull", "Song_Sparrow", "Sooty_Albatross", "Spotted_Catbird",
    "Summer_Tanager", "Swainson_Warbler", "Tennessee_Warbler", "Tree_Sparrow", "Tree_Swallow",
    "Tropical_Kingbird", "Vermilion_Flycatcher", "Vesper_Sparrow", "Warbling_Vireo", "Western_Grebe",
    "Western_Gull", "Western_Meadowlark", "Western_Wood_Pewee", "Whip_poor_Will", "White_Pelican",
    "White_breasted_Kingfisher", "White_breasted_Nuthatch", "White_crowned_Sparrow", "White_eyed_Vireo", "White_necked_Raven",
    "White_throated_Sparrow", "Wilson_Warbler", "Winter_Wren", "Worm_eating_Warbler", "Yellow_Warbler",
    "Yellow_bellied_Flycatcher", "Yellow_billed_Cuckoo", "Yellow_breasted_Chat", "Yellow_headed_Blackbird", "Yellow_throated_Vireo"
]

NUM_CLASSES = len(CLASS_NAMES)

# ==========================================
# 2. MODELİ YÜKLEME
# ==========================================
@st.cache_resource
def load_trained_model():
    """Modeli yükler ve cache'ler (her defasında tekrar yüklemez)"""
    try:
        model = models.efficientnet_b4(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
        
        # CPU veya GPU'ya göre yükle
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        model.to(DEVICE)
        model.eval()
        return model
    except FileNotFoundError:
        st.error(f"❌ '{MODEL_PATH}' bulunamadı! Lütfen model dosyasını bu script ile aynı klasöre koyun.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Beklenmedik hata: {e}")
        st.stop()

# ==========================================
# 3. GÖRÜNTÜ İŞLEME VE TAHMİN
# ==========================================
preprocess = transforms.Compose([
    transforms.Resize(IMG_SIZE + 32),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(image, model, top_k=3):
    img_t = preprocess(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
    
    top_probs, top_indices = torch.topk(probs, top_k)
    results = []
    for i in range(top_k):
        idx = top_indices[i].item()
        prob = top_probs[i].item()
        results.append((CLASS_NAMES[idx], prob))
    return results

# ==========================================
# 4. ARAYÜZ (UI) TASARIMI
# ==========================================
st.set_page_config(page_title="Kuş Türü Tanıma AI", page_icon="🦅", layout="wide")

st.title("🦅 Yapay Zeka Kuş Gözlemcisi")
st.markdown(f"**Model:** EfficientNet-B4 | **Sınıf:** {NUM_CLASSES} Tür | **Cihaz:** `{DEVICE}`")
st.markdown("---")

# Modeli Yükle
with st.spinner('Model yükleniyor...'):
    model = load_trained_model()

# Yan Menü
st.sidebar.header("İşlem Seçimi")
mode = st.sidebar.radio("Mod Seçiniz:", ["📷 Tek Resim Analizi", "📦 Toplu ZIP Testi"])

# --- MOD 1: TEK RESİM ANALİZİ ---
if mode == "📷 Tek Resim Analizi":
    st.header("Fotoğraf Yükle ve Tanı")
    uploaded_file = st.file_uploader("Bir kuş fotoğrafı yükleyin...", type=["jpg", "jpeg", "png", "webp"])
    
    if uploaded_file:
        col1, col2 = st.columns([1, 1])
        image = Image.open(uploaded_file)
        
        with col1:
            st.image(image, caption="Yüklenen Resim", use_column_width=True)
            
        with col2:
            st.subheader("Tahmin Sonuçları")
            predictions = predict_image(image, model)
            
            # En güçlü tahmin
            top_bird, top_score = predictions[0]
            st.success(f"**Tanımlanan:** {top_bird}")
            
            st.markdown("---")
            st.write("**Olasılık Dağılımı:**")
            
            for name, score in predictions:
                st.write(f"**{name}**")
                st.progress(score)
                st.caption(f"Güven Oranı: %{score*100:.2f}")

# --- MOD 2: TOPLU ZIP TESTİ ---
elif mode == "📦 Toplu ZIP Testi":
    st.header("Toplu Test ve Raporlama")
    st.info("ZIP dosyasının içinde, kuş türlerinin isimleriyle klasörler olmalıdır. Örn: `Serce/resim1.jpg`")
    
    zip_file = st.file_uploader("Test verilerini içeren ZIP dosyasını yükleyin", type="zip")
    
    if zip_file and st.button("Analizi Başlat"):
        temp_dir = "temp_dataset"
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)
        
        with zipfile.ZipFile(zip_file, 'r') as z:
            z.extractall(temp_dir)
            
        results = []
        correct_count = 0
        total_count = 0
        
        st_bar = st.progress(0)
        status_text = st.empty()
        
        # Dosyaları topla
        image_files = []
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(root, file))
        
        total_images = len(image_files)
        
        for i, file_path in enumerate(image_files):
            try:
                true_label = os.path.basename(os.path.dirname(file_path))
                
                img = Image.open(file_path)
                pred_list = predict_image(img, model, top_k=1)
                pred_label = pred_list[0][0]
                confidence = pred_list[0][1]
                
                is_correct = (true_label == pred_label)
                if is_correct: correct_count += 1
                
                results.append({
                    "Dosya": os.path.basename(file_path),
                    "Gerçek": true_label,
                    "Tahmin": pred_label,
                    "Güven": f"%{confidence*100:.1f}",
                    "Durum": "✅" if is_correct else "❌"
                })
                
                total_count += 1
                st_bar.progress((i + 1) / total_images)
                status_text.text(f"İşleniyor: {i+1}/{total_images}")
                
            except Exception as e:
                pass
        
        shutil.rmtree(temp_dir)
        st_bar.empty()
        status_text.empty()
        
        if total_count > 0:
            accuracy = (correct_count / total_count) * 100
            st.success(f"İşlem Tamamlandı! Toplam Resim: {total_count}")
            
            metric_col1, metric_col2 = st.columns(2)
            metric_col1.metric("Doğruluk Oranı", f"%{accuracy:.2f}")
            metric_col2.metric("Doğru Bilinen", f"{correct_count} / {total_count}")
            
            st.subheader("Detaylı Rapor")
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Raporu İndir (CSV)", csv, "kus_tahmin_raporu.csv", "text/csv")
        else:
            st.warning("ZIP dosyasında uygun resim bulunamadı.")