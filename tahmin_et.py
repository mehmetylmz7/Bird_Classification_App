import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# ==========================================
# 1. AYARLAR
# ==========================================
# Model dosyasının adı (Bu dosya script ile aynı klasörde olmalı)
MODEL_PATH = 'model_son.pth' 

# Resim Boyutu (EfficientNet-B4 için 380)
IMG_SIZE = 380

# Cihaz Seçimi (Varsa GPU, yoksa CPU kullanır)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. SINIF İSİMLERİ (Sizin Listeniz)
# ==========================================
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
# 3. MODELİ YÜKLEME FONKSİYONU
# ==========================================
def load_trained_model():
    print(f"🤖 Model yükleniyor: {MODEL_PATH}")
    print(f"⚙️ Cihaz: {DEVICE}")
    
    # EfficientNet-B4 iskeletini oluştur
    model = models.efficientnet_b4(weights=None)
    
    # Son katmanı 200 sınıfa göre ayarla
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
    
    # Ağırlıkları yükle
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        model.to(DEVICE)
        model.eval() # Değerlendirme modu (Dropout vb. kapatır)
        print("✅ Model başarıyla yüklendi!")
        return model
    except FileNotFoundError:
        print(f"❌ HATA: '{MODEL_PATH}' dosyası bulunamadı!")
        print("Lütfen model dosyasını bu script ile aynı klasöre koyun.")
        exit()
    except Exception as e:
        print(f"❌ Beklenmedik hata: {e}")
        exit()

# Global olarak modeli bir kere yükle
model = load_trained_model()

# ==========================================
# 4. GÖRÜNTÜ ÖN İŞLEME VE TAHMİN
# ==========================================
# Eğitimdeki 'test' transformasyonlarının aynısı
preprocess = transforms.Compose([
    transforms.Resize(IMG_SIZE + 32),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def tahmin_et(image_path, top_k=3):
    """
    Bir resim yolunu alır ve tahmin sonuçlarını döndürür.
    """
    if not os.path.exists(image_path):
        return f"❌ Hata: Dosya bulunamadı -> {image_path}"

    try:
        # Resmi aç ve dönüştür
        img = Image.open(image_path).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
        
        # Tahmin yap
        with torch.no_grad():
            outputs = model(img_tensor)
            # Softmax ile olasılıklara çevir
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            
            # En yüksek k tahmini al
            top_probs, top_indices = torch.topk(probs, top_k)
            
        # Sonuçları formatla
        results = []
        print(f"\n📸 Analiz edilen resim: {image_path}")
        print("-" * 30)
        
        for i in range(top_k):
            idx = top_indices[i].item()
            score = top_probs[i].item() * 100
            bird_name = CLASS_NAMES[idx]
            
            print(f"{i+1}. Tahmin: {bird_name} (%{score:.2f})")
            results.append((bird_name, score))
            
        return results[0][0] # En yüksek ihtimalli kuşun ismini döndür

    except Exception as e:
        return f"❌ İşlem hatası: {e}"

# ==========================================
# 5. KULLANIM ÖRNEĞİ (Test Kısmı)
# ==========================================
if __name__ == "__main__":
    # Kullanıcıdan resim yolu iste
    print("\n🦅 KUŞ TÜRÜ TAHMİN SİSTEMİ 🦅")
    print("Çıkmak için 'q' yazın.\n")
    
    while True:
        resim_yolu = input("Resim yolunu girin (örn: kus.jpg): ")
        
        if resim_yolu.lower() == 'q':
            print("Çıkış yapılıyor...")
            break
            
        # Tırnak işaretlerini temizle (Sürükle bırak yapınca oluşabilir)
        resim_yolu = resim_yolu.strip('"').strip("'")
        
        if resim_yolu:
            tahmin_et(resim_yolu)