"""
LotusAI - Çoklu Ekran Yöneticisi (Screen Manager)
Açıklama: Windows üzerindeki bağlı monitörleri tespit edip,
PyGame/GUI arayüzünü otomatik olarak hedeflenen ekrana taşır.
"""

import os
import json
import subprocess
import logging

logger = logging.getLogger("LotusAI.ScreenManager")

def get_windows_screens():
    """Windows PowerShell kullanarak bağlı ekranların koordinatlarını çeker."""
    try:
        ps_script = """
        Add-Type -AssemblyName System.Windows.Forms
        $screens = [System.Windows.Forms.Screen]::AllScreens
        $result = @()
        foreach ($s in $screens) {
            $result += @{
                DeviceName = $s.DeviceName
                X = $s.Bounds.X
                Y = $s.Bounds.Y
                Width = $s.Bounds.Width
                Height = $s.Bounds.Height
                IsPrimary = $s.Primary
            }
        }
        $result | ConvertTo-Json -Compress
        """
        
        # Karakter kodlaması ve hata yoksayma eklendi
        result = subprocess.run(
            ["powershell.exe", "-Command", ps_script], 
            capture_output=True, 
            text=True,
            encoding='utf-8',
            errors='ignore'
        )
        
        output = result.stdout.strip()
        if output:
            screens = json.loads(output)
            # Eğer sadece 1 ekran varsa PowerShell bunu dict olarak döner, liste yaparız
            if isinstance(screens, dict):
                screens = [screens]
            return screens
            
        return []
        
    except Exception as e:
        logger.error(f"Ekranlar alınırken hata oluştu: {e}")
        return []

def set_target_screen(target_index: int = 1):
    """
    Hedef ekranı ayarlar.
    0: Ana Ekran, 1: İkinci Ekran, 2: Üçüncü Ekran vs.
    """
    screens = get_windows_screens()
    
    if not screens:
        logger.warning("⚠️ Windows ekran bilgileri alınamadı, ana ekran kullanılacak.")
        return False

    # İstenen ekran sayısı, mevcut ekrandan fazlaysa ana ekrana (0) düş
    if target_index >= len(screens):
        logger.warning(f"⚠️ Hedef ekran ({target_index}) bulunamadı! Toplam {len(screens)} ekran var. Ana ekrana dönülüyor.")
        target_index = 0 

    target = screens[target_index]
    x, y = target['X'], target['Y']
    
    # WSL / Linux / PyGame (SDL2) için pencere başlangıç pozisyonunu zorunlu ayarla
    os.environ['SDL_VIDEO_WINDOW_POS'] = f"{x},{y}"
    
    print(f"🖥️  Arayüz Ekran {target_index} üzerine taşındı (Çözünürlük: {target['Width']}x{target['Height']}, Konum: X={x} Y={y})")
    return True

# ═══════════════════════════════════════════════════════════════
# TEST BLOĞU (Doğrudan çalıştırıldığında burası tetiklenir)
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("="*60)
    print(" 🔍 LotusAI - Windows Ekranları Taranıyor...")
    print("="*60)
    
    screens = get_windows_screens()
    
    if not screens:
        print("❌ Hiç ekran bulunamadı veya PowerShell komutu çalıştırılamadı.")
    else:
        print(f"✅ Toplam {len(screens)} ekran bulundu:\n")
        
        for i, s in enumerate(screens):
            primary_tag = "(ANA EKRAN)" if s.get('IsPrimary') else ""
            print(f"  [{i}] {s.get('DeviceName')} {primary_tag}")
            print(f"      Çözünürlük : {s.get('Width')}x{s.get('Height')}")
            print(f"      Başlangıç  : X={s.get('X')}, Y={s.get('Y')}\n")
        
        print("-" * 60)
        # Test amaçlı: 2. ekran varsa ona, yoksa 1. ekrana odaklan
        test_index = 1 if len(screens) > 1 else 0
        print(f"Test İşlemi: Hedef Ekran {test_index} olarak ayarlanıyor...")
        
        set_target_screen(test_index)
        
        print(f"SDL_VIDEO_WINDOW_POS Değeri: {os.environ.get('SDL_VIDEO_WINDOW_POS')}")
    print("="*60)