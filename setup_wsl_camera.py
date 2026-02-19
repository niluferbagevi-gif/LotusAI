"""
LotusAI - WSL Kamera & USB Otomatik Kurulum Aracı
Sürüm: 1.6
Açıklama: Yeni bir WSL/Ubuntu kurulumunda veya Windows yeniden başlatıldığında
kamerayı otomatik olarak bulur, Windows yetkisiyle WSL'e bağlar ve Linux izinlerini verir.
(Base64 Encoded UAC Command ile boşluk/tırnak hataları giderilmiştir)
"""

import os
import sys
import time
import glob
import subprocess
import re
import base64

def is_wsl():
    """Sistemin WSL üzerinde çalışıp çalışmadığını kontrol eder."""
    try:
        with open("/proc/version", "r") as f:
            return "microsoft" in f.read().lower()
    except Exception:
        return False

def setup_wsl_camera():
    print("="*65)
    print(" 🚀 LotusAI - WSL Kamera & USB Otomatik Kurulum Aracı (v1.6)")
    print("="*65)

    # 1. WSL kontrolü
    if not is_wsl():
        print("❌ Bu araç yalnızca WSL (Ubuntu/Windows) ortamı için tasarlanmıştır.")
        sys.exit(1)

    print("\n🔍 1. Windows üzerindeki USB kameralar taranıyor...")
    
    # WSL Path sorunlarını aşmak için Windows'taki tam dosya yolunu kullanıyoruz
    usbipd_path = "C:\\Program Files\\usbipd-win\\usbipd.exe"
    
    try:
        # PowerShell'e komutu doğrudan dosya konumu ile gönderiyoruz
        result = subprocess.run(
            ["powershell.exe", "-Command", f"& '{usbipd_path}' list"], 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Format çözümleme
        try:
            output = result.stdout.decode('utf-8', errors='ignore')
            error_output = result.stderr.decode('utf-8', errors='ignore')
        except Exception:
            output = result.stdout.decode('cp1254', errors='ignore') 
            error_output = result.stderr.decode('cp1254', errors='ignore')
            
        output = output.replace('\x00', '')
        error_output = error_output.replace('\x00', '')
        
    except Exception as e:
        print(f"❌ PowerShell tetiklenirken kritik bir hata oluştu: {e}")
        sys.exit(1)

    busid = None
    device_name = None
    
    # Çıktıyı analiz et
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    for line in lines:
        if any(kw in line.lower() for kw in ["camera", "webcam", "video", "uvc"]):
            # BUSID değerini yakala
            match = re.match(r'^([\d\-]+)\s', line)
            if match:
                busid = match.group(1)
                device_name = line.split("  ")[-2].strip() if "  " in line else "Kamera"
                break

    if not busid:
        print("❌ Windows tarafında uygun bir kamera bulunamadı veya usbipd dizini hatalı.")
        print("\n--- Sistemden Gelen HATA (STDERR) ---")
        print(error_output if error_output.strip() else "Hata mesajı yok.")
        print("---------------------------------------\n")
        sys.exit(1)

    print(f"✅ Kamera Bulundu: {device_name} (BUSID: {busid})")
    print("\n⚙️  2. Kamera WSL sistemine aktarılıyor...")
    print("⚠️  DİKKAT: Ekranda Windows Yönetici (UAC) izin penceresi çıkabilir, lütfen 'Evet' diyerek onaylayın.")
    
    # Doğrudan tam yol kullanarak UAC izniyle işlemi başlatıyoruz
    # Tırnak hatalarını önlemek için komutu UTF-16LE Base64 formatına çeviriyoruz
    ps_bind_attach = f"& '{usbipd_path}' bind --busid {busid}; & '{usbipd_path}' attach --wsl --busid {busid}"
    encoded_command = base64.b64encode(ps_bind_attach.encode('utf-16-le')).decode('utf-8')
    
    uac_command = f"Start-Process powershell -ArgumentList '-NoProfile -ExecutionPolicy Bypass -EncodedCommand {encoded_command}' -Verb RunAs"
    
    subprocess.run(["powershell.exe", "-Command", uac_command])
    
    print("⏳ Kameranın Linux (Ubuntu) ortamına geçmesi bekleniyor (5 saniye)...")
    time.sleep(5)

    print("\n🔐 3. Linux /dev/video port izinleri ayarlanıyor...")
    devices = glob.glob("/dev/video*")
    
    if devices:
        try:
            # İşlemi otomatik algılaması için uyarı ekledik
            print("Lütfen istendiğinde Ubuntu parolanızı girin:")
            subprocess.run(["sudo", "chmod", "777"] + devices)
            print(f"✅ İzinler başarıyla verildi: {', '.join(devices)}")
        except Exception as e:
            print(f"❌ İzin verilirken hata oluştu: {e}")
    else:
        print("⚠️ /dev/video portları bulunamadı. Windows tarafında kamera kullanımda/kilitli olabilir.")

    print("\n🎉 İŞLEM TAMAM! LotusAI artık kameranızı kullanabilir.")
    print("Hemen başlatmak için: python main.py")
    print("="*65)

if __name__ == "__main__":
    setup_wsl_camera()