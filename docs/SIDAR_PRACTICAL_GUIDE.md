# Sidar Pratik Kullanım Kılavuzu (Web Dashboard)

Bu kılavuz, LotusAI sistemini başlattıktan sonra `http://localhost:5000/` üzerinden **Sidar’ın özel sohbet ekranında** Sidar yeteneklerini adım adım test etmeniz için hazırlanmıştır.

---

## 1) Başlangıç Kontrolü

Sistem açılış loglarında aşağıdakileri görmeniz, Sidar testleri için yeterlidir:

- `LotusAI.Sidar ... Teknik Liderlik modülü başlatıldı`
- `Aktif agent'lar: ... SIDAR ...`
- `Web Dashboard: http://localhost:5000`
- `Erişim: sandbox`

> Not: Kamera/mikrofon yoksa bile dashboard üzerinden metin tabanlı Sidar testleri yapılabilir.

---

## 2) Dashboard’da Sidar Sayfasına Geçiş

1. Tarayıcıdan `http://localhost:5000/` adresini açın.
2. Sol taraftaki ajan listesinde **Sidar (💻)** satırına tıklayın.
3. Sağ panel başlığında ajan adı **Sidar** olarak görünmelidir.
4. Bu andan sonra mesaj kutusundan gönderdiğiniz istekler `target_agent=SIDAR` ile işlenir.

---

## 3) Önemli Çalışma Modu Bilgisi (Sandbox)

Sizdeki çalıştırma çıktısına göre:

- Erişim seviyesi: `SANDBOX`
- Sidar modeli: `qwen2.5-coder:7b`

Bu modda Sidar:

- Dosya okuyabilir,
- Güvenli şekilde dosya yazabilir,
- Python/JSON doğrulaması yapar,
- Sistem/GPU denetimi yapabilir,
- Tehlikeli, sınır dışı veya yetkisiz işlemleri reddeder.

---

## 4) Sidar Yetenek Test Planı

Aşağıdaki testleri **sırayla** uygularsanız Sidar’ın ana yeteneklerini kapsarsınız.

### Test A — Proje Dosyalarını Listeleme

**Mesaj**

```text
Sidar, ana klasördeki dosyaları listele.
```

**Beklenen**

- Sidar çalışma dizini dosyalarını listeler.
- Yanıtta klasör/dosya çıktısı görünür.

---

### Test B — Dosya Okuma ve İnceleme

**Mesaj**

```text
Sidar, agents/sidar.py dosyasını oku ve kısaca özetle.
```

**Beklenen**

- Dosya içerik tabanlı analiz üretir.
- Uydurma kod yazmak yerine mevcut dosyaya dayanır.

---

### Test C — Sistem Denetimi (Audit)

**Mesaj**

```text
Sidar, sistemi tara ve teknik denetim raporu ver.
```

**Beklenen**

- Proje yapısı, GPU durumu, sağlık özeti, Python dosya sayısı gibi başlıklarla rapor gelir.
- “Kritik sorun tespit edilmedi” veya sorun listesi gibi net sonuç döner.

---

### Test D — GPU Optimizasyonu

**Mesaj**

```text
Sidar, GPU belleğini optimize et.
```

**Beklenen**

- GPU aktifse optimize sonucu ve serbest bırakılan VRAM bilgisi gelir.
- GPU aktif değilse uyarı mesajı dönebilir.

---

### Test E — Hata Analizi

**Mesaj**

```text
Sidar, şu hatayı analiz et: ModuleNotFoundError: No module named 'xyz'
```

**Beklenen**

- Hata tipi sınıflandırması,
- Ciddiyet seviyesi,
- Teşhis + çözüm önerisi verir.

---

### Test F — Güvenli Kod Yazma (Sandbox Uyumlu)

**Mesaj**

```text
Sidar, test/sidar_demo.py dosyasına "hello sidar" yazdıran basit bir fonksiyon ekle ve kaydet.
```

**Beklenen**

- Dosya yazma işlemi gerçekleştirir (izinli kapsamda).
- Python sözdizimi bozuksa kaydı reddeder.

> İsteğe bağlı doğrulama: Sonrasında “`Sidar, test/sidar_demo.py dosyasını oku`” deyip içeriği kontrol edin.

---

### Test G — GitHub Entegrasyonu (Token Geçersizse Negatif Test)

**Mesaj**

```text
Sidar, GitHub'daki son commitleri listele.
```

**Beklenen**

- GitHub token geçerliyse commit listesi döner.
- Token hatalıysa (sizdeki gibi `401 Bad credentials`) hata mesajını kontrollü döner.

Bu test, Sidar’ın **harici servis hata yönetimini** doğrulamak için önemlidir.

---

## 5) Kısa “Hazır Komut Seti” (Kopyala-Yapıştır)

```text
Sidar, ana klasördeki dosyaları listele.
Sidar, agents/sidar.py dosyasını oku ve özetle.
Sidar, sistemi tara ve teknik rapor ver.
Sidar, GPU belleğini optimize et.
Sidar, şu hatayı analiz et: SyntaxError: invalid syntax
Sidar, test/sidar_demo.py dosyasına basit bir hello fonksiyonu ekle.
Sidar, GitHub'daki son commitleri listele.
```

---

## 6) Sonuçları Değerlendirme Ölçütü

Sidar testiniz başarılı sayılır, eğer:

- En az 1 dosya okuma ve 1 denetim çıktısı aldınız,
- En az 1 hata analizinde teşhis+çözüm üretti,
- Sandbox modunda güvenli yazma yaptı veya kurala aykırı isteği reddetti,
- GitHub çağrısında başarılı sonuç veya anlamlı hata raporu verdi.

---

## 7) Sizin Çalıştırma Çıktınıza Göre Hızlı Notlar

- ✅ Sidar doğru yüklenmiş.
- ✅ Dashboard aktif (`localhost:5000`).
- ✅ Sandbox modu aktif (güvenli test için ideal).
- ⚠️ Kamera/mikrofon yok: Ses/kimlik akışlarını etkileyebilir ama web metin sohbeti için kritik değil.
- ⚠️ GitHub token hatalı (`401`): GitHub testlerinde negatif sonuç normaldir.

---

## 8) Önerilen Sonraki Adım

İlk turda Test A-B-C-D-E’yi uygulayın. Ardından Test F ile küçük bir yazma testi yapın. En sonda Test G’yi çalıştırıp GitHub kimlik doğrulamasını düzeltmeden önce mevcut davranışı belgeleyin.

Bu akış, Sidar’ın **okuma + analiz + denetim + optimizasyon + güvenli yazma + entegrasyon hata yönetimi** yeteneklerini kısa sürede kapsar.
