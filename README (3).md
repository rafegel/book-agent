# 📚 Kitap Ajanı

Kameraya gösterdiğin kitabı tanıyan, o konuya özel bir yapay zeka ajan başlatan sistem.

## Nasıl Çalışır?

1. Kamera açılır, YOLO ile kitap aranır
2. Kitap bulununca OCR ile kapak metni okunur
3. Metin `books.json` ile eşleştirilir → konu belirlenir
4. Ollama'dan o konuya özel ajan çağrılır
5. Terminalde sohbet başlar

## Kurulum

```bash
pip install -r requirements.txt
ollama pull llama3.2
```

## Çalıştırma

```bash
python main.py
```

## Yeni Kitap Eklemek

`books.json` dosyasına yeni bir blok ekle:

```json
{
  "keywords": ["anahtar", "kelimeler"],
  "topic": "konu_adi",
  "system_prompt": "Sen ... öğretmenisin."
}
```

## Dosya Yapısı

```
book-agent/
├── main.py        # Ana döngü (kamera + akış kontrolü)
├── detector.py    # YOLO tespiti + OCR
├── agent.py       # Ollama ajan + konu eşleştirme
├── books.json     # Kitap-konu veritabanı
└── requirements.txt
```
