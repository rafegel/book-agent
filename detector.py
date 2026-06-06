import cv2
import easyocr
from ultralytics import YOLO

# EasyOCR reader (ilk çalıştırmada model indirir)
reader = easyocr.Reader(["tr", "en"], gpu=False)

# YOLO — kitap sınıfı COCO dataset'te class 73
model = YOLO("yolov8n.pt")
BOOK_CLASS_ID = 73


def detect_book(frame) -> tuple[bool, list[int]]:
    """
    Karede kitap var mı kontrol eder.
    Dönüş: (bulundu_mu, [x1, y1, x2, y2])
    """
    results = model(frame, verbose=False)[0]
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if cls == BOOK_CLASS_ID and conf > 0.08:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            return True, [x1, y1, x2, y2]
    return False, []


def extract_text(frame, bbox: list[int]) -> str:
    """
    Tespit edilen bölgeden OCR ile metin okur.
    """
    x1, y1, x2, y2 = bbox

    # Kapak bölgesini kes, biraz büyüt (OCR daha iyi okur)
    padding = 10
    h, w = frame.shape[:2]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    crop = frame[y1:y2, x1:x2]

    # Önişleme: griye çevir, kontrastı artır
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=20)

    results = reader.readtext(gray, detail=0)
    text = " ".join(results).lower().strip()
    return text


def get_book_text(frame) -> tuple[bool, str, list[int]]:
    """
    Ana fonksiyon: karede kitap tespiti + OCR.
    Dönüş: (kitap_bulundu, okunan_metin, bbox)
    """
    found, bbox = detect_book(frame)
    if not found:
        return False, "", []

    text = extract_text(frame, bbox)
    return True, text, bbox
