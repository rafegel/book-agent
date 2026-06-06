import cv2
import time
from detector import get_book_text
from agent import match_topic, BookAgent

# ── Ayarlar ───────────────────────────────────────────────
CAMERA_INDEX = 0        # Farklı kamera için 1, 2... dene
SCAN_INTERVAL = 3.0     # Her kaç saniyede bir kitap taransın
CONFIRM_FRAMES = 2      # Kaç kez üst üste görülmeli (hatalı tetiklenmeyi önler)
# ──────────────────────────────────────────────────────────


def draw_overlay(frame, bbox, topic, score):
    """Kare üzerine tespit kutusunu ve bilgileri çizer."""
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 120), 2)
    label = f"Konu: {topic}  ({score:.0f}%)"
    cv2.putText(frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 120), 2)


def terminal_chat(agent: BookAgent):
    """Sohbet döngüsü — 'q' veya 'çıkış' ile ana döngüye döner."""
    print("\n" + "="*50)
    print(f"  Ajan hazır! Konu: {agent.topic.upper()}")
    print("  Sorunuzu yazın. Çıkmak için: q")
    print("="*50)

    while True:
        try:
            user_input = input("\nSen: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() in ("q", "çıkış", "exit"):
            print("Ana ekrana dönülüyor...\n")
            break

        print("Ajan: ", end="", flush=True)
        reply = agent.chat(user_input)
        print(reply)


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    time.sleep(2)
    if not cap.isOpened():
        print(f"[HATA] Kamera {CAMERA_INDEX} açılamadı.")
        return

    print("Kitap Ajanı başlatıldı. Kamerayı kitaba doğrult.")
    print("Pencereyi kapatmak için 'q' tuşuna bas.\n")

    last_scan = 0.0
    confirm_count = 0
    last_topic = None
    current_agent: BookAgent | None = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()

        # Belirli aralıklarla tara
        if now - last_scan >= SCAN_INTERVAL:
            last_scan = now
            found, text, bbox = get_book_text(frame)

            if found and text:
                match = match_topic(text)
                topic = match["topic"]

                # Aynı konuysa sayacı artır, farklıysa sıfırla
                if topic == last_topic:
                    confirm_count += 1
                else:
                    confirm_count = 1
                    last_topic = topic

                print(f"[Tespit] Metin: '{text[:60]}...' → {topic} (%{match['score']:.0f})")

                # Yeterince onaylandıysa ajanı başlat
                if confirm_count >= CONFIRM_FRAMES:
                    confirm_count = 0

                    # Yeni kitap — ajanı sıfırla
                    if current_agent is None or current_agent.topic != topic:
                        current_agent = BookAgent(
                            system_prompt=match["system_prompt"],
                            topic=topic,
                        )

                    draw_overlay(frame, bbox, topic, match["score"])
                    cv2.imshow("Kitap Ajanı", frame)
                    cv2.waitKey(1)

                    # Kamerayı kapat, sohbete gir
                    cap.release()
                    cv2.destroyAllWindows()
                    terminal_chat(current_agent)

                    # Sohbet bitti, kamerayı yeniden aç
                    cap = cv2.VideoCapture(CAMERA_INDEX)
                    last_scan = time.time()
                    continue
            else:
                confirm_count = 0
                last_topic = None

        # Ekrana yaz
        status = "Kitap aranıyor..." if last_topic is None else f"Tespit: {last_topic}"
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.imshow("Kitap Ajanı", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Uygulama kapatıldı.")


if __name__ == "__main__":
    main()
