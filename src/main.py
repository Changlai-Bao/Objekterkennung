from ultralytics import YOLO
import cv2

# Modellauswahl
def modell_waehlen():
    print("Verfügbare Modelle:")
    print("1. YOLOv8n (nano)")
    print("2. YOLOv8s (small)") 
    print("3. YOLOv8m (medium)")
    print("4. YOLOv8l (large)")
    auswahl = input("Wählen Sie ein Modell (1-4): ")
    
    if auswahl == "1":
        return "src/models/yolov8n.pt"
    elif auswahl == "2":
        return "src/models/yolov8s.pt" 
    elif auswahl == "3":
        return "src/models/yolov8m.pt"
    elif auswahl == "4":
        return "src/models/yolov8l.pt"
    else:
        print("Ungültige Auswahl. Verwende Standardmodell (YOLOv8n).")
        return "src/models/yolov8n.pt"

# Hauptfunktion
def objekterkennung():
    # Modell laden
    modell_pfad = modell_waehlen()
    modell = YOLO(modell_pfad)
    
    # Kamera initialisieren
    kamera = cv2.VideoCapture(0)
    
    while True:
        # Bild von der Kamera lesen
        erfolg, bild = kamera.read()
        if not erfolg:
            print("Fehler beim Zugriff auf die Kamera.")
            break
        
        # Objekte erkennen
        ergebnisse = modell(bild)
        
        # Ergebnisse anzeigen
        annotiertes_bild = ergebnisse[0].plot()
        cv2.imshow("Objekterkennung", annotiertes_bild)
        
        # Beenden bei Drücken der 'q'-Taste
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Ressourcen freigeben
    kamera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    objekterkennung()
