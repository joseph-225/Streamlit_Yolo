from ultralytics import YOLO
from playsound3 import playsound


model = YOLO("/home/wecode-063/Rendu_Data/C-DAT-900-ABJ-2-2-ecp-joseph.m-baye/best.pt")
alerte = model.predict(source='Dataset/train/images/y2mate_com-CCTV-CAR-CRASHES-COMPILATION-2018-EP-20_480p_mp4-40_jpg.rf.f8adac26a3b8995d4b1490f0276a7c65.jpg', imgsz=640, conf=0.35, save=True, show = True)
sound = "Sound/UNE MENACE A ETE DETECTEE VIDEO OFFICIELLES.mp3"
# while True:
for alert in alerte:
    if alert.boxes:
                print("Accident detecté !")
                playsound(sound)
                break