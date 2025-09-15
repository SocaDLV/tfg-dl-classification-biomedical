import pandas as pd
import numpy as np

data = [
    ("Portátil MSI", 400, "MobileNetV2", 80.50, 210.0),
    ("Portátil MSI", 400, "ResNet18", 93.80, 28.10),
    ("Portátil MSI", 400, "YOLOv5s", 81.40, 25.30),
    ("RPi Zero 2 W", 20, "MobileNetV2", 74.70, 554.50),
    ("RPi Zero 2 W", 20, "ResNet18", 93.80, 289.15),
    ("RPi Zero 2 W", 20, "YOLOv5s", 81.00, 637.30),
    ("RPi Zero 2 W + NCS2", 60, "MobileNetV2", 73.50, 160.30),
    ("RPi Zero 2 W + NCS2", 60, "ResNet18", 93.20, 26.70),
    ("RPi Zero 2 W + NCS2", 60, "YOLOv5s", 82.21, 59.00),
    ("RPi 5", 80, "MobileNetV2", 73.20, 103.50),
    ("RPi 5", 80, "ResNet18", 93.80, 110.50),
    ("RPi 5", 80, "YOLOv5s", 82.50, 121.00),
    ("Jetson Nano", 220, "MobileNetV2", 72.80, 285.50),
    ("Jetson Nano", 220, "ResNet18", 93.87, 72.50),
    ("Jetson Nano", 220, "YOLOv5s", 81.37, 130.69),
    ("PC GPU RTU", 2100, "MobileNetV2", 80.80, 220.30),
    ("PC GPU RTU", 2100, "ResNet18", 94.10, 15.45),
    ("PC GPU RTU", 2100, "YOLOv5s", 79.50, 22.40),
]

df = pd.DataFrame(data, columns=["Dispositivo", "PVP", "Modelo", "Precisión", "Tiempo"])

resize_bonus = {
    "MobileNetV2": 0.15,
    "YOLOv5s": 0.10,
    "ResNet18": 0.0,
}
df["Resize Bonus"] = df["Modelo"].map(resize_bonus)

df["Norm_Precisión"] = (df["Precisión"] - df["Precisión"].min()) / (df["Precisión"].max() - df["Precisión"].min())
df["Norm_Tiempo"] = (df["Tiempo"].max() - df["Tiempo"]) / (df["Tiempo"].max() - df["Tiempo"].min())
df["Norm_PVP"] = (df["PVP"].max() - df["PVP"]) / (df["PVP"].max() - df["PVP"].min())

w_precision = 0.25
w_time = 0.35
w_cost = 0.25

df["Score"] = (
    w_precision * df["Norm_Precisión"] +
    w_time * df["Norm_Tiempo"] +
    w_cost * df["Norm_PVP"] +
    df["Resize Bonus"]
)

df_sorted = df.sort_values(by="Score", ascending=False).reset_index(drop=True)
print(df_sorted)