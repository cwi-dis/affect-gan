import pandas as pd
import numpy as np

def data_label_merger(file_count):

    for i in range(1, file_count+1):
        print(i)
        data = pd.read_csv(f"../../Dataset/CASE_dataset/interpolated/physiological/sub_{i}.csv")
        labels = pd.read_csv(f"../../Dataset/CASE_dataset/interpolated/annotations/sub_{i}.csv")

        interpolated_valence = np.interp(data["daqtime"], labels["jstime"], labels["valence"])
        interpolated_arousal = np.interp(data["daqtime"], labels["jstime"], labels["arousal"])

        data["Valence"] = interpolated_valence
        data["Arousal"] = interpolated_arousal

        data.to_csv(f"../../Dataset/CASE_dataset/merged/sub_{i}.csv", encoding="utf-8", index=False, float_format='%g')

if __name__ == '__main__':
    data_label_merger(30)