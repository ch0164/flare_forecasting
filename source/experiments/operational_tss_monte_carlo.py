import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix


def main():
    ratio = 15 / 2725
    print(ratio * 100)

    scheme_names = ["random", "non_major", "major", "probabilistic"]
    tss = {scheme_name: [] for scheme_name in scheme_names}
    for trial_index in range(1000):
        scheme = {scheme_name: [] for scheme_name in scheme_names}
        actual_events = []
        # Generate a flare according to the ratio of flaring activity
        for flare in range(3000):
            r = np.random.random()
            if r < ratio:
                event = 1
            else:
                event = 0
            actual_events.append(event)

            # Use one of four schemes to predict flare class
            r2 = np.random.random()
            # 1) Random guessing
            if r2 < 0.5:
                scheme["random"].append(0)
            else:
                scheme["random"].append(1)
            # 2) Non-major
            scheme["non_major"].append(0)
            # 3) Major
            scheme["major"].append(1)
            # 4) Probabilistic
            if r2 < ratio:
                scheme["probabilistic"].append(1)
            else:
                scheme["probabilistic"].append(0)
        for scheme_name in scheme_names:
            cm = confusion_matrix(actual_events, scheme[scheme_name])
            tn, fp, fn, tp = cm.ravel()
            detection_rate = tp / float(tp + fn)
            false_alarm_rate = fp / float(fp + tn)
            tss[scheme_name].append(detection_rate - false_alarm_rate)
    for scheme_name in scheme_names:
        print(f"Scheme: {scheme_name}\n\tMean: {np.mean(tss[scheme_name])}\n\tStd: {np.std(tss[scheme_name])}\n")


if __name__ == "__main__":
    main()