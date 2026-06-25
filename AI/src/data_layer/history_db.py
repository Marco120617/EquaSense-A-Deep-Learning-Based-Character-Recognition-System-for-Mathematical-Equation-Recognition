import os
import time
import pandas as pd

def save_result(file_name, expected, recognized, accuracy, cer):
    result_file = "equasense_results.csv"
    new_data = pd.DataFrame([{
        "Image File": file_name,
        "Expected Equation": expected,
        "Recognized Equation": recognized,
        "Accuracy (%)": round(accuracy, 2),
        "Character Error Rate": round(cer, 4),
        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }])
    if os.path.exists(result_file):
        old_data = pd.read_csv(result_file)
        final_data = pd.concat([old_data, new_data], ignore_index=True)
    else:
        final_data = new_data
    final_data.to_csv(result_file, index=False)