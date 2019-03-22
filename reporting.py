import os

import pandas as pd

csv_output_dir = "/home/tunahansalih/PycharmProjects/EE58J_Assignment_1/results/csv_results"
json_dir = "/home/tunahansalih/PycharmProjects/EE58J_Assignment_1/results/json_results"

for json_file in os.listdir(json_dir):
    df = pd.read_json(os.path.join(json_dir, json_file))
    df = df.sort_values(by=['score'], ascending=False)

    output_file_name = os.path.join(csv_output_dir, os.path.splitext(json_file)[0] + ".csv")
    df.to_csv(output_file_name)
