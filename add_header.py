import pandas as pd

# Define column names
cols = []
for i in range(21):
    cols.extend([f"x{i}", f"y{i}", f"z{i}"])
cols.append("label")

df = pd.read_csv("hand_data.csv", header=None)
df.columns = cols
df.to_csv("hand_data_with_header.csv", index=False)

print("Done! Saved as hand_data_with_header.csv")
