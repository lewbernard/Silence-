import numpy as np
import pandas as pd

# Set a fixed random seed so our test data is reproducible
np.random.seed(42)

num_samples_per_letter = 100
num_letters = 26
num_coordinates = 63

rows = []

for label in range(num_letters):
    # Create a unique base hand posture for each letter (0 to 25)
    base_pattern = np.sin(np.linspace(0, np.pi * (label + 1), num_coordinates))
    
    for _ in range(num_samples_per_letter):
        # Add slight random wobble (noise) to simulate natural human hand movement
        noise = np.random.normal(0, 0.15, size=num_coordinates)
        sample_coordinates = base_pattern + noise
        
        # Combine the integer label (Column 0) with the 63 coordinates
        row = [label] + sample_coordinates.tolist()
        rows.append(row)

# Save all 2,600 rows into dummy_data.csv without column headers
df = pd.DataFrame(rows)
df.to_csv("dummy_data.csv", index=False, header=False)
print("Successfully generated dummy_data.csv with 2,600 samples (26 letters x 100 samples)!")