import json
import pandas as pd
import argparse
import sys
parser = argparse.ArgumentParser(description='Short sample app')



# 1) Point this at your actual log file
file_path = sys.argv[1]

# 2) Read in each line as JSON, extract the fields
records = []
with open(file_path, 'r') as f:
    for line in f:
        rec = json.loads(line)
        records.append({
            'idx':            rec.get('idx'),
            'time_total':     rec.get('time_total'),
            'time_llm':       rec.get('time_llm'),
            'prompt_length':  rec.get('prompt_length'),
        })

# 3) Build a DataFrame
df = pd.DataFrame(records)

# 4) Display the full table
# import ace_tools as tools
# tools.display_dataframe_to_user(name="Per‐Example Timings & Prompt Lengths", dataframe=df)

# 5) Compute and print the averages
averages = df[['time_total', 'time_llm', 'prompt_length']].mean()
print("\n⟶ Average time_total:    {:.2f} sec".format(averages['time_total']))
print("⟶ Average time_llm:      {:.2f} sec".format(averages['time_llm']))
print("⟶ Average prompt_length: {:.0f} chars".format(averages['prompt_length']))
