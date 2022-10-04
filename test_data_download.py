import os
from pathlib import Path

data_download_message = """
These are the necessary files in the folder structure the code expects.
```
data
├── london_smart_meters
│   ├── hhblock_dataset
│   │   ├── hhblock_dataset
│   │       ├── block_0.csv
│   │       ├── block_1.csv
│   │       ├── ...
│   │       ├── block_109.csv
│── acorn_details.csv
├── informations_households.csv
├── uk_bank_holidays.csv
├── weather_daily_darksky.csv
├── weather_hourly_darksky.csv
```
"""

def check_downloaded_data():
    root = Path("data/london_smart_meters")
    assert root.exists(), f"{data_download_message}"

    chck_files = ['acorn_details.csv',
    'hhblock_dataset',
    'informations_households.csv',
    'uk_bank_holidays.csv',
    'weather_daily_darksky.csv',
    'weather_hourly_darksky.csv']

    dir_files = os.listdir(root)
    assert all([f in dir_files for f in chck_files]), f"{data_download_message}"
    hhblock_root = root/"hhblock_dataset"/"hhblock_dataset"
    assert hhblock_root.exists(), f"{data_download_message}"
    assert all([(hhblock_root/f"block_{i}.csv").exists() for i in range(110)]), f"{data_download_message}"
    print("#"*25+" All data downloaded correctly! "+"#"*25)

if __name__=="__main__":
    check_downloaded_data()