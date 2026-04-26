# Unsupervised Learning Project

Title: Booking complexity & behavior dynamics
Core question: Can hotel bookings be segmented by complexity and reservation-management behavior (lead time, modifications, special requests, planning horizon)?
Operational goal: Distinguish low-friction from high-touch bookings to optimize staff allocation and enable proactive service strategies.

Dataset: 119,390 bookings × 32 features (data/hotel_bookings_course_release_v1.csv, ~16.8 MB, gitignored). A reproducible 30k-row subsample is specified in data/subsample_indices_v1_n30000_seed12345.txt.


## Setup
### Obtaining the dataset
The dataset is available at https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand/data. After downloading, extract the zip to obtain `hotel_bookings.csv`. Rename this file to `hotel_bookings_course_release_v1.csv` and place it inside the `data` folder to run any experiments.
> You can (and should) verify the SHA256 hash of the file against the respective hash in `docs/dataset/SHA256SUMS.txt`.

### Environment setup

#### Using python virtual environments

Create environment with dependencies:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install kernel at the user level.
```
python3 -m ipykernel install --user --name=hotel-booking-ml --display-name="Python UL" 
```

Or, install kernel at the project level (tied to active environemnt, (e.g., current python venv)).
```
python3 -m ipykernel install --sys-prefix \
    --name=hotel-booking-ml \
    --display-name="Python UL"

```

#### Using conda

```
conda create -n hotel-booking-ml python=3.14 -y
conda activate hotel-booking-ml
pip install -r requirements.txt
```