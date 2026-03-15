*sample text*

## Environment setup

### Using python virtual environments

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

### Using conda

```
conda create -n hotel-booking-ml python=3.14 -y
conda activate hotel-booking-ml
pip install -r requirements.txt
```

---

`main.py` includes cell anotations and thus, can be opened as a `.ipynb` file. 
- In JupyterLab, open it as a notebook (right-click) and save to automatically create a `main.ipynb` file.

## Dataset
Download the zip file including the dataset and other files at [here](https://thiswebsitedoesnotexit.com). Extract the zip and place the `hotel_bookings_course_release_v1.csv` in a new `data/` folder from the root directory.
