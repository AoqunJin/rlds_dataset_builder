# LIBERO to RLDS format

## Install

Please run the following commands in the given order to install the dependency for LIBERO.

```bash
conda create -n libero python=3.8.13
conda activate libero
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

Then install the libero package:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
```

## Download dataset (collected by human)

They provide high-quality human teleoperation demonstrations for the four task suites in **LIBERO**. To download the demonstration dataset, run:
```python
python benchmark_scripts/download_libero_datasets.py
```
By default, the dataset will be stored under the ```LIBERO``` folder and all four datasets will be downloaded. To download a specific dataset, use
```python
python benchmark_scripts/download_libero_datasets.py --datasets DATASET
```
where ```DATASET``` is chosen from `[libero_spatial, libero_object, libero_100, libero_goal`.

## Run regenerate

```bash
python regenerate_libero_dataset.py \
    --libero_task_suite [ libero_spatial | libero_object | libero_goal | libero_10 ] \
    --libero_raw_data_dir <PATH TO RAW HDF5 DATASET DIR> \
    --libero_target_dir <PATH TO TARGET DIR>
```

Dataset info:

`libero_spatial`: (Different layouts, same objects) - 50 tral x 10 sub_task
1. pick up the black bowl between the plate and the ramekin and place it on the plate
2. pick up the black bowl next to the ramekin and place it on the plate
3. pick up the black bowl from table center and place it on the plate
4. pick up the black bowl on the cookie box and place it on the plate
5. pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate
6. pick up the black bowl on the ramekin and place it on the plate
7. pick up the black bowl next to the cookie box and place it on the plate
8. pick up the black bowl on the stove and place it on the plate
9. pick up the black bowl next to the plate and place it on the plate
10. pick up the black bowl on the wooden cabinet and place it on the plate

`libero_object`: (Different objects, same layout) - 50 tral x 10 sub_task
1. pick up the alphabet soup and place it in the basket
2. pick up the cream cheese and place it in the basket
3. pick up the salad dressing and place it in the basket
4. pick up the bbq sauce and place it in the basket
5. pick up the ketchup and place it in the basket
6. pick up the tomato sauce and place it in the basket
7. pick up the butter and place it in the basket
8. pick up the milk and place it in the basket
9. pick up the chocolate pudding and place it in the basket
10. pick up the orange juice and place it in the basket

`libero_goal`: (Different goals, same objects & layout)
1. open the middle drawer of the cabinet
2. put the bowl on the stove
3. put the wine bottle on top of the cabinet
4. open the top drawer and put the bowl inside
5. put the bowl on top of the cabinet
6. push the plate to the front of the stove
7. put the cream cheese in the bowl
8. turn on the stove
9. put the bowl on the plate
10. put the wine bottle on the rack

`libero_90`: (Diverse objects, layouts, backgrounds, short-horizon task)
1. ...

`libero_10`: (Diverse objects, layouts, backgrounds, long-horizon task)
1. put both the alphabet soup and the tomato sauce in the basket
2. put both the cream cheese box and the butter in the basket
3. turn on the stove and put the moka pot on it
4. put the black bowl in the bottom drawer of the cabinet and close it
5. put the white mug on the left plate and put the yellow and white mug on the right plate
6. pick up the book and place it in the back compartment of the caddy
7. put the white mug on the plate and put the chocolate pudding to the right of the plate
8. put both the alphabet soup and the cream cheese box in the basket
9. put both moka pots on the stove
10. put the yellow and white mug in the microwave and close it

## Build dataset
