# Perception KPI tools

## Introduction
KPI tools for perception algorithms, currently support evaluating of:
* obstacle detection
* trafficlight

---

## Environment setup

### Step 1. Create python virtual environment
```angular2html
conda create -n evaluation python=3.8
conda activate evaluation
```

### Step 2. Install required python site-packages
```angular2html
pip install -r requirements.txt
```

### Step 3. Install pypcd from source
```angular2html
git clone https://github.com/dimatura/pypcd
cd pypcd
git fetch origin pull/9/head:python3
git checkout python3
pip install -e . -v
```

---

## How to use

### Example 1: 
Run obstacle evaluation task with example data:
1. Example data already provided inside 'example_data/eval_obstacle'
2. The default value 'gt_data_path' and 'pred_data_path' in 'config/config_files/eval_obstacle_config.py' are set as the example data.
```angular2html
python eval_main.py --task_type 3d_object
```

### Example 2:
Run obstacle evaluation task with custom data:
1. Keep the format of gt files and predict file as example
2. Replace 'gt_data_path' and 'pred_data_path' in 'config/config_files/eval_obstacle_config.py' to your custom data path.
```angular2html
python eval_main.py --task_type 3d_object
```

and you can change ROI, target category, threshold of IOU, score, etc. in the configuration file.

