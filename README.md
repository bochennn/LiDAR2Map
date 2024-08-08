# MMdetection3d Plugin


## Installation ##
```bash
pip3 install -r requirements.txt
python3 setup.py build_ext --inplace
```

## Getting Started ##

```bash
# prepare data
bash ./tools/scripts/create_data.sh zdrive --root-path $DATA_ROOT --workers 8

# training
bash ./tools/scripts/dist_train.sh $NUM_GPU --config $CONFIG --work-dir $LOG_DIR

# testing
bash ./tools/scripts/dist_test.sh $NUM_GPU --config $CONFIG --ckpt $CKPT_PATH --eval bbox
```
