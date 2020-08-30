DATASET_DIR="train_data"

python download.py --mode cocostuff --dataset_dir $DATASET_DIR
echo "Finish download."
unzip "$DATASET_DIR/cocostuff/train.zip" -d "$DATASET_DIR"