echo "Downloading..."
python download.py
python download.py --mode coco-weights
echo "Finish download."
unzip checkpoints.zip
