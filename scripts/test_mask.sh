INPUT_DIR=example
OUTPUT_DIR=results

python inference_bbox.py --test_img_dir $INPUT_DIR
python test_fusion.py --name test_fusion --sample_p 1.0 --model fusion --fineSize 256 --test_img_dir $INPUT_DIR --results_img_dir $OUTPUT_DIR