python generate_erased_pairs.py \
  --coco "outputs/inpaint_remove_train_lama/images/_annotations.coco.json" \
  --images-dir "outputs/inpaint_remove_train_lama/images" \
  --out data/training-data \
  --category detail \
  --method lama \
  --device cuda \
  --workers 1 \
  --batch-size 16 \
  --pair-mode pair
