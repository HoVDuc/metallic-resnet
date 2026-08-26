python inference.py /home/anlab/hovduc/jigboard_checker/outputs/metal/pair_A/pair_A_masked_1.jpg /home/anlab/hovduc/jigboard_checker/outputs/metal/pair_A/pair_A_masked_2.jpg \
    --checkpoint /home/anlab/Desktop/checkpoints/epoch_0050.pt \
    --out outputs/inference \
    --head cosine \
    --device cuda
