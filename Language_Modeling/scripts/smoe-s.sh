LEARNING_RATE="1e-3"
GROUP_SPARSE_LOSS_WEIGHT="1e-5"
SIGMA_INIT="10.0"
SIGMA_END="1.0"
SIGMA_GAMMA="0.2"
OUTPUT_PATH="./output/SMoGE_S-lr[$LEARNING_RATE]-gslw[$GROUP_SPARSE_LOSS_WEIGHT]-sigma[$SIGMA_INIT-$SIGMA_END-$SIGMA_GAMMA]"

mkdir -p $OUTPUT_PATH

args="
--data /path/to/data/directory/wikitext-103/ \
--base_arch transformer \
--architecture sgsgsg \
--gate_name smoe \
--nlayers 3 \
--hid-sz 128 \
--inner-hid-sz 128 \
--nheads 8 \
--block-sz 256 \
--attn-span 256 \
--dropout 0.7 \
--load_balance 0.01 \
--optim adam \
--lr-warmup 3000 \
--niter 60 \
--batch-sz 96 \
--batch-split 2 \
--nbatches 1000 \
--distributed \
--checkpoint $OUTPUT_PATH/smoe.pt \

--lr $LEARNING_RATE \
--group_sparse_loss_weight $GROUP_SPARSE_LOSS_WEIGHT \
--sigma_init $SIGMA_INIT \
--sigma_end $SIGMA_END \
--sigma_gamma $SIGMA_GAMMA \
"

echo "Training ..."
CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch --master_port 10013 --nproc_per_node=4 --use_env train.py $args

echo "Evaluation ..."
CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch --master_port 10013 --nproc_per_node=4 --use_env train.py $args --resume --full-eval-mode