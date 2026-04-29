LEARNING_RATE="1e-3"
GROUP_SPARSE_LOSS_WEIGHT="1e-5"
SIGMA_INIT="10.0"
SIGMA_END="1.0"
SIGMA_GAMMA="0.2"
OUTPUT_PATH="./output/MomentumSMoGE_M-lr[$LEARNING_RATE]-gslw[$GROUP_SPARSE_LOSS_WEIGHT]-sigma[$SIGMA_INIT-$SIGMA_END-$SIGMA_GAMMA]"

mkdir -p $OUTPUT_PATH

args="
--data /path/to/data/directory/wikitext-103/ \
--base_arch transformer \
--architecture smsmsmsmsmsm \
--gate_name smoe \
--nlayers 6 \
--hid-sz 352 \
--inner-hid-sz 352 \
--nheads 8 \
--block-sz 512 \
--attn-span 1024 \
--dropout 0.1 \
--load_balance 0.01 \
--optim adam \
--lr-warmup 4000 \
--niter 80 \
--batch-sz 48 \
--batch-split 2 \
--nbatches 1000 \
--distributed \
--gamma1 1.0 \
--gamma2 1.0 \
--mu 0.7 \
--beta1 0.9 \
--beta2 0.999 \
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