MI_MODEL_ID=${1:-"Experimento"} 
MI_DES=${2:-"Sin_Descripcion"}
CABECERA=${3:-"flatten"}
CMS_LR=${4:-"0.0001"}       
POLICY=${5:-"spc"}
USE_MID_CMS=${6:-0}         
MID_POSITION=${7:-0}            

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=104
model_name=PatchTST

root_path_name=./dataset/
data_path_name=national_illness.csv
model_id_name=national_illness
data_name=custom

random_seed=2021
for pred_len in 24 36 48 60
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id ${MI_MODEL_ID}_$seq_len'_'$pred_len \
      --des "$MI_DES" \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 3 \
      --n_heads 4 \
      --d_model 16 \
      --d_ff 128 \
      --dropout 0.3 \
      --fc_dropout 0.3 \
      --head_dropout 0 \
      --head_type ${CABECERA} \
      --cms_lr "$CMS_LR" \
      --update_policy "$POLICY" \
      --use_mid_cms "$USE_MID_CMS" \
      --mid_position "$MID_POSITION" \
      --patch_len 24 \
      --stride 2 \
      --train_epochs 100 \
      --lradj 'constant'\
      --use_gpu 1 \
      --num_workers 0 \
      --itr 1 --batch_size 16 --learning_rate 0.0025
done