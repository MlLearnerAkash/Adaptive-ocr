#!/bin/bash
#SBATCH -A research
#SBATCH --nodelist=gnode084
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=1024
#SBATCH --time=96:00:00
#SBATCH --mail-type=END
source activate indicocr
nvidia-smi
cd /ssd_scratch/cvit
mkdir ajoy
cd ajoy
mkdir printed
cd printed
mkdir english
cd english

############# original training  ######################
cd ~
cd /ssd_scratch/cvit/ajoy/printed/english/code/
rm -rf test_accuracy.txt
rm -rf test_gt_and_predicted_text.txt

cd ~
cd /ssd_scratch/cvit/ajoy/printed/english/code/

python3 lang_train.py --mode train \
--lang english --trainRoot /ssd_scratch/cvit/ajoy/printed/english/dataset/train_lmdb \
--workers 5 \
--pretrained /ssd_scratch/cvit/ajoy/printed/english/code/out/crnn_results/best_cer.pth \
--valRoot /ssd_scratch/cvit/ajoy/printed/english/dataset/val_lmdb \
--batchSize 64 --nepoch 40 --cuda --out out  \
--store_probability no \
--displayInterval 10000 --valInterval 10000  --adadelta

cd ~
cp -r /ssd_scratch/cvit/ajoy/printed/english/code/out/crnn_results  0_new_experiments/1_printed/V10_Experiments/trained_model/english/out/
cp -r /ssd_scratch/cvit/ajoy/printed/english/code/out/crnn_resultslog.txt  0_new_experiments/1_printed/V10_Experiments/trained_model/english/out/


# python3 lang_train.py --mode train --lang english --trainRoot /home/akash/ws/limited_supervision_ocr/synthetic_data_gen/datasets/train/train_lucida_0.002_lmdb/ --workers 5  --valRoot /home/akash/ws/limited_supervision_ocr/synthetic_data_gen/datasets/val/val_lucida_calligraphy_lmdb/ --batchSize 64 --nepoch 100  --displayInterval 10 --valInterval 10  --adadelta --lr 0.001 --pretrained /home/akash/ws/limited_supervision_ocr/V10_English_Model_Code_Script_charlist/code/out_font1/best_cer.pth --out /home/akash/ws/limited_supervision_ocr/V10_English_Model_Code_Script_charlist/code/font2_train_font1 --node_dir ./exp2/font2_trainSize666_lr0.001_pretrainedFont1 --cuda --use_tb