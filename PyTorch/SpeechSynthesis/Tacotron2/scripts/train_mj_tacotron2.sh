#mkdir -p output
pretrained=/mnt/data/pretrained/tacotron2/tacotron2_1032590_6000_amp
outputroot=/mnt/data/checkpoints
trainfiles=/mnt/data/datasets/MJ/filelists/mj_mel_text_train_filelist.txt
valfiles=/mnt/data/datasets/MJ/filelists/mj_mel_text_val_filelist.txt
#weight_file=./output/checkpoint_Tacotron2_6150
#TODO: pretrained models having epoch 6001; so adjusting epoch from 6001 to 6200
#python -m multiproc train.py -m Tacotron2 -o output/ --checkpoint-path $pretrained --amp-run -lr 1e-3 --epochs 6200 -bs 38 --weight-decay 1e-6 --grad-clip-thresh 1.0 --cudnn-enabled --load-mel-from-disk --training-files=/mnt/data/datasets/MJ/filelists/mj_mel_text_train_filelist.txt --validation-files=/mnt/data/datasets/MJ/filelists/mj_mel_text_val_filelist.txt --log-file nvlog_mj_tacotron.json --anneal-steps 500 1000 1500 --anneal-factor 0.3 --use-saved-learning-rate true --epochs-per-checkpoint 5 -o ./output_mj_tacotron_A

# run_B
# second try set 
# weight-decay = 0 (as we don't want to destroy weight any more?)
# aneal to 10% when the epohcs hist 6050 where it seems to start diverge
# when the lr is kept @ 3e-05
#python -m multiproc train.py -m Tacotron2 -o output_mj_tactron2_run_B/ --checkpoint-path $pretrained --amp-run -lr 1e-4 --epochs 8000 -bs 38 --weight-decay 1e-8 --grad-clip-thresh 1.0 --cudnn-enabled --load-mel-from-disk --training-files=/mnt/data/datasets/MJ/filelists/mj_mel_text_train_filelist.txt --validation-files=/mnt/data/datasets/MJ/filelists/mj_mel_text_val_filelist.txt --log-file nvlog_mj_tacotron2.json --anneal-steps 6050 6300 6500 --anneal-factor 0.1 --use-saved-learning-rate true

outputdir=$outputroot/output_mj_tacotron2_run_fr_em
checkpointpath=$pretrained
mkdir -p $outputdir
python -m multiproc train.py -m Tacotron2 --checkpoint-path $checkpointpath \
 --learning-rate 3e-5 --epochs 9001 --batch-size 38 --weight-decay 1e-8 --grad-clip-thresh 1.0 \
 --load-mel-from-disk --training-files=$trainfiles --validation-files=$valfiles \
 --log-file nvlog.json --anneal-steps 6500 7000 8000 --anneal-factor 0.3 \
 --use-saved-learning-rate true \
 --output $outputdir \
 --amp-run --cudnn-enabled
