topic=set14
dataset=data/natural/${topic}
dict=experiment/$topic
val_hr=data/set14
val_lr=experiment/$topic/val_lr
result=experiment/$topic/result
result_gt=experiment/$topic/result_gt

mkdir -p $val_lr $val_hr $result $result_gt
#cp data/set14/pepper.png $val_hr/

dict_prefix=${dict}/${topic}_
python3 src/rescale.py --val_hr $val_hr --val_lr $val_lr
#python3 src/dict_train.py --dataset $dataset --dict_prefix $dict_prefix
#python3 src/run.py --val_hr $val_hr --val_lr $val_lr --dict_prefix $dict_prefix --result $result
python3 src/run.py --val_hr $val_hr --val_lr $val_lr --dict_prefix ground_truth/ --result $result_gt
