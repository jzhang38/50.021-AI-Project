python demo/restoration_video_demo.py \
    ./configs/restorers/basicvsr/basicvsr_reds4.py \
    https://download.openmmlab.com/mmediting/restorers/basicvsr/basicvsr_reds4_20120409-0e599677.pth \
    ../../../../../data/peiyuann/REDS/train_sharp_bicubic/X4/000 \
    ./output


CUDA_VISIBLE_DEVICES=1 ./tools/dist_train.sh ./configs/restorers/basicvsr/basicvsr_reds4_custom.py 1

CUDA_VISIBLE_DEVICES=3 ./tools/dist_train_load.sh ./configs/restorers/basicvsr/basicvsr_reds4_custom_load.py 1 

CUDA_VISIBLE_DEVICES=2 ./tools/dist_train_load2.sh ./configs/restorers/basicvsr/basicvsr_reds4_custom_load.py 1 


CUDA_VISIBLE_DEVICES=1 ./tools/dist_train_load.sh ./configs/restorers/basicvsr/basicvsr_reds4_custom_load_lr_small.py 1 


CUDA_VISIBLE_DEVICES=0 ./tools/dist_test.sh ./configs/restorers/basicvsr/basicvsr_reds4_custom.py work_dirs/basicvsr_reds4_custom/iter_10000.pth 1 

CUDA_VISIBLE_DEVICES=1 ./tools/dist_test.sh ./configs/restorers/basicvsr/basicvsr_reds4_custom_load.py work_dirs/basicvsr_reds4_custom_load/iter_10000.pth 1 



CUDA_VISIBLE_DEVICES=0  ./tools/dist_test.sh ./configs/restorers/basicvsr/basicvsr_reds4_custom_load.py work_dirs/basicvsr_reds4_custom_load/iter_10000.pth 1  --save-path /home/peiyuan_zhang/Peiyuan/AIProj/custom_load

CUDA_VISIBLE_DEVICES=1  ./tools/dist_test.sh ./configs/restorers/basicvsr/basicvsr_reds4_custom.py work_dirs/basicvsr_reds4_custom/iter_10000.pth 1    --save-path /home/peiyuan_zhang/Peiyuan/AIProj/custom



CUDA_VISIBLE_DEVICES=0  ./tools/dist_test.sh ./configs/restorers/basicvsr/basicvsr_reds4_custom_denoise.py work_dirs/basicvsr_reds4_custom_denoise/iter_10000.pth 1 --save-path /home/peiyuan_zhang/Peiyuan/AIProj/custom_denoise