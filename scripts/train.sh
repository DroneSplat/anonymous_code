#! /bin/bash

GPU_ID=3
DATA_ROOT_DIR="data"
DATASETS=(
    dronesplat
    )

SCENES=(
    Cultural_Center
    TangTian
    Pavilion
    Simingshan
    Intersection
    Sculpture
    )

gs_train_iter=7000

for DATASET in "${DATASETS[@]}"; do
    for SCENE in "${SCENES[@]}"; do

        SOURCE_PATH=${DATA_ROOT_DIR}/${DATASET}/${SCENE}
        MODEL_PATH=./output/${DATASET}/${SCENE}
        COLMAP_PATH=${DATA_ROOT_DIR}/${DATASET}/${SCENE}/colmap

        CMD_P="CUDA_VISIBLE_DEVICES=${GPU_ID} python preprocess.py \
        --img_base_path ${SOURCE_PATH} \
        --preset_colmap_path ${COLMAP_PATH} \
        --preset_pose
        "

        CMD_T="CUDA_VISIBLE_DEVICES=${GPU_ID} proxychains python -W ignore ./train_joint_v8.py \
        -s ${SOURCE_PATH} \
        -m ${MODEL_PATH}  \
        --scene ${SCENE} \
        --iter ${gs_train_iter} \
        --use_masks \
        --video_segment \
        --use_hooks
        "

        CMD_RI="CUDA_VISIBLE_DEVICES=${GPU_ID} python render_interp.py \
        -s ${SOURCE_PATH} \
        -m ${MODEL_PATH}  \
        --iter ${gs_train_iter} \
        --eval \
        --get_video \
        "

        echo "========= ${SCENE}: Preprocess images ========="
        eval $CMD_P
        echo "========= ${SCENE}: Train: jointly optimize pose ========="
        eval $CMD_T
        echo "========= ${SCENE}: Render interpolated pose & output video ========="
        eval $CMD_R
        done
    done