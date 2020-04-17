#!/usr/bin/env bash

parse_yaml() {
   local prefix=$2
   local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
   sed -ne "s|^\($s\)\($w\)$s:$s\"\(.*\)\"$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p"  $1 |
   awk -F$fs '{
      indent = length($1)/2;
      vname[indent] = $2;
      for (i in vname) {if (i > indent) {delete vname[i]}}
      if (length($3) > 0) {
         vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
         printf("%s%s%s=\"%s\"\n", "'$prefix'",vn, $2, $3);
      }
   }'
}

TABLE_FOLDER=$1

if [ -z "$TABLE_FOLDER" ]
then
    echo "Usage:"
    echo "./run.sh [table-1|table-2|table-3|table-4|table-5]"
    exit
fi

CONFIGS="configs/$TABLE_FOLDER"

FILES=$(find $CONFIGS -name '*.yml')

CURRENT_EXPERIMENT=1
ALL_EXPERIMENTS=$(ls -1 $CONFIGS/*.yml | wc -l | awk '{$1=$1};1')

for filename in $FILES; do
    NAME=$(basename "$filename" .yml)
    echo "----------------------------------------------------------------"
    echo "Running experiment $NAME ($CURRENT_EXPERIMENT/$ALL_EXPERIMENTS)."

    eval $(parse_yaml $filename "config_")

    mkdir -p experiments/$TABLE_FOLDER/$NAME
    cd experiments/$TABLE_FOLDER/$NAME

    rasa train nlu --nlu "../../../$config_data_train_file" --config "../../../$filename" &> "train.log"
    rasa test nlu --nlu "../../../$config_data_test_file" --config "../../../$filename" &> "test.log"

    python ../../../evaluation_scripts/evaluation_nlu_evaluation_data.py -i results/diet-paper-eval.json
    python ../../../evaluation_scripts/evaluation_atis_snips.py -i results/diet-paper-eval.json

    cd ../../..
    cp $filename experiments/$TABLE_FOLDER/$NAME/

    CURRENT_EXPERIMENT=$((CURRENT_EXPERIMENT + 1))
done

python evaluation_scripts/evaluation_nlu_evaluation_data.py -f "experiments/$TABLE_FOLDER/config-NLU-Evaluation-Data-Fold-{}"