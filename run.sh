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


FILES=$(find configs -name '*.yml')

CURRENT_EXPERIMENT=1
ALL_EXPERIMENTS=$(ls -1 configs/*.yml | wc -l | awk '{$1=$1};1')

for filename in $FILES; do
    NAME=$(basename "$filename" .yml)
    echo "Running experiment $NAME ($CURRENT_EXPERIMENT/$ALL_EXPERIMENTS)."

    eval $(parse_yaml $filename "config_")

    mkdir -p experiments/$NAME
    cd experiments/$NAME

    rasa train nlu --nlu "../../$config_data_train_file" --config "../../$filename" &> "train.log"
    rasa test nlu --nlu "../../$config_data_test_file" --config "../../$filename" &> "test.log"

    python ../../evaluation_scripts/evaluation_nlu_evaluation_data.py -i results/diet-paper-eval.json
    python ../../evaluation_scripts/evaluation_atis_snips.py -i results/diet-paper-eval.json

    cd ../..
    cp $filename experiments/$NAME/

    CURRENT_EXPERIMENT=$((CURRENT_EXPERIMENT + 1))
done