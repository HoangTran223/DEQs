echo "start"

bash distillm-2-master/scripts/gpt2/distillm_2_gpt2_0.1b.sh > distillm-2-master/outputs/gpt2-0.1b-distillm2-train.log 2>&1
echo "done gpt2 distillm"


bash distillm-2-master/scripts/gpt2/span_distillm_2_gpt2_0.1b.sh > distillm-2-master/outputs/span-gpt2-0.1b-distillm2-train.log 2>&1
echo "done gpt2 span distillm"
bash ./scripts/distillm2/eval_span_gpt2_0.1B.sh

echo "done"