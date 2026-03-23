echo "=== DEQ + DistiLLM ==="
bash ./distillm-fdd/scripts/gpt2/deq/train_0.1B_1.5B.sh
bash ./scripts/eval_deq_gpt2_0.1B.sh
echo "done DEQ + DistiLLM"

echo "=== DEQ + DistiLLM-2 ==="
bash distillm-2-master/scripts/gpt2/deq_distillm_2_gpt2_0.1b.sh > distillm-2-master/outputs/gpt2-deq-distillm2-train.log 2>&1
echo "done DEQ + DistiLLM-2"
