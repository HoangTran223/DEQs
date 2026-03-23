bash ./distillm-fdd/scripts/gpt2/distillm/train_0.1B_1.5B_on.sh ./distillm-fdd
bash ./scripts/eval_gpt2_0.1B.sh
bash ./distillm-fdd/scripts/gpt2/spandistillm/train_0.1B_1.5B.sh ./distillm-fdd
bash ./scripts/eval_span_gpt2_0.1B.sh

bash ./distillm-fdd/scripts/gpt2/fdd/train_0.1B_1.5B.sh ./distillm-fdd
bash ./scripts/eval_fdd_gpt2_0.1B.sh 
bash ./distillm-fdd/scripts/gpt2/spanfdd/train_0.1B_1.5B.sh ./distillm-fdd
bash ./scripts/eval_spanfdd_gpt2_0.1B.sh

