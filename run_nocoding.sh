export VLLM_WORKER_MULTIPROC_METHOD=spawn
CUDA_VISIBLE_DEVICES=0 olmes --model ./model_clean \
    --task gsm8k::tulu drop::llama3 ifeval::tulu popqa::tulu mmlu:mc::tulu bbh:cot-v1::tulu truthfulqa::tulu \
    --output-dir workspace \
    --model-type vllm \
    --model-args '{"trust_remote_code":"true", "max_length":4096}' --batch-size 96