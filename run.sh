olmes --model Qwen/Qwen3-4B-Base \
    --task gsm8k::tulu drop::llama3 minerva_math::tulu codex_humaneval::tulu codex_humanevalplus::tulu ifeval::tulu popqa::tulu mmlu:mc::tulu bbh:cot-v1::tulu truthfulqa::tulu \
    --output-dir workspace \
    --model-type vllm \
    --model-args '{"trust_remote_code":"true", "max_length":4096}' --batch-size 96