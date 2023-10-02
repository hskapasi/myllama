import fire
from llama import Llama
from typing import List
from transformers import AutoTokenizer
import time
import concurrent.futures

tokenizer = AutoTokenizer.from_pretrained("upstage/Llama-2-70b-instruct")

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 50,
    max_gen_len: int = 100,
    max_batch_size: int = 8,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    prompts: List[str] = ["I believe the meaning of life is"]
    
    tokens_per_second_list = []

    for _ in range(100):
        # Record the start time
        start_time = time.time()
        
        results = generator.text_completion(
            prompts,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        
        # Record the end time
        end_time = time.time()
        
        # Calculate the tokens per second
        max_length = len(tokenizer.encode(results[0]['generation']))
        elapsed_time = end_time - start_time
        tokens_per_sec = max_length / elapsed_time
        
        # Store the tokens per second
        tokens_per_second_list.append(tokens_per_sec)
        
        print(f"Tokens per second: {tokens_per_sec}")

    def measure_tokens_per_second(prompt):
        start_time = time.time()
        result = generator.text_completion(
            [prompt],
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )[0]
        end_time = time.time()
        max_length = len(tokenizer.encode(result['generation']))
        elapsed_time = end_time - start_time
        return max_length / elapsed_time
    
    prompt = "I believe the meaning of life is"
    avg_tokens_per_second_list = []

    # Simulate for 1 to 10 users
    for num_users in range(1, 11):
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_users) as executor:
            tokens_per_second_values = list(executor.map(measure_tokens_per_second, [prompt] * num_users))
            avg_tokens_per_second = sum(tokens_per_second_values) / num_users
            avg_tokens_per_second_list.append(avg_tokens_per_second)
            print(f"Average tokens per second for {num_users} users: {avg_tokens_per_second}")

if __name__ == "__main__":
    fire.Fire(main)
