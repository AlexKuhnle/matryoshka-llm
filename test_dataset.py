import datasets
import sys
import tokenizers
import torch
from tqdm.auto import tqdm


if __name__ == "__main__":
    dataset_name = sys.argv[1]
    if dataset_name == "tinystories":
        dataset = "roneneldan/TinyStories"
    elif dataset_name == "tinyshakespeare":
        dataset = "tiny_shakespeare"
    else:
        raise NotImplementedError

    dataset = datasets.load_dataset(dataset)
    dataset = dataset["train"]

    tokenizer = tokenizers.Tokenizer.from_file(f"data/tokenizers/{dataset_name}-bpe")
    eos_token = tokenizer.token_to_id("</s>")
    pad_token = tokenizer.get_vocab_size()

    context_length = 1024
    num_beyond_context_length = 0
    for x in tqdm(dataset):
        if len(tokenizer.encode(x["text"]).ids) > context_length:
            num_beyond_context_length += 1
    print(f"Beyond context length: {num_beyond_context_length} of {len(dataset)}")
    # Beyond context length: 1226 of 2119719
