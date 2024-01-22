import datasets
import tokenizers


if __name__ == "__main__":
    DEBUG = False
    dataset_name = "tinystories"
    vocab_size = 1024

    if dataset_name == "tinystories":
        dataset = "roneneldan/TinyStories"
    else:
        raise NotImplementedError

    dataset = datasets.load_dataset(dataset)
    train_dataset = dataset["train"]
    if DEBUG:
        train_dataset = train_dataset.select(list(range(10000)))

    tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())
    tokenizer.normalizer = tokenizers.normalizers.Sequence([
        tokenizers.normalizers.NFD(),
        tokenizers.normalizers.Lowercase(),
        tokenizers.normalizers.StripAccents(),
    ])
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Sequence([
        tokenizers.pre_tokenizers.Whitespace(),
        tokenizers.pre_tokenizers.Punctuation(behavior="isolated"),
        tokenizers.pre_tokenizers.Digits(individual_digits=True),
    ])
    tokenizer.post_processor = tokenizers.processors.TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> <s> $B </s>",
        special_tokens=[("<s>", 0), ("</s>", 1)],
    )
    tokenizer.decoder = tokenizers.decoders.BPEDecoder(suffix="</w>")
    trainer = tokenizers.trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<s>", "</s>"],
        continuing_subword_prefix="<w>",
        end_of_word_suffix="</w>",
    )
    tokenizer.train_from_iterator((x["text"] for x in train_dataset), trainer=trainer)

    if not DEBUG:
        tokenizer_path = f"data/tokenizers/{dataset_name}-bpe"
        tokenizer.save(tokenizer_path)

    # Example usage:
    # text = "abc"
    # ids = self.tokenizer.encode(text).ids
