from datasets import load_dataset
import sentencepiece as spm

def main():
    print("Loading dataset...")
    dataset = load_dataset("iwslt2017", "iwslt2017-en-zh")
    train_data = dataset["train"]

    print("Writing training text...")

    with open("spm_train.txt", "w", encoding="utf-8") as f:
        #5万句训练tokenizer
        for i, sample in enumerate(train_data):
            if i >= 50000:
                break
            en = sample["translation"]["en"]
            zh = sample["translation"]["zh"]

            f.write(en.strip() + "\n")
            f.write(zh.strip() + "\n")

    print("Training SentencePiece...")

    spm.SentencePieceTrainer.train(
        input="spm_train.txt",
        model_prefix="spm",
        vocab_size=8000,
        character_coverage=1.0,
        model_type="bpe",
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3
    )

    print("Done.")

if __name__ == "__main__":
    main()