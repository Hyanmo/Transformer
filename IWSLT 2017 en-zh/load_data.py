from datasets import load_dataset

def main():
    print("Loading IWSLT 2017 en-zh...")

    dataset = load_dataset("iwslt2017", "iwslt2017-en-zh")

    train_data = dataset["train"]
    valid_data = dataset["validation"]
    test_data = dataset["test"]

    print("Train size:", len(train_data))
    print("Valid size:", len(valid_data))
    print("Test size:", len(test_data))

    print("\nSample example:")
    print(train_data[0])

if __name__ == "__main__":
    main()