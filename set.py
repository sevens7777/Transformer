import os.path


def split_dataset(file_path, output_path):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    datas = []
    with open(file_path, "r", encoding='utf-8') as f:
        for line in f:
            if not line or line == "":
                continue
            datas.append(line)
    train = datas[0:20000]
    val = datas[20000:22000]
    with open(os.path.join(output_path, "train.json"), "w", encoding="utf-8") as w:
        for line in train:
            w.write(line)
            w.flush()

    with open(os.path.join(output_path, "val.json"), "w", encoding="utf-8") as w:
        for line in val:
            w.write(line)
            w.flush()
    print("train count: ", len(train))
    print("val count: ", len(val))


if __name__ == '__main__':
    file_path = "data/train.jsonl"
    split_dataset(file_path=file_path, output_path="data")
