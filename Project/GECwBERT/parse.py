def parse(data_path):
    # preprocessing input sentences
    file = open(data_path).read().strip().split("\n\n")

    input_sentences = []
    for i in range(len(file)):
        input_tokens = file[i].split("\n")[0].split(" ")
        input_tokens = input_tokens[1:]  # remove S from input sentence
        input_sentence = ' '.join(input_tokens)
        input_sentences.append(input_sentence)

    print(len(input_sentences))

    # create two parallel files for input and output sentences
    with open("input.txt", "x") as f:
        f.write("\n".join(input_sentences))


if __name__ == "__main__":
    parse('./data/official-2014.combined.m2')
