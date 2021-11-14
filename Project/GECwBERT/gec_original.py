import sys
import os
import math
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from keras.preprocessing.sequence import pad_sequences
from pytorch_pretrained_bert import BertForSequenceClassification
from difflib import SequenceMatcher
from hunspell import Hunspell
import spacy
from tqdm import tqdm
import numpy as np
import logging

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained(
    'bert-base-cased', do_lower_case=False)

# Load pre-trained model tokenizer (vocabulary)
tokenizerLarge = BertTokenizer.from_pretrained(
    'bert-large-cased', do_lower_case=False)

# gn_GB dictionary for hunspell
gb = Hunspell("en_GB-large", hunspell_data_dir=".")

# List of common determiners
det = ['the', 'a', 'an', 'this', 'that', 'these', 'those', 'my', 'your', 'his',
       'her', 'its', 'our', 'their', 'all', 'both', 'half', 'either', 'neither',
       'each', 'every', 'other', 'another', 'such', 'what', 'rather', 'quite']

# List of common prepositions
prep = ["about", "at", "by", "for", "from", "in", "of", "on", "to", "with",
        "into", "during", "including", "until", "against", "among",
        "throughout", "despite", "towards", "upon", "concerning"]

# List of helping verbs
helping_verbs = ['am', 'is', 'are', 'was', 'were', 'being', 'been', 'be',
                 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                 'shall', 'should', 'may', 'might', 'must', 'can', 'could']


def progress_bar(some_iter):
    try:
        return tqdm(some_iter)
    except ModuleNotFoundError:
        return some_iter


def check_GE(sents, modelGEDs):
    """Check of the input sentences have grammatical errors
    :param list: list of sentences
    :return: error, probabilities
    :rtype: (boolean, (float, float))
    """

    # Create sentence and label lists
    # We need to add special tokens at the beginning and end of each sentence
    # for BERT to work properly
    sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sents]
    labels = [0]

    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

    # Padding Sentences
    # Set the maximum sequence length. The longest sequence in our training set
    # is 47, but we'll leave room on the end anyway.
    # In the original paper, the authors used a length of 512.
    # MAX_LEN = max([len(text) for text in tokenized_texts])
    MAX_LEN = 128

    predictions = []
    true_labels = []

    # Pad our input tokens
    input_ids = pad_sequences(
        [tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
        maxlen=MAX_LEN, dtype="long", truncating="post", padding="post"
    )

    # Attention masks
    # Create attention masks
    attention_masks = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    prediction_inputs = torch.tensor(input_ids)
    prediction_masks = torch.tensor(attention_masks)
    prediction_labels = torch.tensor(labels)

    for modelGED in modelGEDs:
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            logits = modelGED(prediction_inputs, token_type_ids=None,
                              attention_mask=prediction_masks)

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        # label_ids = b_labels.to("cpu").numpy()

        # Store predictions and true labels
        predictions.append(logits)
        # true_labels.append(label_ids)
        # print('logits:\n{}\n'.format(logits))
        # print('predictions:\n{}\n'.format(predictions))

    prob_vals = np.mean([np.array(pred) for pred in predictions], axis=0)
    # print('prob_vals:\n{}\n'.format(prob_vals))
    flat_predictions = np.argmax(prob_vals, axis=1).flatten()
    # print('flat_predictions:\n{}\n'.format(flat_predictions))
    # flat_true_labels = [item for sublist in true_labels for item in sublist]

    return flat_predictions, prob_vals


def create_spelling_set(org_text, modelGEDs):
    """ Create a set of sentences which have possible corrected spellings
    """

    sent = org_text.lower().strip().split()

    nlp = spacy.load("en_core_web_sm")
    proc_sent = nlp.tokenizer.tokens_from_list(sent)
    nlp.tagger(proc_sent)

    sentences = []

    for tok in proc_sent:
        # check for spelling for alphanumeric
        if tok.text.isalpha() and not gb.spell(tok.text):
            new_sent = sent[:]
            # append new sentences with possible corrections
            for sugg in gb.suggest(tok.text):
                new_sent[tok.i] = sugg
                sentences.append(" ".join(new_sent))

    spelling_sentences = sentences

    # retain new sentences which have a
    # minimum chance of correctness using BERT GED
    new_sentences = []

    for sent in spelling_sentences:
        no_error, prob_val = check_GE([sent], modelGEDs)
        exps = [np.exp(i) for i in prob_val[0]]
        sum_of_exps = sum(exps)
        softmax = [j/sum_of_exps for j in exps]
        if(len(softmax) > 1 and softmax[1] > 0.6):
            new_sentences.append(sent)

    # if no corrections, append the original sentence
    if len(spelling_sentences) == 0:
        spelling_sentences.append(" ".join(sent))

    # eliminate dupllicates
    [spelling_sentences.append(sent) for sent in new_sentences]
    spelling_sentences = list(dict.fromkeys(spelling_sentences))

    return spelling_sentences


def create_grammar_set(spelling_sentences, modelGEDs):
    """ create a new set of sentences with deleted determiners, 
        prepositions & helping verbs
    """
    new_sentences = []

    for text in spelling_sentences:
        sent = text.strip().split()
        for i in range(len(sent)):
            new_sent = sent[:]

            if new_sent[i] not in list(set(det + prep + helping_verbs)):
                continue

            del new_sent[i]
            text = " ".join(new_sent)

            # retain new sentences which have a minimum chance of correctness using BERT GED
            no_error, prob_val = check_GE([text], modelGEDs)
            exps = [np.exp(i) for i in prob_val[0]]
            sum_of_exps = sum(exps)
            softmax = [j/sum_of_exps for j in exps]
            if(len(softmax) > 1 and softmax[1] > 0.6):
                new_sentences.append(text)

    # eliminate dupllicates
    [spelling_sentences.append(sent) for sent in new_sentences]
    spelling_sentences = list(dict.fromkeys(spelling_sentences))
    return spelling_sentences


def create_mask_set(spelling_sentences):
    """For each input sentence create 2 sentences
        (1) [MASK] each word
        (2) [MASK] for each space between words
    """
    sentences = []

    for sent in spelling_sentences:
        sent = sent.strip().split()
        for i in range(len(sent)):
            # (1) [MASK] each word
            new_sent = sent[:]
            new_sent[i] = '[MASK]'
            text = " ".join(new_sent)
            new_sent = '[CLS] ' + text + ' [SEP]'
            sentences.append(new_sent)

            # (2) [MASK] for each space between words
            new_sent = sent[:]
            new_sent.insert(i, '[MASK]')
            text = " ".join(new_sent)
            new_sent = '[CLS] ' + text + ' [SEP]'
            sentences.append(new_sent)

    return sentences


def check_grammar(org_sent, sentences, spelling_sentences, model, modelGEDs):
    """ check grammar for the input sentences
    """

    n = len(sentences)

    # what is the tokenized value of [MASK]. Usually 103
    text = '[MASK]'
    tokenized_text = tokenizerLarge.tokenize(text)
    mask_token = tokenizerLarge.convert_tokens_to_ids(tokenized_text)[0]

    LM_sentences = []
    new_sentences = []
    i = 0  # current sentence number
    l = len(org_sent.strip().split())*2  # l is no of sentencees
    mask = False  # flag indicating if we are processing space MASK

    for sent in sentences:
        i += 1

        print(".", end="")
        if i % 50 == 0:
            print("")

        # tokenize the text
        tokenized_text = tokenizerLarge.tokenize(sent)
        indexed_tokens = tokenizerLarge.convert_tokens_to_ids(tokenized_text)

        # Create the segments tensors.
        segments_ids = [0] * len(tokenized_text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        # Predict all tokens
        with torch.no_grad():
            predictions = model(tokens_tensor, segments_tensors)

        # index of the masked token
        mask_index = (tokens_tensor == mask_token).nonzero()[0][1].item()
        # predicted token
        predicted_index = torch.argmax(predictions[0, mask_index]).item()
        predicted_token = tokenizerLarge.convert_ids_to_tokens(
            [predicted_index])[0]

        text = sent.strip().split()
        mask_index = text.index('[MASK]')

        if not mask:
            # case of MASKed words
            mask = True
            text[mask_index] = predicted_token
            try:
                # retrieve original word
                org_word = spelling_sentences[i //
                                              l].strip().split()[mask_index-1]
            except:
                print("!", end="")
                continue
            # use SequenceMatcher to see if predicted word is similar to original word
            if SequenceMatcher(None, org_word, predicted_token).ratio() < 0.6:
                if org_word not in list(set(det + prep + helping_verbs)) or predicted_token not in list(set(det + prep + helping_verbs)):
                    continue
            if org_word == predicted_token:
                continue
        else:
            # case for MASKed spaces
            mask = False
            # only allow determiners / prepositions  / helping verbs in spaces
            if predicted_token in list(set(det + prep + helping_verbs)):
                text[mask_index] = predicted_token
            else:
                continue

        text.remove('[SEP]')
        text.remove('[CLS]')
        new_sent = " ".join(text)

        # retain new sentences which have a minimum chance of correctness using BERT GED
        no_error, prob_val = check_GE([new_sent], modelGEDs)
        exps = [np.exp(i) for i in prob_val[0]]
        sum_of_exps = sum(exps)
        softmax = [j/sum_of_exps for j in exps]
        if no_error and len(softmax) > 1 and softmax[1] > 0.996:
            print('{} : {}\n'.format(new_sent, softmax[1]))
            print("*", end="")
            new_sentences.append(new_sent)

    print("")

    # remove duplicate suggestions
    spelling_sentences = []
    [spelling_sentences.append(sent) for sent in new_sentences]
    spelling_sentences = list(dict.fromkeys(spelling_sentences))
    spelling_sentences

    return spelling_sentences


def predict(model_paths, data_path, start, end):
    # Check to confirm that GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    torch.cuda.get_device_name(0)

    modelGEDs = []

    for model_path in model_paths:
        # load previously trained BERT Grammar Error Detection model
        modelGED = BertForSequenceClassification.from_pretrained(
            "bert-base-cased", num_labels=2)
        # restore model
        modelGED.load_state_dict(torch.load(model_path))
        modelGED.eval()
        modelGEDs.append(modelGED)

    # Load pre-trained model (weights) for Masked Language Model (MLM)
    model = BertForMaskedLM.from_pretrained('bert-large-cased')
    model.eval()

    # preprocessing input sentences
    file = open(data_path).read().strip().split("\n\n")

    input_sentences = []
    for i in range(len(file)):
        input_tokens = file[i].split("\n")[0].split(" ")
        input_tokens = input_tokens[1:]  # remove S from input sentence
        input_sentence = ' '.join(input_tokens)
        input_sentences.append(input_sentence)

    input_sentences = input_sentences[start:end]

    print('Predicting for {} sentences from the input file'.format(
        len(input_sentences)))
    logging.info('Predicting for {} sentences from the input file'.format(
        len(input_sentences)))

    output_sentences = []

    for input_sentence in input_sentences:
        print('Input : {}'.format(input_sentence))
        logging.info('Input : {}'.format(input_sentence))
        spelling_sentences = create_spelling_set(
            input_sentence, modelGEDs)
        grammar_sentences = create_grammar_set(
            spelling_sentences, modelGEDs)
        mask_sentences = create_mask_set(
            grammar_sentences)

        candidate_sentences = check_grammar(
            input_sentence, mask_sentences, grammar_sentences, model, modelGEDs)

        print('Processing {} possibilities'.format(len(candidate_sentences)))
        logging.info('Processing {} possibilities'.format(
            len(candidate_sentences)))

        if len(candidate_sentences) == 0:  # no candidate sentences (> 0.996)
            output_sentence = input_sentence
            output_sentences.append(output_sentence)
            print('Output : (no change)\n')
            logging.info('Output : (no change)\n')
            continue

        no_error, prob_val = check_GE(candidate_sentences, modelGEDs)

        max = 0
        max_idx = 0

        for i in range(len(prob_val)):
            exps = [np.exp(i) for i in prob_val[i]]
            sum_of_exps = sum(exps)
            softmax = [j/sum_of_exps for j in exps]
            if len(softmax) > 1 and softmax[1] > max:
                max = softmax[1]
                max_idx = i

        # output the sentence with the highest probability
        output_sentence = candidate_sentences[max_idx]
        output_sentences.append(output_sentence)
        print('Output : {}\n'.format(output_sentence))
        logging.info('Output : {}\n'.format(output_sentence))

    no_of_models = len(modelGEDs)

    # create two parallel files for input and output sentences
    with open("input_{}mod_{}_{}.txt".format(no_of_models, start, end), "x") as f:
        f.write("\n".join(input_sentences))

    with open("output_{}mod_{}_{}.txt".format(no_of_models, start, end), "w") as f:
        f.write("\n".join(output_sentences))


def main():
    model_paths = sys.argv[1:-3]
    no_of_models = len(model_paths)
    data_path = sys.argv[-3]
    start = int(sys.argv[-2])
    end = int(sys.argv[-1])
    logging.basicConfig(level=logging.INFO, filename='original_{}_{}.log'.format(
        no_of_models, start, end))
    print('Detected {} GED models\n'.format(no_of_models))
    logging.info('Detected {} GED models\n'.format(no_of_models))
    predict(model_paths, data_path, start, end)
    print('Successfully finished prediction\n')
    logging.info('Successfully finished prediction\n')


if __name__ == "__main__":
    main()
