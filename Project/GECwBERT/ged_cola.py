import sys
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
from tqdm import tqdm, trange
import pandas as pd
import numpy as np


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat)/len(labels_flat)


def train(data_path, out_path, num_epochs):
    # Check to confirm the specific GPU model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    # Move Data to a Panda DataFrame
    df = pd.read_csv(data_path, delimiter='\t', header=None,
                     names=['sentence_source', 'label', 'label_notes', 'sentence'])

    # Sentence and Label Lists
    sentences = df.sentence.values
    sentences = ["[CLS]" + sentence + " [SEP]" for sentence in sentences]
    labels = df.label.values

    # Tokenize Inputs
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-cased', do_lower_case=False)
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

    # Padding Sentences
    # Set the maximum sequence length to the longest sequence in the training set
    MAX_LEN = max([len(text) for text in tokenized_texts])

    # Pad our input tokens
    input_ids = pad_sequences(
        [tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
        maxlen=MAX_LEN, dtype="long", truncating="post", padding="post"
    )

    # Attention masks
    # Create attention masks of 1s for each token followed by 0s for padding
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    # Train and Validation Set
    RANDOM_STATE = 4248
    TEST_SIZE = 0.1
    train_inputs, validation_inputs, train_labels, validation_labels = \
        train_test_split(input_ids, labels,
                         random_state=RANDOM_STATE, test_size=TEST_SIZE)
    train_masks, validation_masks, _, _ = \
        train_test_split(attention_masks, input_ids,
                         random_state=RANDOM_STATE, test_size=TEST_SIZE)

    # transform all data into torch tensors
    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)
    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)
    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)

    # Preparation for Training
    # Select a batch size for training. For fine tuning BERT on a
    # specific task , BERT authors recommend a batch size of 16 or 32
    BATCH_SIZE = 32

    # Create an iterator of our data with torch DataLoader
    # This helps save on memory during training because, unlike a for loop,
    # with iterator the entire dataset does not need to be loaded into memory
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)

    train_dataloader = \
        DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)
    validation_data = \
        TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = \
        DataLoader(validation_data, sampler=validation_sampler,
                   batch_size=BATCH_SIZE)

    # Load BertForSequenceClassification, the pretrained BERT model
    # with a single linear classification layer on top
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-cased", num_labels=2)
    model.cuda()

    # Hyperparameters
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']

    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.01
        },
        {
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.00
        }
    ]

    optimizer = BertAdam(optimizer_grouped_parameters, lr=2e-5, warmup=0.1)

    # Training Loop

    # Store our loss and accuracy for plotting
    train_loss_set = []

    # Number of training epochs (authors recommend between 2 and 4)
    # trange is a tqdm wrapper around the normal python range
    for _ in trange(num_epochs, desc="Epoch"):
        # Training
        # Set our model to training mode (as opposed to evaluation mode)
        model.train()
        # Tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # Forward pass
            loss = model(b_input_ids, token_type_ids=None,
                         attention_mask=b_input_mask, labels=b_labels)

            train_loss_set.append(loss.item())
            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()
            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
        print("Train loss: {}".format(tr_loss/nb_tr_steps))

    # Validation
    # Put model in evaluation mode to evaluate loss on the validation set
    model.eval()

    # Tracking variables
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients,
        # ve memory and speede up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            logits = model(b_input_ids, token_type_ids=None,
                           attention_mask=b_input_mask)

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

    print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))

    torch.save(model.state_dict(), out_path)

    print(f"Model saved at {out_path}")


def main():
    data_path = sys.argv[1]
    out_path = sys.argv[2]
    num_epochs = int(sys.argv[3])
    train(data_path, out_path, num_epochs)


if __name__ == "__main__":
    main()
