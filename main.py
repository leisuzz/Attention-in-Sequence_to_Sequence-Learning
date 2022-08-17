import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import spacy
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
from torchtext.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator
from torchtext.data.utils import get_tokenizer


spacy_ger = spacy.load("de_core_web_sm")
spacy_eng = spacy.load("en_core_web_sm")

german = get_tokenizer(spacy_ger)
english = get_tokenizer(spacy_eng)

german = Field(
    tokenize=german, lower=True, init_token="<sos>", eos_token="<eos>"
)
english = Field(
    tokenize=english, lower=True, init_token="<sos>", eos_token="<eos>"
)

train_data, valid_data, test_data = Multi30k.splits(
    exts=(".de", ".en"), fields=(german, english)
)

german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)

class Encoder(nn.Module):
    def __int__(self,
                input_size,
                embedding_size,
                hidden_size,
                num_layers,
                dropout):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, bidirectional=True)

        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        # x= (seq_len, N(Batch))
        embedding = self.dropout(self.embedding(x))    # embedding= (seq_len, N, embedding_size)

        encoder_states, (hidden, cell) = self.rnn(embedding)        # (seq_len, N, hidden_size)

        hidden = self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
        cell = self.fc_cell(torch.cat((cell[0:1], cell[1:2]), dim=2))       # keep dim by slicing [idx:idx+1]

        return encoder_states, hidden, cell


class Decoder(nn.Module):
    def __int__(self,
                input_size,
                embedding_size,
                hidden_size,
                output_size,
                num_layers,
                dropout):
        super(Decoder, self).__int__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(hidden_size * 2 + embedding_size, hidden_size, num_layers)

        self.energy = nn.Linear(hidden_size * 3, 1)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.Relu()

    def forward(self,
                x,
                encoder_states,
                hidden,
                cell):
        x = x.unsqueeze(0)      # x = (1,N)
        embedding = self.dropout(self.embedding(x))     # (1, N, embedding_size)

        sequence_length = encoder_states.shape[0]
        h_reshaped = hidden.repeat(sequence_length, 1, 1)       # (seq_len, N, hidden_size*2)

        energy = self.relu(self.energy(torch.cat((h_reshaped, encoder_states), dim=2)))     # (seq_len, N, 1)

        attention = self.softmax(energy)        #(seq_len, N, 1)

        # (seq_len, N, 1) snk + (seq_len, N, hidden_size*2) snl => (1, N, hidden_size*2) kn1
        context_vector = torch.einsum("snk,sn1->kn1", attention, encoder_states)
        rnn_input = torch.cat((context_vector, embedding), dim=2)       # (1, N, hidden_size*2+embedding_size)

        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))       # outputs= (1, N, hidden_size)

        prediction = self.fc(outputs).squeeze(0)    # (N, hidden_size)

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __int__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,
                source,
                target,
                teacher_force_ratio = 0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(english.vocab)

        outputs = torch.zeors(target_len,
                              batch_size,
                              target_vocab_size).to(device)
        encoder_states, hidden, cell = self.encoder

        x = target[0]

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, encoder_states, hidden, cell)
            outputs[t] = output

            best_word = output.argmax(1)

            x = target[t] if random.random() < teacher_force_ratio else best_word

        return outputs

device = torch.device
load_model = False
save_model = True

NUM_EPOCHS = 100
LEARNING_RATE = 3e-4
BATCH_SIZE = 32

encoder_input_size = len(german.vacob)
decoder_input_size = len(english.vocab)
output_size = decoder_input_size

encoder_embedding_size = 300
decoder_embedding_size = 300

HIDDEN_SIZE = 1024
NUM_LAYERS = 4
ENC_DROPOUT = 0.0
DEC_DROPOUT = 0.0

writer = SummaryWriter(f"runs/loss_plot")
step = 0

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device)

encoder_net = Encoder(encoder_input_size,
                      encoder_embedding_size,
                      HIDDEN_SIZE,
                      NUM_LAYERS,
                      ENC_DROPOUT).to(device)

decoder_net = Decoder(decoder_input_size,
                      decoder_embedding_size,
                      HIDDEN_SIZE,
                      output_size,
                      NUM_LAYERS,
                      DEC_DROPOUT).to(device)

model = Seq2Seq(encoder_net, decoder_net).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

pad_idx = english.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

sentence = ()

for epoch in range(NUM_EPOCHS):
    print(f"[Epoch {epoch} / {NUM_EPOCHS}]")

    if save_model:
        checkpoint = { "state_dict": model.state_dict(),
                       "optimizer": optimizer.state_dict(),
                       }
        save_checkpoint(checkpoint)

    model.eval()

    translated_sentence = translate_sentence(model,
                                             sentence,
                                             german,
                                             english,
                                             device,
                                             max_length=50)

    print(f"Translated sentece: \n {translated_sentence}")

    model.train()

    for batch_idx, batch in enumerate(train_iterator):
        input_data = batch.src.to(device)
        target = batch.trg.to(device)

        # forward
        output = model(input_data, target)      # (trg_len, batch_size, output_dim)

        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()
        loss = criterion(output, target)

        loss.backward()     # back propagation

        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=1)

        optimizer.step()

        writer.add_scalar("Training loss", loss, global_step=step)
        step +=1

score = bleu(test_data[1:100], model, german, english, device)
print(f"Bleu score {score * 100:.2f}")