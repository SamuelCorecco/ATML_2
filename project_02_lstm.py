import sys
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import GradScaler, autocast
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score
from collections import Counter
import logging
from langdetect import detect, LangDetectException

def is_english(text):
    if not text or len(text.strip()) < 2:
        return False
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_checkpoint(model, optimizer, epoch, train_losses, valid_losses, num_epochs, batch_size):
    """Save the training state to a file."""
    filename=f"checkpoint-e-{num_epochs}-b-{batch_size}.pkl"

    checkpoint = {
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch':                epoch,
        'train_losses':         train_losses,
        'valid_losses':         valid_losses,
    }
    with open(filename, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"Stored checkpoint into {filename}")

def start_from_checkpoint(model, optimizer, num_epochs, batch_size):
    """Load the training state from a file."""
    filename=f"checkpoint-e-{num_epochs}-b-{batch_size}.pkl"

    if not os.path.exists(filename):
        print(f"No checkpoint found at {filename}. Starting from scratch.")
        return 0, [], []
    else:
        with open(filename, 'rb') as f:
            checkpoint = pickle.load(f)
        print(f"Loaded checkpoint from {filename}")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['train_losses'], checkpoint['valid_losses']

def save_tokenizer(tokenizer, filename='tokenizer.pkl'):
    """Save the tokenizer to a file."""
    with open(filename, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"Stored tokenizer into {filename}")

def load_tokenizer(filename='tokenizer.pkl'):
    """Load the tokenizer from a file."""
    if not os.path.exists(filename):
        print(f"No tokenizer found at {filename}. Creating a new one.")
        return Tokenizer()
    else:
        with open(filename, 'rb') as f:
            tokenizer = pickle.load(f)
        print(f"Loaded tokenizer from {filename}")
        return tokenizer

class Tokenizer:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.vocab_size = 0

    def fit_on_texts(self, texts):
        counter = Counter(word for text in texts for word in text.split())
        self.idx2word = ["<PAD>", "<UNK>"] + [word for word, _ in counter.most_common()]
        self.word2idx = {word: idx for idx, word in enumerate(self.idx2word)}
        self.vocab_size = len(self.idx2word)

    def texts_to_sequences(self, texts):
        return [[self.word2idx.get(word, 1) for word in text.split()] for text in texts]

    def sequences_to_texts(self, sequences):
        return [[self.idx2word[idx] for idx in seq] for seq in sequences]

class Seq2SeqModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Seq2SeqModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, encoder_input, decoder_input):
        encoder_embedded = self.embedding(encoder_input)
        _, (hidden, cell) = self.encoder(encoder_embedded)

        decoder_embedded = self.embedding(decoder_input)
        decoder_output, _ = self.decoder(decoder_embedded, (hidden, cell))

        output = self.fc(decoder_output)
        return output

def tokenize_and_pad(texts, tokenizer, maxlen):
    tokenized = tokenizer.texts_to_sequences(texts)
    tokenized = [torch.tensor(seq, dtype=torch.long) for seq in tokenized]
    return pad_sequence(tokenized, batch_first=True, padding_value=0)[:, :maxlen]

def create_decoder_output(data, tokenizer, maxlen):
    shifted = [torch.tensor(seq[1:], dtype=torch.long) for seq in tokenizer.texts_to_sequences(data)]
    return pad_sequence(shifted, batch_first=True, padding_value=0)[:, :maxlen]

def evaluate(model, test_loader, tokenizer, device, strategy='bert'):
    model.eval()
    total_bleu_score, count = 0, 0
    references, hypotheses = [], []

    with torch.no_grad():
        for encoder_input, decoder_input, target_output in test_loader:
            encoder_input, decoder_input, target_output = (
                encoder_input.to(device),
                decoder_input.to(device),
                target_output.to(device),
            )

            output = model(encoder_input, decoder_input)
            predicted = torch.argmax(output, dim=-1)

            for i in range(len(predicted)):
                reference = " ".join(tokenizer.sequences_to_texts([target_output[i].cpu().numpy()])[0])
                hypothesis = " ".join(tokenizer.sequences_to_texts([predicted[i].cpu().numpy()])[0])
                references.append(reference)
                hypotheses.append(hypothesis)
                total_bleu_score += sentence_bleu([reference.split()], hypothesis.split())
                count += 1

    if strategy == 'bert':
        P, R, F1 = bert_score(hypotheses, references, lang="en", verbose=True)
        return P.mean().item(), R.mean().item(), F1.mean().item()
    elif strategy == 'bleu':
        return total_bleu_score / count if count > 0 else 0
    else:
        raise Exception("Unknown evaluation strategy")

def sample_with_temperature(logits, temperature=1.0, top_k=10):
    logits = logits / temperature 
    top_k_logits, top_k_indices = torch.topk(logits, top_k)
    probabilities = torch.softmax(top_k_logits, dim=-1)
    sampled_idx = torch.multinomial(probabilities, 1).item()
    return top_k_indices[sampled_idx]

def generate_response(model, tokenizer, input_text, maxlen_questions, device, max_len=50, temperature=1.0, top_k=10):
    model.eval()
    input_seq = tokenize_and_pad([input_text], tokenizer, maxlen_questions).to(device).long()

    with torch.no_grad():
        embedded_seq = model.embedding(input_seq)
        _, (hidden, cell) = model.encoder(embedded_seq)

        decoder_input = torch.tensor([tokenizer.word2idx["<START>"]], device=device).unsqueeze(0)

        output_sentence = []
        for _ in range(max_len):
            embedded_decoder_input = model.embedding(decoder_input)
            output, (hidden, cell) = model.decoder(embedded_decoder_input, (hidden, cell))
            logits = model.fc(output.squeeze(0))

            token = sample_with_temperature(logits[0], temperature=temperature, top_k=top_k)
            word = tokenizer.idx2word[token]

            if word == "<END>":
                break

            output_sentence.append(word)
            decoder_input = torch.tensor([[token]], device=device).long()

    return " ".join(output_sentence)

def main(num_epochs, batch_size, train=False):
    # Load datasets
    splits = [
        f"hf://datasets/timdettmers/openassistant-guanaco/openassistant_best_replies_{s}.jsonl"
        for s in ['train', 'eval']
    ]
    df_train = pd.read_json(splits[0], lines=True)
    df_test = pd.read_json(splits[1], lines=True)

    logger.info(f"Started training with num_epochs={num_epochs}, batch_size={batch_size}")

    logger.info('1. Loaded datasets.')

    # Preprocess datasets
    def extract_prompt_and_response(row):
        parts = row.split("### Assistant:")
        prompt = parts[0].replace("### Human:", "").strip()
        response = parts[1].strip() if len(parts) > 1 else ""
        return prompt, response

    df_train[["prompt", "response"]] = df_train["text"].apply(lambda x: pd.Series(extract_prompt_and_response(x)))
    df_test[["prompt", "response"]] = df_test["text"].apply(lambda x: pd.Series(extract_prompt_and_response(x)))

    # Filter out non-English rows
    df_train = df_train[df_train["prompt"].apply(is_english) & df_train["response"].apply(is_english)]
    df_test = df_test[df_test["prompt"].apply(is_english) & df_test["response"].apply(is_english)]

    train_questions = df_train["prompt"].tolist()
    train_answers = ["<START> " + response + " <END>" for response in df_train["response"]]
    test_questions = df_test["prompt"].tolist()
    test_answers = ["<START> " + response + " <END>" for response in df_test["response"]]

    logger.info('2. Preprocessed datasets.')

    tokenizer = load_tokenizer()
    if not tokenizer.word2idx:
        data_for_tokenizer = train_questions + train_answers + test_questions + test_answers
        tokenizer.fit_on_texts(data_for_tokenizer)
        tokenizer.fit_on_texts(data_for_tokenizer)
        tokenizer.idx2word = tokenizer.idx2word[:10000]  # Restrict vocabulary to the top 5000 words
        tokenizer.word2idx = {word: idx for idx, word in enumerate(tokenizer.idx2word)}
        tokenizer.vocab_size = len(tokenizer.idx2word)
        save_tokenizer(tokenizer)

    VOCAB_SIZE = tokenizer.vocab_size

    logger.info('2.5. Loaded tokenizer.')

    # Tokenize and prepare data
    maxlen_questions = max(len(seq) for seq in tokenizer.texts_to_sequences(train_questions + test_questions))
    maxlen_answers = max(len(seq) for seq in tokenizer.texts_to_sequences(train_answers + test_answers))

    encoder_input_data = tokenize_and_pad(train_questions, tokenizer, maxlen_questions)
    decoder_input_data = tokenize_and_pad(train_answers, tokenizer, maxlen_answers)
    decoder_output_data = create_decoder_output(train_answers, tokenizer, maxlen_answers)

    train_dataset = TensorDataset(encoder_input_data, decoder_input_data, decoder_output_data)
    test_dataset = TensorDataset(
        tokenize_and_pad(test_questions, tokenizer, maxlen_questions),
        tokenize_and_pad(test_answers, tokenizer, maxlen_answers),
        create_decoder_output(test_answers, tokenizer, maxlen_answers),
    )

    # Set device
    gpu_opts = {}
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_opts = {'num_workers': 4, 'prefetch_factor': 2, 'pin_memory': True}
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **gpu_opts)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, **gpu_opts)

    logger.info('3. Prepared input data.')

    # Model setup
    embedding_dim = 200
    hidden_dim = 256
    learning_rate = 0.001

    model = Seq2SeqModel(VOCAB_SIZE, embedding_dim, hidden_dim)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    logger.info('4. Created model.')

    model.to(device)

    logger.info('5. Training...')

    # Training loop
    start_epoch, train_losses, valid_losses = start_from_checkpoint(model, optimizer, num_epochs, batch_size)
    if train:
        for epoch in range(start_epoch, num_epochs):
            model.train()
            epoch_loss = 0

            for encoder_input, decoder_input, target_output in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                optimizer.zero_grad()

                encoder_input, decoder_input, target_output = (
                    encoder_input.to(device),
                    decoder_input.to(device),
                    target_output.to(device),
                )

                with torch.cuda.amp.autocast():  # Enable mixed precision
                    output = model(encoder_input, decoder_input)
                    output = output[:, :target_output.size(1), :]
                    output = output.reshape(-1, VOCAB_SIZE)
                    target_output = target_output.reshape(-1)
                    loss = criterion(output, target_output)

                loss = criterion(output, target_output)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            train_losses.append(epoch_loss / len(train_loader))

            # Save checkpoint
            save_checkpoint(model, optimizer, epoch, train_losses, valid_losses, num_epochs, batch_size)

            print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(train_loader)}")

        logger.info('5. Evaluating model...')

        # Evaluation
        strategy = 'bert'
        score = evaluate(model, test_loader, tokenizer, device, strategy=strategy)
        print(f"{strategy.upper()} Score: {score}")

    # Generate response example
    input_text = "What is your name?"
    response = generate_response(model, tokenizer, input_text, maxlen_questions, device)
    print(f"Bot: {response}")

if __name__ == "__main__":
    main(
        num_epochs=int(sys.argv[1]), 
        batch_size=int(sys.argv[2]),
        train=(int(sys.argv[3]) == 1),
    )
