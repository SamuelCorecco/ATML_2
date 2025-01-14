import sys
import re
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
from langdetect import detect, LangDetectException
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

class JobParams:
    def __init__(self, num_epochs, lr, batch_size, langs, train, evaluate, interact):
        self.num_epochs = num_epochs
        self.lr         = lr
        self.batch_size = batch_size
        self.langs      = langs        
        self.train      = train
        self.evaluate   = evaluate
        self.interact   = interact

def get_checkpoint_name(job_params):
    return f"checkpoint, langs: {job_params.langs}, epochs: {job_params.num_epochs}, batch_size: {job_params.batch_size}.pkl"

# Function to split text into prompt-response pairs
def split_interactions(text):
    # Remove newlines
    text = text.replace("\n", " ").strip()

    # Use regex to find all Human and Assistant parts
    pattern = r"(### Human:.*?)(?=(### Human:|$))"  # Match one Human section and stop at the next
    matches = re.findall(pattern, text, flags=re.DOTALL)
    
    pairs = []
    for match in matches:
        human_part = match[0]
        # Extract Human prompt and Assistant response
        human_prompt = re.search(r"### Human:(.*?)### Assistant:", human_part, flags=re.DOTALL)
        assistant_response = re.search(r"### Assistant:(.*)", human_part, flags=re.DOTALL)
        
        if human_prompt and assistant_response:
            prompt   = human_prompt.group(1).strip()
            response = assistant_response.group(1).strip()
            pairs.append((prompt, response))
    return pairs

def prompt_response_split(df, text_column="text"):
    all_pairs = []

    for _, row in df.iterrows():
        text = row[text_column]
        pairs = split_interactions(text)
        all_pairs.extend(pairs)

    return pd.DataFrame(all_pairs, columns=["prompt", "response"])

def get_tokenizer(job_params, train_df, test_df, num_words=None):
    # If tokenizer exists, return it
    filename = get_checkpoint_name(job_params)
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            checkpoint = pickle.load(f)
            return checkpoint['tokenizer']

    # Combine all text data
    all_text = pd.concat([train_df["prompt"], train_df["response"], 
                           test_df["prompt"], test_df["response"]], axis=0).tolist()

    # Initialize and fit the tokenizer
    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(all_text)

    # Manually add <START> and <END> tokens if not present
    for token in ["<START>", "<END>"]:
        if token not in tokenizer.word_index:
            tokenizer.word_index[token] = len(tokenizer.word_index) + 1

    # Update tokenizer's index-to-word mapping
    tokenizer.index_word = {idx: word for word, idx in tokenizer.word_index.items()}

    return tokenizer

def prepare_tensor_dataset(df_pairs, tokenizer, padding="post"):
    # Tokenize the prompts and responses
    prompts   = tokenizer.texts_to_sequences(df_pairs["prompt"])
    responses = tokenizer.texts_to_sequences(df_pairs["response"])

    # Pad sequences
    prompts_padded   = pad_sequences(prompts, padding=padding)
    responses_padded = pad_sequences(responses, padding=padding)

    return TensorDataset(
        torch.tensor(prompts_padded, dtype=torch.long), 
        torch.tensor(responses_padded, dtype=torch.long),
    )

def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def get_loader(dataset, batch_size, shuffle=False, device='cpu'):
    extra_opts = {}
    if device == 'gpu':
        extra_opts['num_workers']     = 4
        extra_opts['prefetch_factor'] = 2
        extra_opts['pin_memory']      = True
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **extra_opts)

class Seq2SeqModel(nn.Module):
    def __init__(self, job_params, tokenizer, embedding_dim, hidden_dim):
        super(Seq2SeqModel, self).__init__()

        self.vocab_size    = len(tokenizer.word_index) + 1
        self.embedding_dim = embedding_dim
        self.hidden_dim    = hidden_dim

        self.start_epoch  = 0
        self.train_losses = []
        self.valid_losses = []

        self.embedding = nn.Embedding(self.vocab_size, embedding_dim, padding_idx=0)
        self.encoder   = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.decoder   = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc        = nn.Linear(hidden_dim, self.vocab_size)

    def load_checkpoint(self, job_params, optimizer, tokenizer):
        filename = get_checkpoint_name(job_params)
        
        if not os.path.exists(filename):
            print(f"No checkpoint found. Starting from scratch.")
        else:
            with open(filename, 'rb') as f:
                checkpoint = pickle.load(f)
            print(f"Successfully loaded checkpoint!")

            self.start_epoch  = checkpoint['epoch']
            self.train_losses = checkpoint['train_losses']
            self.valid_losses = checkpoint['valid_losses']

            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            tokenizer = checkpoint['tokenizer']
            
    def store_checkpoint(self, job_params, optimizer, tokenizer, epoch):
        filename = get_checkpoint_name(job_params)

        checkpoint = {
            'model_state_dict':     self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch':                epoch,
            'train_losses':         self.train_losses,
            'valid_losses':         self.valid_losses,
            'tokenizer':            tokenizer,
        }

        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"Stored checkpoint into '{filename}'")

    def forward(self, encoder_input, decoder_input):
        encoder_embedded  = self.embedding(encoder_input)
        _, (hidden, cell) = self.encoder(encoder_embedded)

        decoder_embedded  = self.embedding(decoder_input)
        decoder_output, _ = self.decoder(decoder_embedded, (hidden, cell))

        output = self.fc(decoder_output)
        return output

    def train_loop(self, job_params, train_loader, valid_loader, optimizer, tokenizer, start_token="<START>"):
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        scaler = torch.amp.GradScaler()

        start_token_idx = tokenizer.word_index[start_token]

        for epoch in range(self.start_epoch, job_params.num_epochs):
            self.train()
            total_loss = 0

            for encoder_input, decoder_input in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{job_params.num_epochs}"):
                # Move inputs to device
                encoder_input, decoder_input = encoder_input.to(device), decoder_input.to(device)

                # Add <START> token to decoder_input
                start_tokens  = torch.full((decoder_input.size(0), 1), start_token_idx, dtype=torch.long, device=device)
                decoder_input = torch.cat([start_tokens, decoder_input[:, :-1]], dim=1)  # Shift right and prepend <START>

                # Construct decoder_target by excluding the first token
                decoder_target = decoder_input[:, 1:]

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.amp.autocast(device_type="cuda"):
                    outputs = self(encoder_input, decoder_input)
                    outputs = outputs[:, :-1, :]
                    loss = criterion(outputs.reshape(-1, self.vocab_size), decoder_target.contiguous().view(-1))

                # Backward pass and optimization
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()

            # Average loss for the epoch
            avg_loss = total_loss / len(train_loader)
            self.train_losses.append(avg_loss)
            print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

            # Save checkpoint at the end of each epoch
            self.store_checkpoint(job_params, optimizer, tokenizer, epoch)

            # Validate if validation loader is provided
            if valid_loader is not None:
                self.validate(valid_loader, tokenizer)
        
    def validate(self, valid_loader, tokenizer):
        self.eval()
        total_valid_loss = 0
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        with torch.no_grad():
            for encoder_input, decoder_input in tqdm(valid_loader, desc="Validating"):
                encoder_input, decoder_input = encoder_input.to(device), decoder_input.to(device)

                start_tokens = torch.full((decoder_input.size(0), 1), tokenizer.word_index["<START>"], dtype=torch.long, device=device)
                decoder_input = torch.cat([start_tokens, decoder_input[:, :-1]], dim=1)

                # Construct decoder_target
                decoder_target = decoder_input[:, 1:]

                # Forward pass
                outputs = self(encoder_input, decoder_input)
                outputs = outputs[:, :-1, :]
                loss = criterion(outputs.reshape(-1, self.vocab_size), decoder_target.contiguous().view(-1))

                total_valid_loss += loss.item()

        avg_valid_loss = total_valid_loss / len(valid_loader)
        self.valid_losses.append(avg_valid_loss)
        print(f"Validation Loss: {avg_valid_loss:.4f}")

    def evaluate(self, test_loader, tokenizer, device, strategy='bleu', max_len=50, start_token="<START>"):
        self.eval()
        total_bleu_score, count = 0, 0
        references, hypotheses = [], []

        start_token_idx = tokenizer.word_index[start_token]

        with torch.no_grad():
            for encoder_input, decoder_input in tqdm(test_loader, desc="Evaluating"):
                encoder_input, decoder_input = encoder_input.to(device), decoder_input.to(device)

                # Generate decoder_target dynamically
                decoder_target = decoder_input[:, 1:]  # Exclude the first token
                decoder_input = torch.cat([torch.full((decoder_input.size(0), 1), start_token_idx, dtype=torch.long, device=device), decoder_input[:, :-1]], dim=1)
                
                # Get model predictions
                output = self(encoder_input, decoder_input)
                predicted = torch.argmax(output, dim=-1)

                # Convert token sequences to text
                target_texts    = tokenizer.sequences_to_texts(decoder_target.cpu().numpy())
                predicted_texts = tokenizer.sequences_to_texts(predicted.cpu().numpy())

                # Collect references and hypotheses for evaluation
                for target, predicted in zip(target_texts, predicted_texts):
                    references.append([target.split()])  # BLEU expects list of reference lists
                    hypotheses.append(predicted.split())
                    total_bleu_score += sentence_bleu([target.split()], predicted.split(), smoothing_function=SmoothingFunction().method1)
                    count += 1

        # Calculate BLEU or BERT score
        if strategy == 'bleu':
            avg_bleu_score = total_bleu_score / count if count > 0 else 0
            print(f"BLEU Score: {avg_bleu_score:.4f}")
            return avg_bleu_score
        elif strategy == 'bert':
            P, R, F1 = bert_score([" ".join(h) for h in hypotheses], [" ".join(r[0]) for r in references], lang="en", verbose=True)
            print(f"BERT Score - Precision: {P.mean().item():.4f}, Recall: {R.mean().item():.4f}, F1: {F1.mean().item():.4f}")
            return P.mean().item(), R.mean().item(), F1.mean().item()
        else:
            raise ValueError("Unknown evaluation strategy. Choose either 'bleu' or 'bert'.")
        
    def respond(self, tokenizer, device, input_text, max_len=50, start_token="<START>", end_token="<END>"):
        model.eval()

        # Tokenize and pad the input text
        input_sequence = tokenizer.texts_to_sequences([input_text])
        input_sequence = pad_sequences(input_sequence, maxlen=max_len, padding="post")
        input_tensor = torch.tensor(input_sequence, dtype=torch.long).to(device)

        # Prepare the decoder input with the start token
        start_idx = tokenizer.word_index[start_token]
        end_idx = tokenizer.word_index[end_token]
        decoder_input = torch.tensor([[start_idx]], dtype=torch.long).to(device)

        generated_tokens = []

        with torch.no_grad():
            # Encode the input sequence
            _, (hidden, cell) = self.encoder(self.embedding(input_tensor))

            for _ in range(max_len):
                # Pass the decoder input through the decoder
                decoder_embedded = self.embedding(decoder_input)
                decoder_output, (hidden, cell) = self.decoder(decoder_embedded, (hidden, cell))

                # Predict the next token
                next_token = torch.argmax(self.fc(decoder_output), dim=-1).item()
                if next_token == end_idx:
                    break

                generated_tokens.append(next_token)
                decoder_input = torch.tensor([[next_token]], dtype=torch.long).to(device)

        # Convert tokens to text
        response = tokenizer.sequences_to_texts([generated_tokens])[0]
        return response

    def interact(self, tokenizer, device):
        print("Chat with the bot! Type 'exit()' to end the conversation.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit()":
                print("Goodbye!")
                break

            response = self.respond(tokenizer, device, user_input)
            print(f"Bot: {response}")

def filter_langs(df, langs=[]):
    def check_lang(text):
        try:
            lang = detect(text)
            return lang in langs
        except LangDetectException:
            return False

    return df if langs == [] else df[df["text"].apply(check_lang)]

if __name__ == "__main__":
    # 0. Define paramaters
    job_params = JobParams(
        num_epochs = sys.argv[1],
        lr         = 1e-3,
        batch_size = sys.argv[2],
        langs      = sys.argv[3].split("+"),
        train      = True,
        evaluate   = True,
        interact   = False,
    )

    # 1. Load datasets
    data_src = [f"hf://datasets/timdettmers/openassistant-guanaco/openassistant_best_replies_{s}.jsonl" for s in ['train', 'eval']]
    df_train = pd.read_json(data_src[0], lines=True)
    df_test  = pd.read_json(data_src[1], lines=True)

    # 2. Retain only valid languages
    df_train = filter_langs(df_train, job_params.langs)
    df_test  = filter_langs(df_test, job_params.langs)

    # 3. Preprocess datasets
    df_train_pairs = prompt_response_split(df_train)
    df_train_pairs, df_valid_pairs = train_test_split(df_train_pairs, test_size=0.2, random_state=42)
    df_test_pairs  = prompt_response_split(df_test)
        
    # 4. Obtain tokenizer
    tokenizer = get_tokenizer(job_params, df_train_pairs, df_test_pairs)

    if job_params.train:
        # 5. Prepare datasets
        train_dataset = prepare_tensor_dataset(df_train_pairs, tokenizer)
        test_dataset  = prepare_tensor_dataset(df_test_pairs,  tokenizer)
        valid_dataset = prepare_tensor_dataset(df_valid_pairs, tokenizer)

        # 6. Get dataset loaders
        device = torch.device(get_device())
        train_loader = get_loader(train_dataset, job_params.batch_size, shuffle=True)
        test_loader  = get_loader(test_dataset,  job_params.batch_size)
        valid_loader = get_loader(valid_dataset, job_params.batch_size)

    # 7. Instantiate the LSTM-based model
    model = Seq2SeqModel(job_params, tokenizer,
        embedding_dim = 200,
        hidden_dim    = 200,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=job_params.lr)
    model.load_checkpoint(job_params, optimizer, tokenizer)

    # 8. Train the model
    if job_params.train:
        model.train_loop(job_params, train_loader, valid_loader, optimizer, tokenizer)

    # 9. Evaluate the model
    if job_params.evaluate:
        model.evaluate(test_loader, tokenizer, device, strategy='bert')

    # 10. Interact with the model
    if job_params.interact:
        model.interact(tokenizer, device)
