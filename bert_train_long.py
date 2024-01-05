import pandas as pd
import numpy as np
import re
import torch
import random
from zipfile import ZipFile
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn.functional as F

TEXT_COLUMN_NAME = 'text'
LABEL_COLUMN_NAME = 'label'
MAX_LEN = 512
LOSS_FUNCTION = nn.CrossEntropyLoss()

class BertClassifier(nn.Module):
    def __init__(self, freeze_bert = False):  # Set freeze_bert to `False` to fine-tune the BERT model
        
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of classifier, and number of labels
        D_in, H, D_out = 768, 50, 2

        # Instantiate BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Instantiate a one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Linear(H, D_out))

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
    def forward(self, input_ids, attention_mask):
        '''Feed input to BERT and the classifier to compute logits.
        
            Args:
            input_ids (torch.Tensor): an input tensor with shape (batch_size, max_length)
            attention_mask (torch.Tensor): a tensor that hold attention mask information with shape (batch_size, max_length)
            
            Returns:
            logits (torch.Tensor): an output tensor with shape (batch_size, num_labels)'''
        
        # Feed input to BERT
        outputs = self.bert(input_ids = input_ids, attention_mask = attention_mask)
        
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits
    
    @classmethod
    def from_pickled(cls, path):
        '''Unpickles the file given by path, returns an instance of the class initialized with the classifier object.'''
        
        classifier = pickle.load(open(path,'rb'))
        return cls(classifier)
    
    def test(self, data):
        '''Evaluates the model on test data, and returns the accuracy.
        
         Args:
         data (df): a dataframe with text and label columns'''
        
        y_test = data[LABEL_COLUMN_NAME].values
        
        input_ids, input_masks = preprocess_for_bert(data[TEXT_COLUMN_NAME])
        input_dataset = TensorDataset(input_ids, input_masks)
        input_sampler = SequentialSampler(input_dataset)
        input_dataloader = DataLoader(input_dataset, sampler = input_sampler, batch_size=16)
        
        probs = bert_predict(main.bert_classifier, input_dataloader)
        
        return evaluate_accuracy(probs, y_test)
    
    def predict_class(self, text): 
        '''Returns a prediction (in an array) for the sentiment of text using a BERT classifier.
            
         Args:
         text (str): a string input'''
        
        if type(text) == str:
            series_text = pd.Series(text)
            test_inputs, test_masks = preprocess_for_bert(series_text)  # This returns input_ids and attention_masks as tensors
            # Create the DataLoader
            test_dataset = TensorDataset(test_inputs, test_masks)
            test_sampler = SequentialSampler(test_dataset)
            test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=16)
            
            prob = bert_predict(bert_classifier, test_dataloader)  # bert_predict returns a 2-d array
            threshold = 0.5
            prediction = np.where(prob[:, 1] > threshold, 'pos', 'neg') 
            
        else:
            print('Input must be a string.')

        return prediction
        
    
    def store(self, path):
        '''Accepts a file path and stores the model at that file path using the pickle library.'''
        
        with open(path,'wb') as f:
            pickle.dump(classifier, f)

            
def clean(text): 
    '''Converts the input text into lower case, removes <br> tags, punctuation, whitespace. 
    Returns the processed words in a list.
    
        Args:
        text(str): input text'''
    
    text = re.sub('<.*?>',' ', text)
    text = re.sub('\(', '', text)
    text = re.sub('\)', '', text)
    text = re.sub('\s+',' ', text)
    
    return text


def preprocess_for_bert(data):
    '''Performs required preprocessing steps for pretrained BERT. 
        
        Args:
        data (np.array): Array of texts to be processed.
        
        Returns:
        input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
        attention_masks (torch.Tensor): Tensor of indices specifying which tokens should be attended to by the model.
    '''
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    
    input_ids = []
    attention_masks = []

    for text in data:
        encoded_sent = tokenizer.encode_plus(
            text = clean(text),              
            add_special_tokens = True, # Add `[CLS]` and `[SEP]`
            max_length = MAX_LEN,           
            padding ='max_length',
            truncation = True,         # Sentences in my data are longer than 512, max length of sequence allowed
            return_attention_mask = True      
            )
               
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks


def initialize_model(epochs=4):
    """Initialize the Bert Classifier, the optimizer and the learning rate scheduler."""
    
    # Instantiate Bert Classifier
    bert_classifier = BertClassifier(freeze_bert = False)

    # Tell PyTorch to run the model on GPU   # TO DO
    bert_classifier.to(device)

    # Create the optimizer
    optimizer = AdamW(bert_classifier.parameters(),
                      lr=5e-5,    # Default learning rate
                      eps=1e-8)    # Default epsilon value

    # Total number of training steps
    total_steps = len(train_dataloader) * epochs

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0, # Default value
                                                num_training_steps=total_steps)
    return bert_classifier, optimizer, scheduler


def set_seed(seed_value=42):
    '''Set seed for reproducibility.'''
    
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    
def train(model, train_dataloader, val_dataloader=None, epochs=4, evaluation=False):
    """Train the BertClassifier model."""
    
    # Start training loop
    print("Start training...\n")
    for epoch_i in range(epochs):

        # TRAINING
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*70)

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0

        # Put the model into the training mode
        model.train()

        # For each batch of training data,
        for step, batch in enumerate(train_dataloader):
            batch_counts += 1
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            logits = model(b_input_ids, b_attn_mask)

            # Compute loss and accumulate the loss values
            loss = LOSS_FUNCTION(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()

            # Print the loss values and time elapsed for every 20 batches
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch

                # Print training results
                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        print("-"*70)
        
        # EVALUATION
        if evaluation == True:
            # After completing each training epoch, measure the model's performance on our validation set.
            val_loss, val_accuracy = evaluate(model, val_dataloader)

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("-"*70)
        print("\n")
    
    print("Training complete!")


def evaluate(model, val_dataloader):   
    '''After the completion of each training epoch, measure the model's performance on our validation set.'''
    # Put the model into the evaluation mode. 
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []

    # For each batch in our validation set,
    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)

        # Compute loss
        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()

        # Calculate the accuracy rate
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy    


def bert_predict(model, test_dataloader):
    '''Perform a forward pass on the trained BERT model to predict probabilities on the test set.'''
    
    # Put the model into the evaluation mode. 
    model.eval()

    all_logits = []

    # For each batch in our test set,
    for batch in test_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        all_logits.append(logits)
    
    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)

    # Apply softmax to calculate probabilities
    probs = F.softmax(all_logits, dim=1).cpu().numpy()

    return probs

def evaluate_accuracy(probs, y_true):  # This will work on test data as long as it has a label column too. 
    preds = probs[:, 1]
    y_pred = np.where(preds >= 0.5, 1, 0)
    accuracy = accuracy_score(y_true, y_pred)
    
    return (f'Accuracy: {accuracy*100:.2f}%')


def main():
    with ZipFile('imdb ratings.zip') as zf:
        f = zf.open('movie.csv')
    df = pd.read_csv(f)
    train_set = df.sample(2000, random_state=2022, ignore_index=True)
    
    X = train_set[TEXT_COLUMN_NAME].values
    y = train_set[LABEL_COLUMN_NAME].values
    
    # Split the data into training and validation sets.
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=2022)
    
    # Use the GPU if possible.
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    
    # Create input ids and attention masks for texts from training and validation sets.
    train_inputs, train_masks = preprocess_for_bert(X_train)
    val_inputs, val_masks = preprocess_for_bert(X_val)
    
    # Convert other data types to torch.Tensor.
    train_labels = torch.tensor(y_train)
    val_labels = torch.tensor(y_val)

    batch_size = 16   # Batch size of 16 or 32 recommended for fine-tuning BERT.

    # Create the DataLoader for training and validation sets.
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)                  # Select the training batches randomly for training.
    train_dataloader = DataLoader(train_data, sampler = train_sampler, batch_size = batch_size)

    val_data = TensorDataset(val_inputs, val_masks, val_labels)
    val_sampler = SequentialSampler(val_data)                 
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)
    
    # Train the model on train set and measure the model's performance on the validation set.
    set_seed(42)    
    bert_classifier, optimizer, scheduler = initialize_model(epochs=2)
    train(bert_classifier, train_dataloader, val_dataloader, epochs=2, evaluation=True)
    
    # Combine train and validation sets, and train the model on the combined data.
    full_train_data = torch.utils.data.ConcatDataset([train_data, val_data])
    full_train_sampler = RandomSampler(full_train_data)
    full_train_dataloader = DataLoader(full_train_data, sampler=full_train_sampler, batch_size=16)

    set_seed(42)
    main.bert_classifier, optimizer, scheduler = initialize_model(epochs=2)
    train(main.bert_classifier, full_train_dataloader, epochs=2)
    

if __name__ == '__main__':
    main()
      
   