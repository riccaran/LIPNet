#Versions
#Python version 3.8.12
#PyTorch version 1.13.1+cu117
#Numpy version 1.21.5

# Libraries loading
import os
import random

import numpy as np
import pandas as pd
from cnn_architecture import CNN
import torch
import torch.optim as optim
import torch.nn as nn

#### Use the model ####

# Padding, batch making and loader making
def pad_and_batch_sequences(embeddings_dict, size, batch_size):
    # splitting
    uniprots = embeddings_dict.keys()
    X = embeddings_dict.values()
    
    # making masks
    mask = [np.ones(elem.shape[0], dtype=int) for elem in X]

    # conversion in tensors    
    tensor_sequences_X = [torch.tensor(sequence, dtype=torch.float) for sequence in X]
    tensor_sequences_mask = [torch.tensor(sequence, dtype=torch.float) for sequence in mask]

    # pad first sequences to 1000
    tensor_sequences_X[0] = nn.functional.pad(
        tensor_sequences_X[0],
        (0, 0, 0, size - tensor_sequences_X[0].shape[0]),
        'constant', 0
    )
    tensor_sequences_mask[0] = nn.functional.pad(
        tensor_sequences_mask[0],
        (0, size - tensor_sequences_mask[0].shape[0]),
        'constant', 0
    )

    # padding
    padded_sequences_X = pad_sequence(tensor_sequences_X, batch_first=True, padding_value=0.0)
    padded_sequences_mask = pad_sequence(tensor_sequences_mask, batch_first=True, padding_value=0.0)
    
    # making batches
    total_sequences_X = padded_sequences_X.shape[0]
    X_batches = [padded_sequences_X[i:i + batch_size] for i in range(0, total_sequences_X, batch_size)]

    total_sequences_mask = padded_sequences_mask.shape[0]
    mask_batches = [padded_sequences_mask[i:i + batch_size] for i in range(0, total_sequences_mask, batch_size)]
    uniprots_batches = [uniprots[i:i + batch_size] for i in range(0, total_sequences_mask, batch_size)]
    
    return uniprots_batches, X_batches, mask_batches

def making_loader(embeddings_dict, size, batch_size):
    uniprots_batches, X_batches, mask_batches = pad_and_batch_sequences(embeddings_dict, size, batch_size)
    loader_set = [(X, mask) for X, mask in zip(X_batches, mask_batches)]
    return uniprots_batches, loader_set





def main():
    #### Set parameters ####

    # Random seeds
    random_seed = 4

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    # Final parameters
    lr = 1e-5
    weight_decay = 1e-3
    channels = 512
    dropout = 0.75
    sequence_cut = 1000
    batch_size = 16

    #### Build the input ####

    input = "input.fasta"

    with open(input, "r") as f:
        lines = f.read().splitlines()

    ind = 0
    input_split = dict() # Keys are porteins UniProt's IDs and values are string amino acid sequences

    while ind < len(lines):
        input_split[lines[ind][1:]] = (lines[ind + 1])
        ind += 2

    #### Embeddings calculation here
    #### The final "embeddings_dict" file would a dictionary in which keys are porteins UniProt's IDs and values are amino acid-based embeddings

    #### Making the output folder
    output_folder = "disorder"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

        # Model loading
    model = CNN(channels = channels , dropout = dropout)
    optimizer = optim.AdamW(model.parameters(), lr = lr , weight_decay = weight_decay )

    checkpoint = torch.load('cnn_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Model evaluation
    model.eval()

    uniprots_batches, test = making_loader(X_test, sequence_cut, batch_size) # 1000 is the limit for cutting sequences

    predictions_proba = dict()

    with torch.no_grad():
        for uniprots_batches, (inputs, mask) in zip(uniprots_batches, test):
            outputs = model(inputs, mask)
            outputs = torch.sigmoid(outputs)
            for example in range(batch_size):
                predictions_proba[uniprots_batches[example]] = outputs[examples, :]

    # Scores saving
    for uniprot, scores in predictions_proba.items():
        sequence = input_split[uniprot][:sequence_cut]
        prot_scores = zip(sequence, scores)
        prot_scores[2] = np.nan
        df.to_csv("{}/{}.caid".forat(output_folder, uniprot), sep='\t', header = False, index=False)


if __name__ == "__main__":
    main()
