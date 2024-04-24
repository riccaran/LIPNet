import argparse
import time
from pathlib import Path
import torch
import h5py
from transformers import T5EncoderModel, T5Tokenizer
import gc
import re 
import time
from Bio import SeqIO
import sys

def save_h5(file_path,dictionary):
    with h5py.File(file_path,'w') as f:
        for key,value in dictionary.items():
            f.create_dataset(key,data=value)

def read_fasta( fasta_path ):
    '''
        Reads in fasta file containing multiple sequences.
        Returns dictionary of holding multiple sequences or only single 
        sequence, depending on input file.
    '''
    
    sequences = dict()
    with open( fasta_path, 'r' ) as fasta_f:
        for line in fasta_f:
            # get uniprot ID from header and create new entry
            if line.startswith('>'):
                uniprot_id = line.replace('>', '').strip()
                # replace tokens that are mis-interpreted when loading h5
                uniprot_id = uniprot_id.replace("/","_").replace(".","_")
                sequences[ uniprot_id ] = ''
            else:
                # repl. all whie-space chars and join seqs spanning multiple lines
                sequences[ uniprot_id ] += ''.join( line.split() ).upper().replace("-","") # drop gaps and cast to upper-case
                
    return sequences

def get_ProtT5_UniRef50_embedding(fasta_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device)
    model.full() if device=='cpu' else model.half()

    model = model.eval()

  
    with open(fasta_path,"r", encoding="utf-8") as handle:

        records = list(SeqIO.parse(handle, "fasta"))
                 
        single_dictionary = {}
        for sliced_rec in [records[i:i+1] for i in range(0, len(records), 1)]:
            gc.collect()
            keys=[record.id for record in sliced_rec]
            
            sequence_examples=[str(record.seq) for record in sliced_rec]
            lens=[len(seq) for seq in sequence_examples]
            print(keys , lens)
            sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]
            ids = tokenizer.batch_encode_plus(sequence_examples, add_special_tokens=True, padding="longest")
            
            input_ids = torch.tensor(ids['input_ids']).to(device)
            attention_mask = torch.tensor(ids['attention_mask']).to(device)
            start_time = time.time()
            with torch.no_grad():
                embedding = model(input_ids=input_ids, attention_mask=attention_mask)
            print("--- %s seconds ---" % (time.time() - start_time))
            
            numpy_embedding = embedding.last_hidden_state.cpu().numpy()
            
           
            for i,embed in enumerate(numpy_embedding):
                new_embed=embed[:lens[i],:]
                # for outputting residue level embedding
                # file.create_dataset(keys[i],data=new_embed)
                single_dictionary[keys[i]] = new_embed

                
    print(single_dictionary)
    print(single_dictionary.keys())
    print(len(single_dictionary))
    return single_dictionary
           
            



if __name__ == '__main__':
    # seq_dict=read_fasta('./1000.fasta')
    parser=argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--output')
    parser.add_argument('--level', help='the level of aggregation: must be equat to residue or protein')
    args=parser.parse_args()
    # seq_dict   = sorted( seq_dict.items(), key=lambda kv: len( seq_dict[kv[0]] ), reverse=True )
    get_ProtT5_UniRef50_embedding(fasta_path=args.input)
