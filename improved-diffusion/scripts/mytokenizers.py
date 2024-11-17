# sci_bert_dir = None
# assert(sci_bert_dir is not None,'fill sci_bert_dir fist.')

import selfies as sf
################################
import torch
import os
from os import path as osp
import regex
import random
################################
def getrandomnumber(numbers,k,weights=None):
    if k==1:
        return random.choices(numbers,weights=weights,k=k)[0]
    else:
        return random.choices(numbers,weights=weights,k=k)
    
# simple smiles tokenizer
# treat every charater as token
def build_simple_smiles_vocab(dir):
    assert dir is not None, 'dir and smiles_vocab can not be None at the same time.'
    if not osp.exists(osp.join(dir,'simple_smiles_tokenizer_vocab.txt')):
        # print('Generating Vocabulary for {} ...'.format(dir))
        dirs = list(osp.join(dir,i) for i in ['train.txt','validation.txt','test.txt'])
        smiles = []
        for idir in dirs:
            with open(idir,'r') as f:
                for i,line in enumerate(f):
                    if i==0: continue
                    line = line.split('\t')
                    assert len(line)==3,'Dataset format error.'
                    if line[1]!='*': smiles.append(line[1].strip())   
        char_set = set()
        for smi in smiles:
            for c in smi:
                char_set.add(c)
        vocabstring = ''.join(char_set)
        with open(osp.join(dir,'simple_smiles_tokenizer_vocab.txt'),'w') as f:
            f.write(osp.join(vocabstring))
        return vocabstring
    else:
        print('Reading in Vocabulary...')
        with open(osp.join(dir,'simple_smiles_tokenizer_vocab.txt'),'r') as f:
            vocabstring = f.readline().strip()
        return vocabstring

class regexTokenizer():
    #def __init__(self,path='../../datasets/SMILES/generate_vocab.txt',max_len=256):
    def __init__(self,path='/mntcephfs/lab_data/wangcm/jhy/tgm-dlm-main/tgm-dlm-main/datasets/SMILES/generate_vocab.txt',max_len=256):
        print('Truncating length:',max_len)
        with open(path,'r') as f:
            x = f.readlines()
        pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        self.rg = regex.compile(pattern)
        self.idtotok  = { cnt+3:i.strip() for cnt,i in enumerate(x)}
        self.idtotok.update(
            {
                0:'[PAD]',
                1:'[SOS]',
                2:'[EOS]'
            }
        )
        self.vocab_size = len(self.idtotok) #SOS, EOS, pad
        self.toktoid = { v:k for k,v in self.idtotok.items()}
        self.max_len = max_len
    def decode_one(self, iter):
        # return "".join([self.ind2Letter(i) for i in iter]).replace('[SOS]','').replace('[EOS]','').replace('[PAD]','')
        return "".join([self.idtotok[i.item()] for i in iter])
    def decode(self,ids:torch.tensor):
        if len(ids.shape)==1:
            return [self.decode_one(ids)]
        else:
            smiles  = []
            for i in ids:
                smiles.append(self.decode_one(i))
            return smiles
    def __len__(self):
        return self.vocab_size
    def __call__(self,smis:list):
        tensors = []
        if type(smis) is str:
            smis = [smis]
        for i in smis:
            tensors.append(self.encode_one(i))
        return torch.concat(tensors,dim=0)
    def corrupt(self,smis:list):
        tensors = []
        if type(smis) is str:
            smis = [smis]
        for i in smis:
            tensors.append(self.corrupt_one(i))
        return torch.concat(tensors,dim=0)
    def encode_one(self, smi):
        res = [self.toktoid[i] for i in self.rg.findall(smi)]
        res = [1] + res + [2]
        if len(res) < self.max_len:
            res += [0]*(self.max_len-len(res))
        else:
            res = res[:self.max_len]
            res[-1] = 2
        return torch.LongTensor([res])
    
class regexTokenizer_two():
    #def __init__(self,path='../../datasets/SMILES/generate_vocab.txt',max_len=256):
    def __init__(self,path='/mntcephfs/lab_data/wangcm/jhy/tgm-dlm-main/tgm-dlm-main/datasets/Join/muti_vocab.txt',max_len=256):
        print('Truncating length:',max_len)
        with open(path,'r') as f:
            x = f.readlines()
        pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        self.rg = regex.compile(pattern)
        self.idtotok  = { cnt+3:i.strip() for cnt,i in enumerate(x)}
        self.idtotok.update(
            {
                0:'[PAD]',
                1:'[SOS]',
                2:'[EOS]'
            }
        )
        self.vocab_size = len(self.idtotok) #SOS, EOS, pad
        self.toktoid = { v:k for k,v in self.idtotok.items()}
        self.max_len = max_len
    def decode_one(self, iter):
        # return "".join([self.ind2Letter(i) for i in iter]).replace('[SOS]','').replace('[EOS]','').replace('[PAD]','')
        return "".join([self.idtotok[i.item()] for i in iter])
    def decode(self,ids:torch.tensor):
        if len(ids.shape)==1:
            return [self.decode_one(ids)]
        else:
            smiles  = []
            for i in ids:
                smiles.append(self.decode_one(i))
            return smiles
    def __len__(self):
        return self.vocab_size
    def __call__(self,smis:list):
        tensors = []
        if type(smis) is str:
            smis = [smis]
        for i in smis:
            tensors.append(self.encode_one(i))
        return torch.concat(tensors,dim=0)
    def corrupt(self,smis:list):
        tensors = []
        if type(smis) is str:
            smis = [smis]
        for i in smis:
            tensors.append(self.corrupt_one(i))
        return torch.concat(tensors,dim=0)
    def encode_one(self, smi):
        res = [self.toktoid[i] for i in self.rg.findall(smi)]
        res = [1] + res + [2]
        if len(res) < self.max_len:
            res += [0]*(self.max_len-len(res))
        else:
            res = res[:self.max_len]
            res[-1] = 2
        return torch.LongTensor([res])


class selfTokenizer():
    
    #def __init__(self,path='../../datasets/SMILES/generate_vocab.txt',max_len=256):
    def __init__(self,path='/mntcephfs/lab_data/wangcm/jhy/tgm-dlm-main/tgm-dlm-main/datasets/SELFIES/generate_vocab_self.txt',max_len=256):
        print('Truncating length:',max_len)
        with open(path,'r') as f:
            x = f.readlines()
        pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        self.rg = regex.compile(pattern)
        self.idtotok  = { cnt+3:i.strip() for cnt,i in enumerate(x)}
        self.idtotok.update(
            {
                0:'[nop]',
                1:'[SOS]',
                2:'[EOS]'
            }
        )
        self.vocab_size = len(self.idtotok) #SOS, EOS, pad
        self.toktoid = { v:k for k,v in self.idtotok.items()}
        self.max_len = max_len
    def decode_one(self, iter):
        # return "".join([self.ind2Letter(i) for i in iter]).replace('[SOS]','').replace('[EOS]','').replace('[PAD]','')
        return "".join([self.idtotok[i.item()] for i in iter])
    def decode(self,ids:torch.tensor):
        if len(ids.shape)==1:
            return [self.decode_one(ids)]
        else:
            smiles  = []
            for i in ids:
                smiles.append(sf.decoder(self.decode_one(i)))
            return smiles
    def __len__(self):
        return self.vocab_size
    def __call__(self,smis:list):
        tensors = []
        if type(smis) is str:
            smis = [smis]
        for i in smis:
            tensors.append(self.encode_one(i))
        return torch.concat(tensors,dim=0)
    def encode_one(self, selfies):
        pad_to_len = max(sf.len_selfies(s) for s in selfies)
        label, one_hot = sf.selfies_to_encoding(
            selfies=selfies,
            vocab_stoi=self.toktoid,
            pad_to_len=pad_to_len,
            enc_type="both"
        )
        res = [1] + label + [2]
        if len(res) < self.max_len:
            res += [0]*(self.max_len-len(res))
        else:
            res = res[:self.max_len]
            res[-1] = 2
        return torch.LongTensor([res])
    
class selfTokenizer_two():
    
    #def __init__(self,path='../../datasets/SMILES/generate_vocab.txt',max_len=256):
    def __init__(self,path='/mntcephfs/lab_data/wangcm/jhy/tgm-dlm-main/tgm-dlm-main/datasets/Join/muti_vocab.txt',max_len=256):
        print('Truncating length:',max_len)
        with open(path,'r') as f:
            x = f.readlines()
        pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        self.rg = regex.compile(pattern)
        self.idtotok  = { cnt+3:i.strip() for cnt,i in enumerate(x)}
        self.idtotok.update(
            {
                0:'[nop]',
                1:'[SOS]',
                2:'[EOS]'
            }
        )
        self.vocab_size = len(self.idtotok) #SOS, EOS, pad
        self.toktoid = { v:k for k,v in self.idtotok.items()}
        self.max_len = max_len
    def decode_one(self, iter):
        return "".join([self.idtotok[i.item()] for i in iter])
    def decode(self,ids:torch.tensor):
        if len(ids.shape)==1:
            mol = self.decode_one(ids)
            return  [sf.decoder(mol.replace('[SOS]','').replace('[EOS]','').replace('[nop]',''))] , [mol]
        else:
            smiles  = []
            selfies = []
            for i in ids:
                mol = self.decode_one(i)
                selfies.append(mol)
                smiles.append(sf.decoder(mol.replace('[SOS]','').replace('[EOS]','').replace('[nop]','')))
            return smiles , selfies
    def __len__(self):
        return self.vocab_size
    def __call__(self,smis:list):
        tensors = []
        if type(smis) is str:
            smis = [smis]
        for i in smis:
            tensors.append(self.encode_one(i))
        return torch.concat(tensors,dim=0)
    def encode_one(self, selfies):
        pad_to_len = max(sf.len_selfies(s) for s in selfies)
        label, one_hot = sf.selfies_to_encoding(
            selfies=selfies,
            vocab_stoi=self.toktoid,
            pad_to_len=pad_to_len,
            enc_type="both"
        )
        res = [1] + label + [2]
        if len(res) < self.max_len:
            res += [0]*(self.max_len-len(res))
        else:
            res = res[:self.max_len]
            res[-1] = 2
        return torch.LongTensor([res])



class SimpleSmilesTokenizer():
    def __init__(self,dir=None,smiles_vocab=None, max_len=256):
        self.smiles_vocab = smiles_vocab if smiles_vocab is not None else build_simple_smiles_vocab(dir)
        self.max_len = max_len
        self.vocab_size = len(self.smiles_vocab) + 3 #SOS, EOS, pad
        
        self.SOS = self.vocab_size - 2
        self.EOS = self.vocab_size - 1
        self.pad = 0

    def letterToIndex(self, letter):
        return self.smiles_vocab.find(letter) + 1 #skip 0 == [PAD]
    
    def get_left_right_id(self):
        return self.letterToIndex('('),self.letterToIndex(')')

    def ind2Letter(self, ind):
        if ind == self.SOS: return '[SOS]'
        if ind == self.EOS: return '[EOS]'
        if ind == self.pad: return '[PAD]'
        return self.smiles_vocab[ind-1]

    def decode(self,ids:torch.tensor):
        if len(ids.shape)==1:
            return [self.decode_one(ids)]
        else:
            smiles  = []
            for i in ids:
                smiles.append(self.decode_one(i))
            return smiles
    def decode_one(self, iter):
        # return "".join([self.ind2Letter(i) for i in iter]).replace('[SOS]','').replace('[EOS]','').replace('[PAD]','')
        return "".join([self.ind2Letter(i) for i in iter])
    def __len__(self):
        return self.vocab_size

    def __call__(self,smis:list):
        tensors = []
        if type(smis) is str:
            smis = [smis]
        for i in smis:
            tensors.append(self.encode_one(i))
        return torch.concat(tensors,dim=0)
    def encode_one(self, smi):
        tensor = torch.zeros(1, self.max_len, dtype=torch.int64)
        tensor[0,0] = self.SOS
        for li, letter in enumerate(smi):
            tensor[0,li+1] = self.letterToIndex(letter)
            if li + 3 == self.max_len: break
        tensor[0, li+2] = self.EOS

        return tensor

if __name__=="__main__":
    tok = regexTokenizer()
    smiles = ['[210Po]',
        'C[C@H]1C(=O)[C@H]([C@H]([C@H](O1)OP(=O)(O)OP(=O)(O)OC[C@@H]2[C@H](C[C@@H](O2)N3C=C(C(=O)NC3=O)C)O)O)O',
        'C(O)P(=O)(O)[O-]',
        'CCCCCCCCCCCC(=O)OC(=O)CCCCCCCCCCC',
        'C[C@]12CC[C@H](C[C@H]1CC[C@@H]3[C@@H]2CC[C@]4([C@H]3CCC4=O)C)O[C@H]5[C@@H]([C@H]([C@@H]([C@H](O5)C(=O)O)O)O)O'
        ]
    z = tok.corrupt(smiles)