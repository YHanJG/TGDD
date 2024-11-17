from torch.utils.data import DataLoader,Dataset
import torch
import random
import selfies as sf
from rdkit import Chem
from rdkit import RDLogger
from torch.utils.data import DistributedSampler
RDLogger.DisableLog('rdApp.*')
def get_dataloader(dataset,batchsize,rank,world_size,is_self,is_join,is_two_way):
    sampler = DistributedSampler(dataset,num_replicas=world_size,rank=rank,shuffle=True)
    if not is_join and not is_two_way:
        def collate(batch):
            try:
                toked_smis= [i['tok_smiles'] for i in batch]
                desc_states = [i['desc_state'] for i in batch]
                desc_mask = [i['desc_mask'] for i in batch]
                corrupted_toked_smis = [i['corrupted_toked_smis'] for i in batch]
            except:
                toked_smis= [i['tok_selfies'] for i in batch]
                desc_states = [i['desc_state'] for i in batch]
                desc_mask = [i['desc_mask'] for i in batch]
                corrupted_toked_smis = toked_smis 
            return torch.concat(toked_smis,dim=0),torch.concat(desc_states,dim=0),torch.concat(desc_mask,dim=0),torch.concat(corrupted_toked_smis,dim=0)
    else:
        if not is_self:
            def collate(batch):
                toked_smis= [i['tok_smiles'] for i in batch]
                desc_states = [i['desc_state'] for i in batch]
                desc_mask = [i['desc_mask'] for i in batch]
                corrupted_toked_smis = toked_smis
                add_toked = [i['tok_selfies'] for i in batch]
                return torch.concat(toked_smis,dim=0),torch.concat(desc_states,dim=0),torch.concat(desc_mask,dim=0),torch.concat(corrupted_toked_smis,dim=0),torch.concat(add_toked,dim=0)
        else:
            def collate(batch):
                toked_self= [i['tok_selfies'] for i in batch]
                desc_states = [i['desc_state'] for i in batch]
                desc_mask = [i['desc_mask'] for i in batch]
                corrupted_toked_smis = toked_self
                add_toked = [i['tok_smiles'] for i in batch]
                return torch.concat(toked_self,dim=0),torch.concat(desc_states,dim=0),torch.concat(desc_mask,dim=0),torch.concat(corrupted_toked_smis,dim=0),torch.concat(add_toked,dim=0)

    dataloader = DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=False,
        collate_fn=collate,
        sampler=sampler
    )
    def cycle():
        ec = 0
        while True:
            dataloader.sampler.set_epoch(ec)
            for i in dataloader:
                yield i
            ec+=1 
    return iter(cycle())

class ChEBIdataset(Dataset):
    def __init__(self,dir,smi_tokenizer,self_tokenizer,split,replace_desc=False,pre=None,prob=0,load_state=True,corrupt_prob=0.4,mask_desc=False,is_self=False,is_join=False,is_two_way=False):
        super().__init__()
        self.is_two_way = is_two_way
        self.is_self = is_self
        self.dir = dir
        self.smi_tokenizer = smi_tokenizer
        self.self_tokenizer = self_tokenizer
        self.split = split
        self.replace_desc = replace_desc
        self.pre = pre
        self.prob=prob
        self.corrupt_prob = corrupt_prob
        self.is_join = is_join
        print('corruption prob is {}'.format(self.corrupt_prob))
        self.mask_desc= mask_desc
        print('mask_desc is {}'.format(self.mask_desc))
        assert split in ['train','test','validation','mini','train_val_256']
        self.ori_data = self.get_ori_data()
        self.load_state=load_state
        if load_state:
            self.desc_state = self.get_desc_state()
    def get_desc_state(self):
        import os.path as osp
        import spacy
        file_path = osp.join(self.dir,self.split+'_desc_states.pt')
        return torch.load(file_path)
    def get_ori_data(self):
        import os.path as osp
        if not self.is_join and not self.is_two_way:
            if not self.is_self:
                if self.replace_desc:
                    import spacy
                    nlp = spacy.load('en_core_web_sm')
                res = []
                file_path = osp.join(self.dir,self.split+'.txt')
                #修改
                with open(file_path,'r') as f:
                    for i,line in enumerate(f):
                        if i==0: continue
                        line = line.split('\t')
                        assert len(line)==3
                        if line[1]!='*':
                            desc = line[2].strip()
                            if self.replace_desc:
                                doc = nlp(desc)
                                for token in doc:
                                    if token.text == 'is':
                                        desc = 'The molecule ' + desc[token.idx:]
                                        break
                            res.append(
                                (int(line[0]),line[1].strip(),desc)
                            )
                return res
            else:
                res = []
                file_path = osp.join(self.dir, self.split+'_re.txt')
                with open(file_path, 'r') as f:
                    for i, line in enumerate(f):
                        if i==0: continue
                        line = line.split('\t')
                        assert len(line) == 4
                        smiles = line[1].strip()  # 提取 SMILES
                        desc = line[2].strip()  # 提取描述
                        selfies = line[3].replace('\n','')
                        if smiles != '*' and selfies != 'None':
                            # 将 SMILES 转换为 SELFIES
                            try :
                                pre_selfies = sf.encoder(smiles)
                                assert pre_selfies == selfies
                                res.append(
                                    (int(line[0]), smiles,  desc, selfies)  # 添加 SELFIES 列
                                )
                            except:
                                res.append(
                                    (int(line[0]), smiles,  desc, "None")  # 添加 SELFIES 列
                                )
                                pass
                return res
        else:
            res = []
            file_path = osp.join(self.dir, self.split+'_re.txt')
            with open(file_path, 'r') as f:
                for i, line in enumerate(f):
                    if i==0: continue
                    line = line.split('\t')
                    assert len(line) == 4
                    smiles = line[1].strip()  # 提取 SMILES
                    desc = line[2].strip()  # 提取描述
                    selfies = line[3].replace('\n','')
                    if smiles != '*':
                        # 将 SMILES 转换为 SELFIES
                        try :
                            pre_selfies = sf.encoder(smiles)
                            assert pre_selfies == selfies
                            res.append(
                                (int(line[0]), smiles,  desc, selfies)  # 添加 SELFIES 列
                            )
                        except:
                            res.append(
                                (int(line[0]), smiles,  desc, "None")  # 添加 SELFIES 列
                            )
                            pass
            return res

            
    def __len__(self):
        return len(self.ori_data)
    def permute(self,smiles):
        p = random.random()
        if p<self.prob:
            print("PERMUTE SMILE")
            return changeorder(smiles,shuffle=True)
        else:
            return smiles

    def __getitem__(self,idx):
        if not self.is_join and not self.is_two_way:
            if not self.is_self:
                data = self.ori_data[idx]
                dic = {'cid':data[0],'smiles':self.permute(data[1]),'desc':data[2]}
                dic['tok_smiles'] = self.smi_tokenizer(dic['smiles'])
                dic['corrupted_toked_smis'] =  self.smi_tokenizer.corrupt(dic['smiles']) if random.random()<self.corrupt_prob else dic['tok_smiles']
                dic['tok_desc'] = None
                dic['dec_mask'] = None
                if self.load_state:
                    dic['desc_state'] = self.desc_state[data[0]]['states']
                    dic['desc_mask'] = self.desc_state[data[0]]['mask']
                    if self.mask_desc:
                        dic['desc_state'] = torch.zeros_like(dic['desc_state'])
                        dic['desc_mask'] = torch.ones_like(dic['desc_mask'])
                return dic
            else:
                data = self.ori_data[idx]
                dic = {'cid':data[0],'desc':data[2],'selfies':data[3]}
                # torch.Size([1, 256])
                # dic['tok_smiles'] = self.smi_tokenizer(dic['smiles'])
                # dic['corrupted_toked_smis'] =  self.smi_tokenizer.corrupt(dic['smiles']) if random.random()<self.corrupt_prob else dic['tok_smiles']
                dic['tok_selfies'] = self.self_tokenizer(dic['selfies'])
                dic['tok_desc'] = None
                dic['dec_mask'] = None
                if self.load_state:
                    dic['desc_state'] = self.desc_state[data[0]]['states']
                    dic['desc_mask'] = self.desc_state[data[0]]['mask']
                    if self.mask_desc:
                        dic['desc_state'] = torch.zeros_like(dic['desc_state'])
                        dic['desc_mask'] = torch.ones_like(dic['desc_mask'])
                return dic
        else:
            data = self.ori_data[idx]
            dic = {'cid':data[0],'smiles':data[1],'desc':data[2],'selfies':data[3]}
            # torch.Size([1, 256])
            dic['tok_smiles'] = self.smi_tokenizer(dic['smiles'])
            # dic['corrupted_toked_smis'] =  self.smi_tokenizer.corrupt(dic['smiles']) if random.random()<self.corrupt_prob else dic['tok_smiles']
            dic['tok_selfies'] = self.self_tokenizer(dic['selfies'])
            dic['tok_desc'] = None
            dic['dec_mask'] = None
            if self.load_state:
                dic['desc_state'] = self.desc_state[data[0]]['states']
                dic['desc_mask'] = self.desc_state[data[0]]['mask']
                if self.mask_desc:
                    dic['desc_state'] = torch.zeros_like(dic['desc_state'])
                    dic['desc_mask'] = torch.ones_like(dic['desc_mask'])
            return dic

def changeorder(smiles,shuffle):
    original_smiles = smiles # Replace with your original SMILES string
    # Convert the original SMILES string to an RDKit molecule object
    mol = Chem.MolFromSmiles(original_smiles)
    if mol is None:
        print("Wrong in original dataset")
    Chem.Kekulize(mol)
    # Get the atom indices in the molecule
    atom_indices = [atom.GetIdx() for atom in mol.GetAtoms()]
    # Reverse the order of the atom indices
    # print(atom_indices)
    # # random.shuffle(atom_indices)
    # # Create a new molecule with the reordered atoms
    # print(atom_indices)
    if shuffle:
        random.shuffle(atom_indices)
    reordered_mol = Chem.RenumberAtoms(mol, atom_indices)
    # if k:
    #     print(reordered_mol)
    # Generate the new SMILES string
    new_smiles = Chem.MolToSmiles(reordered_mol,kekuleSmiles=True)
    return new_smiles
