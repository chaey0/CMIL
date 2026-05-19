import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

class KEEPEncoder(nn.Module):
    def __init__(self, model_name: str = "Astaxanthin/KEEP"):
        # Astaxanthin/KEEP
        # microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        
        self.out_dim = 768 
        for param in self.model.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def encode_texts(self, texts: list, device=None) -> torch.Tensor:
        self.model.to(device)
        self.model.eval()
        tokens = self.tokenizer(texts, max_length=512, padding='max_length', 
                                truncation=True, return_tensors='pt').to(device)
        
        text_features = self.model.encode_text(tokens)

        return F.normalize(text_features, dim=-1)
    '''
    @torch.no_grad()
    def encode_texts(self, texts: list, device=None) -> torch.Tensor:
        self.model.to(device)
        self.model.eval()
        tokens = self.tokenizer(texts, max_length=512, padding='max_length', 
                                truncation=True, return_tensors='pt').to(device)

        outputs = self.model(**tokens)   # BertModel forward
        last_hidden = outputs.last_hidden_state   # [B, L, D]
        attention_mask = tokens["attention_mask"].unsqueeze(-1)  # [B, L, 1]

        # mean pooling
        summed = (last_hidden * attention_mask).sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1)
        text_features = summed / counts

        return F.normalize(text_features, dim=-1)
    '''

'''
# TITANEncoder
class KEEPEncoder(nn.Module):
    def __init__(self, model_name: str = "MahmoodLab/TITAN"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        self.out_dim = 768
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode_texts(self, texts: list, device=None) -> torch.Tensor:
        self.model.to(device)
        self.model.eval()

        tokens = self.tokenizer(
            texts,
            max_length=128,
            padding=True,       
            truncation=True,
            return_tensors="pt",
        )

        input_ids = tokens["input_ids"].to(device)

        text_features = self.model.encode_text(input_ids, normalize=False)
        text_features = F.normalize(text_features, dim=-1)
        return text_features

'''