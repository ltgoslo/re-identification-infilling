import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel


def maxsim(queries: torch.Tensor, documents: torch.Tensor) -> torch.Tensor:
    return torch.matmul(queries.unsqueeze(1), documents.transpose(-1, -2)).max(dim=-1).values.sum(dim=-1)


def filter(documents, token_ids, filter_list):
    return documents


class QueryEncoder(nn.Module):

    def __init__(self, encoder_name, hidden_size, compress_size, query_tokenizer=None):

        super().__init__()

        self.encoder = AutoModel.from_pretrained(encoder_name)
        if query_tokenizer is not None:
            self.encoder.resize_token_embeddings(len(query_tokenizer))
        self.downsampler = nn.Linear(hidden_size, compress_size)

    def forward(self, input_ids, attention_mask):

        hidden_states = self.encoder(input_ids, attention_mask).last_hidden_state
        queries = self.downsampler(hidden_states)
        queries = F.normalize(queries, dim=-1)

        return queries


class DocumentEncoder(nn.Module):

    def __init__(self, encoder_name, hidden_size, compress_size, doc_tokenizer=None, filter_list=None):

        super().__init__()

        self.filter_list = filter_list

        self.encoder = AutoModel.from_pretrained(encoder_name)
        if doc_tokenizer is not None:
            self.encoder.resize_token_embeddings(len(doc_tokenizer))
        self.downsampler = nn.Linear(hidden_size, compress_size)

    def forward(self, input_ids, attention_mask):

        hidden_states = self.encoder(input_ids, attention_mask).last_hidden_state
        documents = self.downsampler(hidden_states)
        documents = F.normalize(documents, dim=-1)
        documents = filter(documents, input_ids, self.filter_list)

        return documents


class ColBERT(nn.Module):

    def __init__(self, encoder_name, hidden_size, compress_size, query_tokenizer=None, doc_tokenizer=None, filter_list=None):

        super().__init__()

        self.query_encoder = AutoModel.from_pretrained(encoder_name)
        if doc_tokenizer is not None:
            self.query_encoder.resize_token_embeddings(len(query_tokenizer))
            self.query_pad_index = query_tokenizer.pad_token_id

        self.document_encoder = AutoModel.from_pretrained(encoder_name)
        if doc_tokenizer is not None:
            self.document_encoder.resize_token_embeddings(len(doc_tokenizer))
            self.doc_pad_index = doc_tokenizer.pad_token_id

        self.downsampler = nn.Linear(hidden_size, compress_size)

    def forward(self, query_ids, query_mask, document_ids, document_mask):

        queries = self.query_encoder(query_ids, query_mask).last_hidden_state
        queries = self.downsampler(queries)
        queries = F.normalize(queries, dim=-1)

        documents = self.document_encoder(document_ids, document_mask).last_hidden_state
        documents = self.downsampler(documents)

        mask = torch.tensor(self.mask(document_ids, self.doc_pad_index), device=document_ids.device).unsqueeze(-1).float()
        documents = documents * mask

        documents = F.normalize(documents, dim=-1)

        documents = documents.unsqueeze(1)
        pos_documents, neg_documents = torch.chunk(documents, 2, dim=0)
        documents = torch.cat([pos_documents, neg_documents], dim=1)

        return maxsim(queries, documents)

    def mask(self, input_ids, pad_id):
        mask = [[(x != pad_id) for x in d] for d in input_ids.cpu().tolist()]

        return mask
