import itertools
import operator
from dataclasses import dataclass, field

import torch

try:
    from awesome_align.configuration_bert import BertConfig
    from awesome_align.modeling import BertForMaskedLM
    from awesome_align.tokenization_bert import BertTokenizer

    awesome_align_available = True
except (ImportError, AttributeError):
    awesome_align_available = False


@dataclass
class Aligner:
    model_name_or_path: str = field(default="bert-base-multilingual-cased")
    extraction: str = field(default="softmax")
    no_cuda: bool = False
    softmax_threshold: float = field(default=0.001)

    def __post_init__(self):
        if not awesome_align_available:
            raise ImportError("To use the automatic aligner, awesone_align must be installed.")

        self.device = torch.device("cuda" if torch.cuda.is_available() and not self.no_cuda else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name_or_path)
        self.config = BertConfig.from_pretrained(self.model_name_or_path)
        self.model = BertForMaskedLM.from_pretrained(self.model_name_or_path,
                                                     self.tokenizer.cls_token_id,
                                                     self.tokenizer.sep_token_id,
                                                     config=self.config)
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, src_sentence, tgt_sentence):
        sents_d = {"src": {"sent": src_sentence}, "tgt": {"sent": tgt_sentence}}

        for srctgt in list(sents_d.keys()):
            sent = sents_d[srctgt]["sent"].strip().split()
            tokens = [self.tokenizer.tokenize(word) for word in sent]
            wids = [self.tokenizer.convert_tokens_to_ids(token) for token in tokens]
            sents_d[srctgt]["ids"] = self.tokenizer.prepare_for_model(list(itertools.chain(*wids)),
                                                                      return_tensors="pt",
                                                                      max_length=self.tokenizer.max_len,
                                                                      return_token_type_ids=False,
                                                                      return_attention_mask=False)["input_ids"]

            sents_d[srctgt]["bpe2word_map"] = [i for i, word_list in enumerate(tokens) for _ in word_list]

        return sents_d["src"]["ids"][0], sents_d["tgt"]["ids"][0], sents_d["src"]["bpe2word_map"], sents_d["tgt"][
            "bpe2word_map"]

    def align(self, src_sentence, tgt_sentence):
        with torch.no_grad():
            ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt = self.preprocess(src_sentence, tgt_sentence)
            # Make it work for batch of one
            inputs = [ids_src.unsqueeze(0), ids_tgt.unsqueeze(0), (bpe2word_map_src,), (bpe2word_map_tgt,)]
            word_aligns = self.model.get_aligned_word(*inputs,
                                                      self.device, 0, 0, align_layer=8,
                                                      extraction=self.extraction,
                                                      softmax_threshold=self.softmax_threshold, test=True)
            aligns = list(word_aligns[0])
            aligns.sort(key=operator.itemgetter(0, 1))

        return aligns

    def align_from_objs(self, src_sentence, tgt_sentence):
        src_sentence = " ".join([w.text for w in src_sentence.no_null_words])
        tgt_sentence = " ".join([w.text for w in tgt_sentence.no_null_words])

        return self.align(src_sentence, tgt_sentence)
