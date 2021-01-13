import logging

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import pandas as pd
from tqdm import tqdm

from astred import AlignedSentences, Sentence, Word
from astred.utils import load_nlp

logger = logging.getLogger("astred")
logger.setLevel("INFO")

NLP = {}


def get_parser(lang):
    if lang not in NLP:
        NLP[lang] = load_nlp(lang, tokenize_pretokenized=True, logging_level="ERROR")
    return NLP[lang]


@dataclass(eq=False, repr=False)
class MetricAdder:
    din: Union[Path, str]
    src_lang: str
    tgt_lang: str
    dout: Union[Path, str] = field(default=None)
    force_parsing: bool = field(default=False)
    st_df: pd.DataFrame = field(default=None, init=False)
    tt_df: pd.DataFrame = field(default=None, init=False)
    sg_df: pd.DataFrame = field(default=None, init=False)

    def __post_init__(self):
        self.din = Path(self.din)
        self.dout = Path(self.dout) if self.dout else None

        if not self.dout:
            logger.info("'dout' (output directory) not provided. Input files will be overwritten!")

        self.id_cols = {"src": "STid", "tgt": "TTid"}
        self.seg_cols = {"src": "STseg", "tgt": "TTseg"}
        self.token_cols = {"src": "SToken", "tgt": "TToken"}

        if self.force_parsing:
            logger.info("'force_parsing' enabled. Will parse segments regardless of information"
                        " that is already available.")

        self.process_dir()

    def process_dir(self):
        st_files = self.din.glob("*.st")
        files = [(f, f.with_suffix(".tt"), f.with_suffix(".sg")) for f in st_files]

        for f_tuple in tqdm(files, unit="file"):
            if any(not f.exists() or not f.is_file() for f in f_tuple):
                logger.warning(
                    f"A file name must have a '.st', '.tt', and '.sg' file. "
                    f" Not the case for {f_tuple[0].stem}, so skipping..."
                )

            self.process_file(*f_tuple)

    def create_alignments(self, group, side="src"):
        # We need to get the indices of the opposite (tgt for src, src for tgt)
        id_str = self.id_cols["tgt"] if side == "src" else self.id_cols["src"]

        def get_token_aligns(r):
            if side == "src":
                return [(int(r["Id"]) - 1, int(idx) - 1) for idx in r[id_str].split("+") if idx != "---"]
            else:
                return [(int(idx) - 1, int(r["Id"]) - 1) for idx in r[id_str].split("+") if idx != "---"]

        aligns = sorted(set([item for sublist in group.apply(get_token_aligns, axis=1) for item in sublist]))

        min_src, min_tgt = map(min, zip(*aligns))

        return [(src - min_src, tgt - min_tgt) for src, tgt in aligns]

    def create_sentence(self, df, lang, side):
        text_attr = self.token_cols[side]
        required_props = ("word_id", "deprel", text_attr, "head")

        def create_word(r):
            if not any(pd.isna(r[prop]) for prop in required_props):
                return Word(id=int(r["word_id"]), text=r[text_attr], head=int(r["head"]), deprel=r["deprel"])
            return None

        if not self.force_parsing and all(prop in df for prop in required_props):
            words = df.apply(lambda r: create_word(r), axis=1).tolist()

            if None not in words:
                return Sentence(words)

        # Fall-back to parsing if the required columns are not found
        return Sentence.from_text(
                " ".join(df[text_attr]), get_parser(lang)
            )

    def process_file(self, st, tt, sg):
        self.st_df = pd.read_csv(st, sep="\t", encoding="utf-8")
        self.tt_df = pd.read_csv(tt, sep="\t", encoding="utf-8")
        self.sg_df = pd.read_csv(sg, sep="\t", encoding="utf-8")

        def process_segment(row):
            row_idx = row.name
            sseg_id = row["STseg"]
            tseg_id = row["TTseg"]

            src_sent = self.create_sentence(self.st_df.query(f"{self.seg_cols['src']}==@sseg_id"),
                                            self.src_lang if self.src_lang else row["SL"],
                                            side="src")
            tgt_sent = self.create_sentence(self.tt_df.query(f"{self.seg_cols['tgt']}==@tseg_id"),
                                            self.tgt_lang if self.tgt_lang else row["TL"],
                                            side="tgt")

            src_aligns = self.create_alignments(self.st_df.query(f"{self.seg_cols['src']}==@sseg_id"))
            tgt_aligns = self.create_alignments(self.tt_df.query(f"{self.seg_cols['tgt']}==@tseg_id"), side="tgt")

            assert src_aligns == tgt_aligns

            aligned = AlignedSentences(src_sent, tgt_sent, src_aligns)

            self.sg_df.loc[row_idx, "alignments"] = " ".join(["-".join(map(str, pair)) for pair in src_aligns])
            self.sg_df.loc[row_idx, "word_cross"] = aligned.word_cross
            self.sg_df.loc[row_idx, "avg_word_cross"] = aligned.word_cross / len(aligned.no_null_word_pairs)
            self.sg_df.loc[row_idx, "seq_cross"] = aligned.seq_cross
            self.sg_df.loc[row_idx, "avg_seq_cross"] = aligned.seq_cross / len(aligned.no_null_seq_pairs)
            self.sg_df.loc[row_idx, "sacr_cross"] = aligned.sacr_cross
            self.sg_df.loc[row_idx, "avg_sacr_cross"] = aligned.sacr_cross / len(aligned.no_null_seq_pairs)
            self.sg_df.loc[row_idx, "astred"] = aligned.ted
            self.sg_df.loc[row_idx, "avg_astred"] = aligned.ted / (
                    (len(src_sent) - 1 + len(tgt_sent) - 1) / 2
            )  # -1 for NULL
            self.sg_df.loc[row_idx, "num_changes"] = aligned.num_changes
            self.sg_df.loc[row_idx, "avg_num_changes"] = aligned.num_changes / len(aligned.no_null_word_pairs)

            self.add_word_metrics(aligned, (sseg_id, tseg_id))

        self.sg_df.apply(process_segment, axis=1)

        try:
            self.st_df.to_csv(self.dout.joinpath(st.name) if self.dout else st, sep="\t", encoding="utf-8", index=False)
            self.tt_df.to_csv(self.dout.joinpath(tt.name) if self.dout else tt, sep="\t", encoding="utf-8", index=False)
            self.sg_df.to_csv(self.dout.joinpath(sg.name) if self.dout else sg, sep="\t", encoding="utf-8", index=False)
        except PermissionError as e:
            raise PermissionError("Could not write to file (see trace above) because you either do not have"
                                  " permissions to do so or because you have it opened in another program."
                                  " Close the file and try again.") from e

    def add_word_metrics(self, aligned, seg_ids):
        df = None
        token_attr = ""
        orig_start_idx = 0

        def set_col(idx, prop, word):
            if prop not in df or pd.isna(df.loc[idx, prop]):
                df.loc[idx, prop] = getattr(word, prop)

        def process_word(r):
            orig_idx = orig_start_idx + r.name
            word = sent[r.name+1]

            assert r[token_attr] == word.text == df.loc[orig_idx, token_attr]
            df.loc[orig_idx, "word_id"] = word.id

            for prop in ("deprel", "lemma", "head", "upos", "xpos", "feats"):
                set_col(orig_idx, prop, word)

            df.loc[orig_idx, "num_changes"] = word.num_changes()
            df.loc[orig_idx, "avg_num_changes"] = word.avg_num_changes()

            df.loc[orig_idx, "word_cross"] = word.cross
            df.loc[orig_idx, "avg_word_cross"] = word.avg_cross

            df.loc[orig_idx, "id_in_seq_group"] = word.id_in_seq_group
            df.loc[orig_idx, "seq_cross"] = word.seq_group.cross
            df.loc[orig_idx, "avg_seq_cross"] = word.seq_group.avg_cross

            df.loc[orig_idx, "id_in_sacr_group"] = word.id_in_sacr_group
            df.loc[orig_idx, "sacr_cross"] = word.sacr_group.cross
            df.loc[orig_idx, "avg_sacr_cross"] = word.sacr_group.avg_cross

            df.loc[orig_idx, "astred_cost"] = word.tree.astred_cost if word.tree else 0

        for side_idx, seg_id in enumerate(seg_ids):
            df = self.st_df if side_idx == 0 else self.tt_df
            token_attr = "SToken" if side_idx == 0 else "TToken"
            seg_attr = "STseg" if side_idx == 0 else "TTseg"
            sent = aligned.src if side_idx == 0 else aligned.tgt

            sub = df.query(f"{seg_attr}==@seg_id")
            orig_start_idx = sub.index[0]
            sub.reset_index(inplace=True)
            sub.apply(process_word, axis=1)


if __name__ == "__main__":
    import argparse

    cparser = argparse.ArgumentParser()
    cparser.add_argument("din", help="Input directory where TPRDB tables are saved")
    cparser.add_argument("--src_lang",
                         help="Source language. By default the 'SL' will be used for the parser's language."
                              " If such column is not present in your data, or you wish to force the source"
                              " language to another language, you can use this option. This must be a"
                              " short-hand code for the language, such as 'en' or 'de'.")
    cparser.add_argument("--tgt_lang",
                         help="Target language. By default the 'TL' will be used for the parser's language."
                              " See 'src_lang' for more info.")
    cparser.add_argument(
        "-d", "--dout", help="Optional output directory. IF NOT GIVEN YOUR FILES WILL CHANGE IN PLACE!"
    )
    cparser.add_argument(
        "-f", "--force_parsing", action="store_true",
        help="By default, segments will not be parsed if the head, word_id, deprel, and SToken or TToken of"
             " their corresponding words is set. The intuition being that one could first run the script and"
             " automatically parse the sentences, then manually correct the dependency labels (deprel) and the head"
             " index (head), and re-run this code to take into account those changes. Then, the sentences will not"
             " be automatically parsed, but the given (corrected) information is used to calculate the metrics."
             " If however, you wish to force parsing even if those values are set, you can enable this option."
    )
    instance = MetricAdder(**vars(cparser.parse_args()))
