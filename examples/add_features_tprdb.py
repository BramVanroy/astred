r"""Automatically adds syntactic metrics to TPR-DB tables by using the astred library by Bram Vanroy. Metrics are added both on the segment level (.sg tablles) and on the token level (.st and .tt).

WARNING: by default, the files are changed in-place. If you want to write the output to different files, use the -d option.

It is recommended that the SG tables are manually annotated with the following columns with TRUE or FALSE values:
  - more_than_one_tseg: indicating whether the translation consists of more than one sentence (one sentence translated as multiple sentences);
  - less_than_one_tseg: indicating whether the translation consists of less than one sentence (one sentence translated as multiple parts);
  - more_than_one_sseg: indicating whether the source segment consists of more than one sentence (multiple sentences translated as one sentence);

By default, the parser will also check how many sentences are in a segment. If more than two, the segment will not be processed. This behaviour can be changed with the --no_check_multiple flag.

Please cite our papers when you use this script. See https://github.com/BramVanroy/astred#citation
"""


from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Union

import pandas as pd
from astred import AlignedSentences, Sentence, Word
from astred.utils import load_parser
from tqdm import tqdm

logger = logging.getLogger("astred")
logger.setLevel("INFO")

NLP = {}


def get_parser(lang: str, is_tokenized: bool = True):
    idn = f"{lang}{is_tokenized}"
    if idn not in NLP:
        logger.info(f"(Down)loading parser for {lang} (with{'out' if is_tokenized else ''} tokenizer)...")
        NLP[idn] = load_parser(lang, is_tokenized=is_tokenized, logging_level="ERROR")
    return NLP[idn]


@dataclass(eq=False, repr=False)
class MetricAdder:
    din: Union[Path, str]
    src_lang: str
    tgt_lang: str
    dout: Union[Path, str] = field(default=None)
    force_parsing: bool = field(default=False)
    no_mwg: bool = field(default=False)
    no_replace_underscore: bool = field(default=False)
    no_check_multiple: bool = field(default=False)
    st_df: pd.DataFrame = field(default=None, init=False)
    tt_df: pd.DataFrame = field(default=None, init=False)
    sg_df: pd.DataFrame = field(default=None, init=False)

    def __post_init__(self):
        self.din = Path(self.din)
        self.dout = Path(self.dout) if self.dout else self.din

        self.dout.mkdir(exist_ok=True)

        if not self.dout:
            logger.info("'dout' (output directory) not provided. Input files will be overwritten!")

        self.id_cols = {"src": "STid", "tgt": "TTid"}
        self.seg_cols = {"src": "STseg", "tgt": "TTseg"}
        self.token_cols = {"src": "SToken", "tgt": "TToken"}

        if self.force_parsing:
            logger.info(
                "'force_parsing' enabled. Will parse segments regardless of information" " that is already available."
            )

        self.add_metrics()

    def add_metrics(self):
        logger.info("Calculating and adding metrics...")
        st_files = self.din.glob("*.st")
        files = [(f, f.with_suffix(".tt"), f.with_suffix(".sg")) for f in st_files]

        if files:
            for f_tuple in tqdm(files, unit="file"):
                if any(not f.exists() or not f.is_file() for f in f_tuple):
                    logger.warning(
                        "A file name must have a '.st', '.tt', and '.sg' file. "
                        f" Not the case for {f_tuple[0].stem}, so skipping..."
                    )
                    continue
                self.process_file(*f_tuple)
        else:
            raise ValueError(f"No suitable files found in {self.din.resolve()}")

    def create_alignments(self, group, min_src, min_tgt, side="src"):
        # We need to get the indices of the opposite (tgt for src, src for tgt)
        id_str = self.id_cols["tgt"] if side == "src" else self.id_cols["src"]

        def get_token_aligns(r):
            if side == "src":
                return [(int(r["Id"]), int(idx)) for idx in r[id_str].split("+") if idx != "---"]
            else:
                return [(int(idx), int(r["Id"])) for idx in r[id_str].split("+") if idx != "---"]

        aligns = sorted(set([item for sublist in group.apply(get_token_aligns, axis=1) for item in sublist]))

        return [(src - min_src, tgt - min_tgt) for src, tgt in aligns if src != -1 and tgt != -1]

    def create_sentence(self, df, lang, side):
        text_attr = self.token_cols[side]
        required_props = ("word_id", "deprel", text_attr, "head")

        def create_word(r):
            if not any(pd.isna(r[prop]) for prop in required_props):
                return Word(
                    id=int(r["word_id"]),
                    text=r[text_attr],
                    head=int(r["head"]),
                    deprel=r["deprel"],
                )
            return None

        if not self.force_parsing and all(prop in df for prop in required_props):
            words = df.apply(lambda r: create_word(r), axis=1).tolist()

            if None not in words:
                return Sentence(words)

        # Fall-back to parsing if the required columns are not found
        text = " ".join(df[text_attr]) if self.no_replace_underscore else " ".join(df[text_attr]).replace("_", "'")

        # Let the parser check how many sentences are present in the segment (requires tokenisation)
        if not self.no_check_multiple:
            doc = get_parser(lang, False)(text)

            if len(doc.sentences) > 1:
                return None

        return Sentence.from_text(text, get_parser(lang), on_multiple="none")

    def process_file(self, st, tt, sg):
        self.st_df = pd.read_csv(st, sep="\t", encoding="utf-8")
        self.tt_df = pd.read_csv(tt, sep="\t", encoding="utf-8")
        self.sg_df = pd.read_csv(sg, sep="\t", encoding="utf-8")

        def process_segment(row):
            row_idx = row.name
            sseg_id = row["STseg"]
            tseg_id = row["TTseg"]

            if "more_than_one_tseg" in row and row["more_than_one_tseg"]:
                logger.error(f"Source segment {row_idx+1} ({st.stem}) is translated as multiple sentences as"
                             " indicated by the 'more_than_one_tseg' column in the SG table. Skipping...")
                return False

            if "more_than_one_sseg" in row and row["more_than_one_sseg"]:
                logger.error(f"Source segment {row_idx+1} ({st.stem}) consists of multiple sentences as indicated by"
                             " the 'more_than_one_sseg' column in the SG table. Skipping...")
                return False

            if "less_than_one_tseg" in row and row["less_than_one_tseg"]:
                logger.error(f"Source segment {row_idx+1} ({st.stem}) is translated as a partial sentence as indicated"
                             " by the 'less_than_one_tseg' column in the SG table. Skipping...")
                return False

            try:
                src_sent = self.create_sentence(
                    self.st_df.query(f"{self.seg_cols['src']}==@sseg_id"),
                    self.src_lang if self.src_lang else row["SL"],
                    side="src",
                )
                tgt_sent = self.create_sentence(
                    self.tt_df.query(f"{self.seg_cols['tgt']}==@tseg_id"),
                    self.tgt_lang if self.tgt_lang else row["TL"],
                    side="tgt",
                )
            except KeyError:
                raise KeyError("--src_lang or --tgt_lang not given and no 'SL' or 'TL' column found. The parser does"
                               " not know which language to use. Please specify the language code with --src_lang"
                               " and --tgt_lang.")

            if src_sent is None:
                logger.error(f"Source segment {row_idx+1} ({st.stem}) consists of multiple sentences as per the parse"
                             " of the automatic parser. Skipping...")
                return False

            if tgt_sent is None:
                logger.error(f"Source segment {row_idx+1} ({st.stem}) is translated as multiple sentences as per the"
                             " parse of the automatic parser. Skipping...")
                return False

            min_src_id = self.st_df.query(f"{self.seg_cols['src']}==@sseg_id")["Id"].min()
            min_tgt_id = self.tt_df.query(f"{self.seg_cols['tgt']}==@tseg_id")["Id"].min()

            try:
                src_aligns = self.create_alignments(
                    self.st_df.query(f"{self.seg_cols['src']}==@sseg_id"),
                    min_src=min_src_id,
                    min_tgt=min_tgt_id,
                )
                tgt_aligns = self.create_alignments(
                    self.tt_df.query(f"{self.seg_cols['tgt']}==@tseg_id"),
                    min_src=min_src_id,
                    min_tgt=min_tgt_id,
                    side="tgt",
                )

                if src_aligns != tgt_aligns:
                    raise ValueError("Inconsistency in the data: the alignment from source to target is not the same"
                                     " as from target to source.")

                if not src_aligns:
                    raise ValueError
            except ValueError:
                logger.error(f"No alignments found for {st.stem}, source segment {row_idx}. Skipping...")
                return False

            aligned = AlignedSentences(src_sent, tgt_sent, src_aligns, allow_mwg=not self.no_mwg)

            self.sg_df.loc[row_idx, "alignments"] = " ".join(["-".join(map(str, pair)) for pair in src_aligns])
            self.sg_df.loc[row_idx, "word_cross"] = aligned.word_cross
            self.sg_df.loc[row_idx, "seq_cross"] = aligned.seq_cross
            self.sg_df.loc[row_idx, "sacr_cross"] = aligned.sacr_cross
            self.sg_df.loc[row_idx, "astred_changes"] = aligned.ted
            self.sg_df.loc[row_idx, "dep_changes"] = aligned.num_changes()
            self.sg_df.loc[row_idx, "pos_changes"] = aligned.num_changes("upos")

            self.add_word_metrics(aligned, (sseg_id, tseg_id))

            return True

        success = self.sg_df.apply(process_segment, axis=1)

        # Only write the file if at least one segment was processed successfully
        if any(s for s in success):
            try:
                self.st_df.to_csv(self.dout.joinpath(st.name), sep="\t", encoding="utf-8", index=False)
                self.tt_df.to_csv(self.dout.joinpath(tt.name), sep="\t", encoding="utf-8", index=False)
                self.sg_df.to_csv(self.dout.joinpath(sg.name), sep="\t", encoding="utf-8", index=False)
            except PermissionError as e:
                raise PermissionError(
                    "Could not write to file (see trace above) because you either do not have"
                    " permissions to do so or because you have it opened in another program."
                    " Close the file and try again."
                ) from e
        else:
            logger.error(
                f"{st.stem} only contained segments with errors (see above)." f" No properties added, so not saving."
            )

    def add_word_metrics(self, aligned, seg_ids):
        df = None
        orig_start_idx = 0

        def set_col(idx, prop, word):
            if prop not in df or pd.isna(df.loc[idx, prop]):
                df.loc[idx, prop] = getattr(word, prop)

        def process_word(r):
            orig_idx = orig_start_idx + r.name
            word = sent[r.name + 1]

            df.loc[orig_idx, "word_id"] = word.id

            for prop in ("deprel", "head", "upos"):
                set_col(orig_idx, prop, word)

            # Don't do bool(word.aligned) because .aligned contains NULL for null alignments
            df.loc[orig_idx, "is_aligned"] = bool(word.aligned_cross)

            num_dep_changes = word.num_changes()
            # dep+change is True for unaligned words
            df.loc[orig_idx, "dep_change"] = num_dep_changes > 0 if num_dep_changes is not None else True
            df.loc[orig_idx, "num_dep_changes"] = num_dep_changes if num_dep_changes is not None else 0

            num_pos_changes = word.num_changes("upos")
            # pos+change is True for unaligned words
            df.loc[orig_idx, "pos_change"] = num_pos_changes > 0 if num_pos_changes is not None else True
            df.loc[orig_idx, "num_pos_changes"] = num_pos_changes if num_pos_changes is not None else 0

            # crosses are 0 for unaligned words/word groups
            df.loc[orig_idx, "word_cross"] = word.cross if word.cross is not None else 0
            df.loc[orig_idx, "seq_cross"] = word.seq_group.cross if word.seq_group.cross is not None else 0
            df.loc[orig_idx, "sacr_cross"] = (
                word.sacr_group.cross if word.sacr_group.cross is not None else 0
            )

            df.loc[orig_idx, "astred_change"] = word.tree.astred_cost > 0
            df.loc[orig_idx, "astred_op"] = word.tree.astred_op

            if not self.no_mwg:
                df.loc[orig_idx, "is_part_of_mwg"] = word.seq_group.is_mwg

        for side_idx, seg_id in enumerate(seg_ids):
            df = self.st_df if side_idx == 0 else self.tt_df
            seg_attr = "STseg" if side_idx == 0 else "TTseg"
            sent = aligned.src if side_idx == 0 else aligned.tgt

            sub = df.query(f"{seg_attr}==@seg_id")
            orig_start_idx = sub.index[0]
            sub.reset_index(inplace=True)
            sub.apply(process_word, axis=1)


if __name__ == "__main__":
    import argparse

    cparser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cparser.add_argument("din", help="Input directory where TPRDB tables are saved (.sg, .st, .tt)")
    cparser.add_argument(
        "--src_lang",
        help="Source language. By default the 'SL' will be used for the parser's language."
        " If such column is not present in your data, or you wish to force the source"
        " language to another language, you can use this option. This must be a"
        " short-hand code for the language, such as 'en' or 'de'. You can find available models and language codes"
        " here: https://stanfordnlp.github.io/stanza/available_models.html",
    )
    cparser.add_argument(
        "--tgt_lang",
        help="Target language. By default the 'TL' will be used for the parser's language."
        " See 'src_lang' for more info.",
    )
    cparser.add_argument(
        "-d",
        "--dout",
        help="Optional output directory. IF NOT GIVEN YOUR FILES WILL CHANGE IN PLACE!",
    )
    cparser.add_argument(
        "-f",
        "--force_parsing",
        action="store_true",
        help="By default, segments will not be parsed if the head, word_id, deprel, and SToken or TToken of"
        " their corresponding words is set. The intuition being that one could first run the script and"
        " automatically parse the sentences, then manually correct the dependency labels (deprel) and the head"
        " index (head), and re-run this code to take into account those changes. Then, the sentences will not"
        " be automatically parsed, but the given (corrected) information is used to calculate the metrics."
        " If however, you wish to force parsing even if those values are set, you can enable this option.",
    )
    cparser.add_argument(
        "--no_mwg",
        help="By default, multi-word groups (MWG) are considered as one sequence group (and are later refined"
        " with SACr to ensure linguistically consistent groups). A MWG, in our case, is defined as a group in"
        " which all source words are aligned with all target words and which contains more than one source and"
        " more than one target words. (One-to-many/many-to-one alignments are a group by default.) You can"
        " disallow the creation of MWG with this flag. Disallowing will likely lead to much higher"
        " seq_cross and sacr_cross values.",
        default=False,
        action="store_true",
    )
    cparser.add_argument(
        "--no_replace_underscore",
        help="In some versions of the TPR-DB, a single quote is replaced by an underscore. This will lead to bad"
        " parses (e.g. possessive s). By default, we replace all underscores with a single quote. Use this option"
        " to disable that behaviour",
        default=False,
        action="store_true",
    )
    cparser.add_argument(
        "--no_check_multiple",
        help="By default, the script will NOT process segments that contain multiple sentences. The reason is that"
             " dependency trees are specific to single sentences. The parser will try to check how many sentences are"
             " present in a segment. If more than one, the segment will not be processed. Alternatively, specific"
             " manually created columns can also be used. See the script description for more. If you do not want the"
             " parser to automatically check for the number of sentences, then you should use this option. When"
             " enabled, all segments will be processed except for those where specific columns are set to TRUE, as "
             " described in the script description.",
        default=False,
        action="store_true",
    )
    instance = MetricAdder(**vars(cparser.parse_args()))
