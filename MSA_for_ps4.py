import pandas as pd
from MSA_batcher import BlastMSABatcher

ps4_df = pd.read_csv("dataset/data.csv")

batcher = BlastMSABatcher(
    default_db="swissprot",        # or "nr"
    email="you@example.com",       # strongly recommended by NCBI
    tool_name="ps4_pssm_pipeline", # arbitrary identifier
    max_hits=100,
    poll_interval=60,
    per_query_delay=3.0,
)

df_with_msa = batcher.run_on_dataframe(
    ps4_df,
    seq_col="sequence",        # column with amino acid sequences
    id_col="protein_id",       # optional; used in filenames
    db_col=None,               # or name of a column with per-row db code
    out_dir="msa_ps4",
)

df_with_msa.to_csv("ps4_with_msa_paths.csv", index=False)