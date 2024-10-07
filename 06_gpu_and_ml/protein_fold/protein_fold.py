# ---
# output-directory: "/tmp/protein-fold"
# ---

# Acronyms
## AA  - Amino Acid
## SSE - Secondary Structure
# Background
## Atom14 refers to a reduced representation of 14 atoms per Amino Acid. Alpha
## Fold and ESMFold seem to run in this representation space for efficiency.
## Atom37 refers to an expanded representation that is computed from Atom14 and
## more closely resembles PDB structure.

# TODO
# - Do we need PDB & PDBX files? Kind of silly but py3DMol doesn't seem to like
# PDB files.
# - Fix errors in some crystallized plot, missing ')' in py3DMol html.
# - Error handling for bad sequence. E.g. non amino acid letter.
# - Ask Charles about masking, k-mers, ...

from fastapi import FastAPI
from fastapi.responses import FileResponse
import modal
from pathlib import Path
import time

esm3_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "esm==3.0.5",
        "torch==2.4.1",
    )
)

web_app_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "esm==3.0.5",
        "gradio~=4.44.0",
        "biotite==0.41.2",
        "pydssp==0.9.0",
        "py3Dmol==2.4.0",
        "torch==2.4.1",
    )
)

app = modal.App("protein_fold")

volume = modal.Volume.from_name(
    "example-protein-fold", create_if_missing=True
)
volume_path = Path("/vol/data")
PDBS_PATH = volume_path / "pdbs"

with esm3_image.imports():
    from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig
    import torch

with web_app_image.imports():
    from py3DmolWrapper import py3DMolViewWrapper

SECONDS_PER_MINUTE = 60
@app.cls(
    gpu='A10G',
    image=esm3_image,
    timeout=20 * SECONDS_PER_MINUTE,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class ModelInference:
    @modal.build()
    @modal.enter()
    def build(self):
        from esm.models.esm3 import ESM3
        self.model: ESM3InferenceClient = ESM3.from_pretrained("esm3_sm_open_v1").to("cuda")
        # Optimizations from:
        # https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/protein_folding.ipynb#scrollTo=cc0f0186
        self.model = self.model.half()
        torch.backends.cuda.matmul.allow_tf32 = True

        self.max_steps = 250

    @modal.method()
    def generate(self, sequence: str) -> bool:
        start_time = time.monotonic()
        # num_steps cannot be longer than the seq length in ESM.
        structure_generation_config = GenerationConfig(
            track="structure",
            num_steps=min(len(sequence), self.max_steps)
            )
        esm_protein = self.model.generate(
            ESMProtein(sequence=sequence),
            structure_generation_config
            )
        latency_s = time.monotonic() - start_time
        print (f"Latency {latency_s:.2f} seconds.")

        # Check that esm_protein did not error.
        if hasattr(esm_protein, "error_msg"):
            raise ValueError(esm_protein.error_msg)

        esm_protein.ptm = esm_protein.ptm.to('cpu') # Move off GPU
        return esm_protein

web_app = FastAPI()
assets_path = Path(__file__).parent / "assets"
@app.function(
    image=web_app_image,
    concurrency_limit=1,
    allow_concurrent_inputs=1000,
    volumes={volume_path: volume},
    mounts=[modal.Mount.from_local_dir(assets_path, remote_path="/assets")],
)
@modal.asgi_app()
def protein_fold_fastapi_app():
    import gradio as gr
    from gradio.routes import mount_gradio_app

    import biotite.database.rcsb as rcsb
    import biotite.structure as b_structure
    import biotite.structure.io.pdbx as pdbx
    from biotite.structure.io.pdbx import CIFFile
    import biotite.application.blast as blast
    from esm.sdk.api import ESMProtein

    import os

    width, height = 400, 400

    def fetch_pdb_if_necessary(pdb_id, pdb_type):
        # Only fetch from server if it's not on the volume already.
        assert pdb_type in ("pdb", "pdbx")
        file_path = PDBS_PATH / f"{pdb_id}.{pdb_type}"
        if not os.path.exists(file_path):
            print (f"Loading PDB {file_path} from server...")
            rcsb.fetch(pdb_id, pdb_type, str(PDBS_PATH))
        return file_path

    def get_sequence(pdb_id):
        try:
            pdb_id = pdb_id.strip() # Remove whitespace
            pdbx_file_path = fetch_pdb_if_necessary(pdb_id, "pdbx")

            structure = pdbx.get_structure(CIFFile.read(pdbx_file_path), model=1)

            # TODO Excluding atoms that are not in a Amino Acid?
            (amino_sequences, _) = (
                b_structure.to_sequence(
                    structure[b_structure.filter_amino_acids(structure)]))
            # Note: len(amino_sequences) == # of chains

            # TODO Should I put something between chains?
            sequence = "".join([str(s) for s in amino_sequences])
            return sequence

        except Exception as e:
            return f"Error: {e}"

    def find_pdb_id_from_sequence(query_sequence, maybe_stale_pdb_id):
        pdb_id_sequence = get_sequence(maybe_stale_pdb_id)

        if pdb_id_sequence == query_sequence:
            print ("Skipped BLAST run, PDB ID already known.")
            not_stale_pdb_id = maybe_stale_pdb_id
            return not_stale_pdb_id
        else:
            # TODO Use Blast or some kind of Sequence Database
            # Skipping Blast which is rate limited for now.
            return None

        print ("Running blast to find PDB ID of sequence...")
        blast_app = blast.BlastWebApp(
            "blastp", query_sequence, database="pdb")
        blast_app.start()
        blast_app.join()

        alignments = blast_app.get_alignments()
        if len(alignments) == 0:
            return None
        return alignments[0].hit_id

    def build_database_html(sequence, maybe_stale_pdb_id):
        pdb_id = find_pdb_id_from_sequence(sequence, maybe_stale_pdb_id)
        if pdb_id is None:
            return "<h3>Folding Structure of Sequence not found.</h3>"

        # Remove chain information if present.
        if pdb_id.find("_") != -1: # "1CRN_A" -> "1CRN"
            pdb_id = pdb_id[:pdb_id.index("_")]

        # Extract secondary structure from PDBX file.
        pdbx_file_path = fetch_pdb_if_necessary(pdb_id, "pdbx")
        structure = pdbx.get_structure(CIFFile.read(pdbx_file_path), model=1)
        # TODO Should I only use AA atoms?
        atoms = structure[
            b_structure.filter_amino_acids(structure)]
        residue_secondary_structures = b_structure.annotate_sse(atoms)

        # Extract PDB string from PDB file.
        pdb_file_path = fetch_pdb_if_necessary(pdb_id, "pdb")
        pdb_string = Path(pdb_file_path).read_text()

        return py3DMolViewWrapper().build_html_with_secondary_structure(
            width, height, pdb_string, residue_secondary_structures)

    def postprocess_html(html):
        html = html.replace("'", '"') # ' -> " for HTML compatibility.

        html_wrapped =  f"""<!DOCTYPE html><html>{html}</html>"""
        iframe_html =  (f"""<iframe style="width: 100%; height: 400px;" """
            f"""allow="midi; display-capture;" frameborder="0" """
            f"""srcdoc='{html_wrapped}'></iframe>""")
        return iframe_html

    def run_esm_and_graph(sequence, maybe_stale_pdb_id):
        # Data cleaning
        sequence = sequence.strip() # Remove whitespace
        maybe_stale_pdb_id = maybe_stale_pdb_id.strip() # Remove whitespace

        # Run ESM
        esm_protein: ESMProtein = ModelInference().generate.remote(sequence)
        residue_pLDDTs = (100 * esm_protein.plddt).tolist()
        atoms = esm_protein.to_protein_chain().atom_array
        residue_secondary_structures = b_structure.annotate_sse(atoms)

        esm_sse_html = (
            py3DMolViewWrapper().build_html_with_secondary_structure(
            width, height, esm_protein.to_pdb_string(),
            residue_secondary_structures))

        esm_pLDDT_html = (
            py3DMolViewWrapper().build_html_with_pLDDTs(
            width, height, esm_protein.to_pdb_string(), residue_pLDDTs))

        database_sse_html = build_database_html(sequence, maybe_stale_pdb_id)

        return [postprocess_html(h)
                    for h in (esm_sse_html, esm_pLDDT_html, database_sse_html)]

    # custom styles: an icon, a background, and a theme
    @web_app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        return FileResponse("/assets/favicon.svg")

    @web_app.get("/assets/background.svg", include_in_schema=False)
    async def background():
        return FileResponse("/assets/background.svg")

    with open("/assets/index.css") as f:
        css = f.read()

    theme = gr.themes.Default(
        primary_hue="green", secondary_hue="emerald", neutral_hue="neutral"
    )

    with gr.Blocks(
        theme=theme, css=css, title="ESM3 Protein Folding"
    ) as interface:
        # Title
        gr.Markdown("# Fold Proteins using ESM3 Fold")

        # PDB ID text box + Get sequence from it button
        pdb_id = gr.Textbox(
            label="Enter PDB ID",
            placeholder="e.g. '5JQ4', '1MBO', '1TPO', etc.")
        get_sequence_button = gr.Button("Retrieve Sequence from PDB ID")

        # Sequence text box + Run ESM from it button
        sequence = gr.Textbox(
            label="Enter a sequence or retrieve it from a PDB ID",
            placeholder="e.g. 'MVTRLE...', 'GKQEG...', etc.")
        run_esm_and_graph_button = gr.Button("Run ESM3 Fold on Sequence")

        # 3 Columns of Protein Folding
        htmls = []
        with gr.Row():
            # Column 1: ESM Prediction with SSE coloring.
            with gr.Column():
                gr.Markdown("## ESM3 Prediction - Secondary Structs")
                gr.Image(  # output image component
                    height=100, width=400,
                    value="/assets/secondaryStructureLegend.png",
                    show_download_button=False, show_label=False,
                    show_fullscreen_button=False,
                )
                htmls.append(gr.HTML())

            # Column 2: ESM Prediction with Confidence coloring.
            with gr.Column():
                gr.Markdown("## ESM3 Prediction - PLTT Confidence")
                gr.Image(  # output image component
                    height=100, width=400, value="/assets/plddtLegend2.png",
                    show_download_button=False, show_label=False,
                    show_fullscreen_button=False,
                )
                htmls.append(gr.HTML())

            # Column 3: Database showing Crystalized form of Protein if avail.
            with gr.Column():
                gr.Markdown("## Crystalized Structure from PDB")
                gr.Image(  # output image component
                    height=100, width=400,
                    value="/assets/secondaryStructureLegend.png",
                    show_download_button=False, show_label=False,
                    show_fullscreen_button=False,
                )
                htmls.append(gr.HTML())

        get_sequence_button.click(
            fn=get_sequence, inputs=[pdb_id], outputs=[sequence])
        run_esm_and_graph_button.click(fn=run_esm_and_graph,
            inputs=[sequence, pdb_id], outputs=htmls)

    # mount for execution on Modal
    return mount_gradio_app(
        app=web_app,
        blocks=interface,
        path="/",
    )

@app.local_entrypoint()
def main():
    # YFP?
    sequence = (
        "AMFSKVNNQKMLEDCFYIRKKVFVEEQGIPEESEIDEYESESIHLIGYDNGQPVATARIRPINETTVKIERVAVMKSHRGQGMGRMLMQAVESLAKDEGFYVATMNAQCHAIPFYESLNFKMRGNIFLEEGIEHIEMTKKLT")
    for i in range(10):
        ModelInference().generate.remote(sequence)
        x = 1

if __name__ == "__main__":
    main()
