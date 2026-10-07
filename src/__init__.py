"""Customer-support RAG pipeline."""
import os

# transformers imports TensorFlow whenever it is installed, which is slow and only adds
# warnings; this project uses PyTorch. Must be set before sentence-transformers is imported.
os.environ.setdefault("USE_TF", "0")
# Windows without Developer Mode cannot symlink the Hugging Face cache; that is harmless here.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
