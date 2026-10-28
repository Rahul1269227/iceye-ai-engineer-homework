"""ICEYE AI Engineer homework: NER chatbot (Part 1) and rotated blob boxes (Part 2)."""

import warnings

# Import-time noise from pinned third-party code: glinker 0.1.1 declares Pydantic v1-style
# models, and GLiNER compiles a helper with the deprecated torch.jit.script.
for _message in (
    "Valid config keys have changed in V2",
    'Field name "schema" in "PipeNode" shadows an attribute',
    "`torch.jit.script` is deprecated",
):
    warnings.filterwarnings("ignore", message=_message)
