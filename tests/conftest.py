"""Global test environment: never download a production model during pytest."""

import os


os.environ.setdefault("MODEL_PRELOAD_ENABLED", "false")
os.environ.setdefault("MODEL_SOURCE", "modelscope")
