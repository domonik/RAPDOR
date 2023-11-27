from RDPMSpecIdentifier.datastructures import RDPMSpecData
from dash_extensions.enrich import FileSystemBackend
import time
import logging

logger = logging.getLogger(__name__)


class DisplayModeBackend(FileSystemBackend):
    def __init__(self, json_file: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.items = {}
        with open(json_file, "r") as handle:
            json_string = handle.read()
        self.rdpmspec_data = RDPMSpecData.from_json(json_string)
        self.max_items = 3

    def get(self, key, ignore_expired=False) -> any:
        if len(key.split("_")) > 1:
            return super().get(key, ignore_expired)
        else:
            return self.rdpmspec_data

    def set(self, key, value, timeout=None,
            mgmt_element: bool = False, ):
        if isinstance(value, RDPMSpecData):
            pass
        else:
            super().set(key, value)

    def has(self, key):
        return super().has(key)
