from RDPMSpecIdentifier.datastructures import RDPMSpecData
from dash_extensions.enrich import ServersideBackend
import time
import logging

logger = logging.getLogger(__name__)



class DisplayModeBackend(ServersideBackend):
    def __init__(self, json_file: str):
        self.items = {}
        with open(json_file, "r") as handle:
            json_string = handle.read()
        self.rdpmspec_data = RDPMSpecData.from_json(json_string)
        self.max_items = 3

    def get(self, key, ignore_expired=False) -> any:
        if len(key.split("_")) > 1:
            if key in self.items:
                return self.items[key][0]
            else:
                return None
        else:
            return self.rdpmspec_data

    def _delete_old_items(self):
        current_time = time.time()
        tmp = []
        for key, (_, timestamp) in self.items.items():
            if current_time - timestamp > 86400:
                del self.items[key]
            else:
                tmp.append((key, timestamp))
        if len(self.items) >= self.max_items:
            logger.warning("Not enough expired items. Deleting the 10 oldest items")
            tmp = sorted(tmp, key=lambda x: x[1], reverse=True)
            for key, timestamp in tmp[0:10]:
                del self.items[key]

    def set(self, key, value):
        if isinstance(value, RDPMSpecData):
            pass
        else:
            if value is not None:
                self.items[key] = (value, time.time())
            if len(self.items) > self.max_items:
                self._delete_old_items()

    def has(self, key):
        return key in self.items
