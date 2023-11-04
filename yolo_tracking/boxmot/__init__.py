# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

__version__ = '10.0.43'

from yolo_tracking.boxmot.postprocessing.gsi import gsi
from yolo_tracking.boxmot.tracker_zoo import create_tracker, get_tracker_config
from yolo_tracking.boxmot.trackers.botsort.bot_sort import BoTSORT
from yolo_tracking.boxmot.trackers.bytetrack.byte_tracker import BYTETracker
from yolo_tracking.boxmot.trackers.deepocsort.deep_ocsort import DeepOCSort as DeepOCSORT
from yolo_tracking.boxmot.trackers.hybridsort.hybridsort import HybridSORT
from yolo_tracking.boxmot.trackers.ocsort.ocsort import OCSort as OCSORT
from yolo_tracking.boxmot.trackers.strongsort.strong_sort import StrongSORT

TRACKERS = ['bytetrack', 'botsort', 'strongsort', 'ocsort', 'deepocsort', 'hybridsort']

__all__ = ("__version__",
           "StrongSORT", "OCSORT", "BYTETracker", "BoTSORT", "DeepOCSORT", "HybridSORT",
           "create_tracker", "get_tracker_config", "gsi")
