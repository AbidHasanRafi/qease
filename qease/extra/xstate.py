"""Run-time state of the extra experiments (configuration and the uid -> record index of the current model)."""
import copy

from qease.extra.config import DEFAULT_XCFG

XCFG = {}
UID2REC = {}


def configure_extra(overrides=None):
    """Build the configuration of the extra experiments (DEFAULT_XCFG plus overrides)."""
    global XCFG, UID2REC
    XCFG = copy.deepcopy(DEFAULT_XCFG)
    XCFG.update(overrides or {})
    UID2REC = {}
    return XCFG
