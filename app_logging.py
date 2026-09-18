"""Small structured stderr logger for local application operations."""
import json
import logging
import sys
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "event": getattr(record, "event", "application_event"),
            "message": record.getMessage(),
        }
        for key in ("operation", "endpoint", "symbol", "sec_type", "con_id", "exchange", "mode", "rows", "contract_count", "snapshot", "host", "port", "client_id", "what_to_show", "end_date_time", "duration", "bar_size", "use_rth", "format_date", "error_type", "error_detail"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = str(value)
        return json.dumps(payload, ensure_ascii=True)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def log(logger, level, event, message, **fields):
    logger.log(level, message, extra={"event": event, **fields})
