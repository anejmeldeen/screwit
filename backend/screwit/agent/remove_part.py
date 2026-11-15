"""The remove_part tool: delete a part of the object when the user asks (SPEC 6.3 tool layer).

Trust boundary: part_id comes from the model and must name an existing part of this scan; nothing else
is accepted. The delete itself is screwit.app.part_delete.delete_part, the same code the inspector's
Delete part button runs, so the user can bring the part back with Undo. Object-agnostic.
"""

from __future__ import annotations

import logging
import re

from screwit.app.part_delete import PartDeleteRefused, delete_part
from screwit.app.registry import Registry
from screwit.app.store import ScanStore

log = logging.getLogger(__name__)

_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_LISTED_IDS = 30


def tool_spec() -> dict:
    """The remove_part tool in Anthropic tool form."""
    return {
        "name": "remove_part",
        "description": (
            "Delete one part from the workspace, either a scanned part or one added in chat. It "
            "disappears from the parts list, the hierarchy, the scene graph and the 3D viewer, and the "
            "user can bring it back with Undo in the inspector. Use it only for a part the user asked "
            "you to remove, or one you just added by mistake."),
        "input_schema": {
            "type": "object",
            "properties": {
                "part_id": {"type": "string",
                            "description": "Id of the part to delete, as listed by list_parts."},
            },
            "required": ["part_id"],
        },
    }


def _bad(message: str) -> dict:
    return {"error": "bad_request", "message": message}


def remove_part(reg: Registry, store: ScanStore, scan_id: str, args: dict) -> dict:
    if not isinstance(args, dict):
        return _bad("remove_part needs an object of arguments")
    unknown = sorted(set(args) - {"part_id"})
    if unknown:
        return _bad(f"unknown argument(s) {unknown}; remove_part takes only part_id")
    part_id = args.get("part_id")
    if not isinstance(part_id, str) or not _ID_RE.match(part_id):
        return _bad("part_id must be the id of an existing part (letters, digits, '_', '-' or '.')")
    ids = [p.part_id for p in store.list_parts(scan_id)]
    if part_id not in ids:
        known = ", ".join(ids[:_LISTED_IDS]) or "none"
        return {"error": "unknown_part", "message": f"{part_id!r} is not a part of this object; parts: {known}"}
    try:
        part = delete_part(reg, scan_id, part_id)
    except KeyError:
        return {"error": "unknown_part", "message": f"{part_id!r} is no longer a part of this object"}
    except PartDeleteRefused as e:
        return {"error": "refused", "message": str(e)}
    except Exception as e:  # noqa: BLE001 - report the failure to the model instead of ending the reply
        log.exception("remove_part %s on %s failed", part_id, scan_id)
        return {"error": "failed",
                "message": f"the delete did not complete ({type(e).__name__}); check list_parts before retrying"}
    return {"status": "removed", "part_id": part.part_id, "label": part.label,
            "message": ("Deleted. It is gone from the parts list, the hierarchy and the 3D viewer; the "
                        "user can bring it back with Undo in the inspector.")}
