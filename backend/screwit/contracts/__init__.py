from .models import *  # noqa: F403
from .models import Contract

# Top-level documents exported as JSON Schema for the web app's generated TypeScript types.
SCHEMA_ROOTS: dict[str, type[Contract]] = {
    "Scan": Scan,  # noqa: F405
    "Reconstruction": Reconstruction,  # noqa: F405
    "Cameras": Cameras,  # noqa: F405
    "Part": Part,  # noqa: F405
    "SceneGraph": SceneGraph,  # noqa: F405
    "RepairPlan": RepairPlan,  # noqa: F405
    "CadRequest": CadRequest,  # noqa: F405
    "CadResult": CadResult,  # noqa: F405
    "ChatRequest": ChatRequest,  # noqa: F405
    "ChatMessage": ChatMessage,  # noqa: F405
    "CreateScanRequest": CreateScanRequest,  # noqa: F405
    "CreateScanResponse": CreateScanResponse,  # noqa: F405
    "ScaleRequest": ScaleRequest,  # noqa: F405
    "PartPatch": PartPatch,  # noqa: F405
    "MergeRequest": MergeRequest,  # noqa: F405
    "MeasureRequest": MeasureRequest,  # noqa: F405
    "MeasureResponse": MeasureResponse,  # noqa: F405
    "DemoScan": DemoScan,  # noqa: F405
    "PrinterProfile": PrinterProfile,  # noqa: F405
}
