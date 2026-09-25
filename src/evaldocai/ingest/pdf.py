"""PyMuPDF-first PDF parser with a local RapidOCR fallback."""

from __future__ import annotations

import hashlib
import io
from collections.abc import Callable
from os import PathLike
from pathlib import Path
from typing import Any, BinaryIO

from evaldocai.models import BoundingBox, DocumentBlock, DocumentPage, ParsedDocument

from .layout import BlockCandidate, classify_blocks, order_for_reading
from .ocr import normalize_rapidocr_output

try:  # PyMuPDF switched its preferred import name; support both during transition.
    import pymupdf
except ImportError:  # pragma: no cover - exercised only on older PyMuPDF releases
    import fitz as pymupdf  # type: ignore[no-redef]


class DocumentParseError(RuntimeError):
    """The supplied bytes could not be parsed as an accessible PDF."""


class OCRUnavailableError(DocumentParseError):
    """A scanned page requires OCR but RapidOCR is unavailable."""


class PDFParser:
    """Parse PDFs locally, preferring embedded text over OCR.

    ``ocr_engine`` may be any callable returning a RapidOCR-compatible result.
    When it is omitted, RapidOCR is imported and initialized only if a page does
    not have enough embedded text. This keeps born-digital PDFs fast and makes
    the extraction path explicit in every block's ``source`` field.
    """

    def __init__(
        self,
        *,
        native_text_threshold: int = 40,
        ocr_dpi: int = 200,
        ocr_engine: Callable[[Any], Any] | None = None,
        detect_stamps: bool = True,
    ) -> None:
        if native_text_threshold < 0:
            raise ValueError("native_text_threshold must be non-negative")
        if ocr_dpi < 72:
            raise ValueError("ocr_dpi must be at least 72")
        self.native_text_threshold = native_text_threshold
        self.ocr_dpi = ocr_dpi
        self.detect_stamps = detect_stamps
        self._ocr_engine = ocr_engine

    def parse(
        self,
        source: str | PathLike[str] | bytes | bytearray | BinaryIO,
        *,
        file_name: str | None = None,
    ) -> ParsedDocument:
        """Parse a path, PDF bytes, or binary stream into canonical models."""

        data, resolved_name = _read_source(source, file_name=file_name)
        digest = hashlib.sha256(data).hexdigest()
        document_id = f"doc_{digest[:16]}"

        try:
            document = pymupdf.open(stream=data, filetype="pdf")
        except Exception as exc:  # PyMuPDF uses several exception types by version.
            raise DocumentParseError(f"Unable to open {resolved_name!r} as a PDF: {exc}") from exc

        try:
            if getattr(document, "needs_pass", False):
                raise DocumentParseError("Password-protected PDFs must be unlocked before parsing")
            pages = [
                self._parse_page(page, digest=digest, page_number=index + 1)
                for index, page in enumerate(document)
            ]
        finally:
            document.close()

        return ParsedDocument(
            document_id=document_id,
            file_name=resolved_name,
            sha256=digest,
            parser="pymupdf-native+rapidocr-v3",
            pages=pages,
        )

    def _parse_page(self, page: Any, *, digest: str, page_number: int) -> DocumentPage:
        width = float(page.rect.width)
        height = float(page.rect.height)
        native_blocks = _extract_native_blocks(page)
        meaningful_chars = sum(
            char.isalnum() for block in native_blocks for char in block.text
        )

        pixmap = None
        if native_blocks and meaningful_chars >= self.native_text_threshold:
            blocks = native_blocks
        else:
            pixmap = _render_page(page, dpi=self.ocr_dpi)
            blocks = self._extract_ocr_blocks(page, pixmap)

        table_regions = _native_table_regions(page)
        if pixmap is not None:
            table_regions.extend(
                _raster_table_regions(pixmap, page_width=width, page_height=height)
            )
        if self.detect_stamps:
            stamp_pixmap = pixmap or _render_page(page, dpi=96)
            stamp = _detect_red_stamp(stamp_pixmap, page_width=width, page_height=height)
            if stamp is not None:
                stamp_bbox, stamp_confidence, density = stamp
                overlapping = [
                    block
                    for block in blocks
                    if _intersection_fraction(block.bbox, stamp_bbox) >= 0.2
                ]
                if overlapping:
                    for block in overlapping:
                        block.kind = "stamp"
                        block.metadata["stamp_detector"] = "rgb-red-region"
                        block.metadata["red_pixel_density"] = round(density, 4)
                else:
                    blocks.append(
                        BlockCandidate(
                            text="[red stamp]",
                            bbox=stamp_bbox,
                            confidence=stamp_confidence,
                            source="rapidocr" if pixmap is not None else "native",
                            kind="stamp",
                            metadata={
                                "stamp_detector": "rgb-red-region",
                                "red_pixel_density": round(density, 4),
                                "visual_only": True,
                            },
                        )
                    )

        blocks = _merge_table_regions(blocks, table_regions)
        classify_blocks(
            blocks,
            page_width=width,
            page_height=height,
            table_regions=table_regions,
        )
        ordered = order_for_reading(blocks, page_width=width)
        document_blocks = [
            _to_document_block(
                block,
                digest=digest,
                page_number=page_number,
                reading_order=reading_order,
            )
            for reading_order, block in enumerate(ordered)
        ]
        return DocumentPage(
            page=page_number,
            width=width,
            height=height,
            blocks=document_blocks,
        )

    def _extract_ocr_blocks(self, page: Any, pixmap: Any) -> list[BlockCandidate]:
        engine = self._get_ocr_engine()
        image = _pixmap_as_bgr_array(pixmap)
        try:
            output = engine(image)
        except Exception as exc:
            raise DocumentParseError(f"RapidOCR failed on page {page.number + 1}: {exc}") from exc

        scale_x = float(page.rect.width) / pixmap.width
        scale_y = float(page.rect.height) / pixmap.height
        blocks: list[BlockCandidate] = []
        for line in normalize_rapidocr_output(output):
            x0, y0, x1, y1 = line.bbox
            bbox = _clipped_bbox(
                x0 * scale_x,
                y0 * scale_y,
                x1 * scale_x,
                y1 * scale_y,
                width=float(page.rect.width),
                height=float(page.rect.height),
            )
            if bbox is None:
                continue
            blocks.append(
                BlockCandidate(
                    text=line.text,
                    bbox=bbox,
                    confidence=line.confidence,
                    source="rapidocr",
                    metadata={"ocr_dpi": self.ocr_dpi},
                )
            )
        return blocks

    def _get_ocr_engine(self) -> Callable[[Any], Any]:
        if self._ocr_engine is not None:
            return self._ocr_engine
        try:
            from rapidocr import RapidOCR
        except ImportError as exc:  # pragma: no cover - dependency is installed in normal builds
            raise OCRUnavailableError(
                "This PDF contains a scanned page. Install `rapidocr[onnxruntime]` "
                "or inject an OCR engine into PDFParser."
            ) from exc
        try:
            self._ocr_engine = RapidOCR()
        except Exception as exc:  # model/provider errors should remain actionable
            raise OCRUnavailableError(f"RapidOCR could not be initialized: {exc}") from exc
        return self._ocr_engine


def parse_pdf(
    source: str | PathLike[str] | bytes | bytearray | BinaryIO,
    *,
    file_name: str | None = None,
    native_text_threshold: int = 40,
    ocr_dpi: int = 200,
    ocr_engine: Callable[[Any], Any] | None = None,
    detect_stamps: bool = True,
) -> ParsedDocument:
    """Convenience wrapper around :class:`PDFParser`."""

    return PDFParser(
        native_text_threshold=native_text_threshold,
        ocr_dpi=ocr_dpi,
        ocr_engine=ocr_engine,
        detect_stamps=detect_stamps,
    ).parse(source, file_name=file_name)


def _read_source(
    source: str | PathLike[str] | bytes | bytearray | BinaryIO,
    *,
    file_name: str | None,
) -> tuple[bytes, str]:
    if isinstance(source, (str, PathLike)):
        path = Path(source)
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise DocumentParseError(f"Unable to read PDF {path}: {exc}") from exc
        resolved_name = file_name or path.name
    elif isinstance(source, (bytes, bytearray)):
        data = bytes(source)
        resolved_name = file_name or "document.pdf"
    elif hasattr(source, "read"):
        try:
            value = source.read()
        except (OSError, io.UnsupportedOperation) as exc:
            raise DocumentParseError(f"Unable to read PDF stream: {exc}") from exc
        if not isinstance(value, (bytes, bytearray)):
            raise TypeError("PDF streams must be opened in binary mode")
        data = bytes(value)
        stream_name = getattr(source, "name", None)
        resolved_name = file_name or (Path(stream_name).name if stream_name else "document.pdf")
    else:
        raise TypeError("source must be a path, PDF bytes, or a binary stream")

    if not data:
        raise DocumentParseError("The supplied PDF is empty")
    return data, Path(resolved_name).name


def _extract_native_blocks(page: Any) -> list[BlockCandidate]:
    try:
        payload = page.get_text("dict", sort=False)
    except (RuntimeError, ValueError):
        return []

    blocks: list[BlockCandidate] = []
    for raw_block in payload.get("blocks", []):
        if raw_block.get("type", 0) != 0:
            continue
        lines: list[str] = []
        sizes: list[float] = []
        for raw_line in raw_block.get("lines", []):
            spans = raw_line.get("spans", [])
            line = "".join(str(span.get("text", "")) for span in spans).strip()
            if line:
                lines.append(line)
            sizes.extend(
                float(span.get("size", 0.0))
                for span in spans
                if span.get("text", "").strip()
            )
        text = "\n".join(lines).strip()
        if not text:
            continue
        raw_bbox = raw_block.get("bbox")
        if not raw_bbox or len(raw_bbox) < 4:
            continue
        bbox = _clipped_bbox(
            *raw_bbox[:4],
            width=float(page.rect.width),
            height=float(page.rect.height),
        )
        if bbox is None:
            continue
        blocks.append(
            BlockCandidate(
                text=text,
                bbox=bbox,
                confidence=1.0,
                source="native",
                font_size=max(sizes, default=0.0),
                metadata={"line_count": len(lines)},
            )
        )
    return blocks


def _native_table_regions(page: Any) -> list[BoundingBox]:
    find_tables = getattr(page, "find_tables", None)
    if not callable(find_tables):
        return []
    try:
        finder = find_tables()
    except Exception:  # Table detection is additive and must not break ingestion.
        return []
    regions: list[BoundingBox] = []
    for table in getattr(finder, "tables", []):
        raw_bbox = getattr(table, "bbox", None)
        if raw_bbox and len(raw_bbox) >= 4:
            bbox = _clipped_bbox(
                *raw_bbox[:4],
                width=float(page.rect.width),
                height=float(page.rect.height),
            )
            if bbox is not None:
                regions.append(bbox)
    return regions


def _render_page(page: Any, *, dpi: int) -> Any:
    scale = dpi / 72.0
    matrix = pymupdf.Matrix(scale, scale)
    return page.get_pixmap(matrix=matrix, colorspace=pymupdf.csRGB, alpha=False)


def _pixmap_as_bgr_array(pixmap: Any) -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - NumPy is a required dependency
        raise OCRUnavailableError("NumPy is required to render pages for OCR") from exc
    rgb = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width, pixmap.n
    )[..., :3]
    # RapidOCR follows OpenCV's BGR convention for ndarray inputs.
    return np.ascontiguousarray(rgb[..., ::-1])


def _detect_red_stamp(
    pixmap: Any, *, page_width: float, page_height: float
) -> tuple[BoundingBox, float, float] | None:
    """Detect a compact red-ink region, which is a useful stamp/seal signal."""

    try:
        import numpy as np
    except ImportError:  # pragma: no cover - NumPy is a required dependency
        return None

    rgb = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width, pixmap.n
    )[..., :3]
    red = rgb[..., 0].astype(np.int16)
    green = rgb[..., 1].astype(np.int16)
    blue = rgb[..., 2].astype(np.int16)
    mask = (red >= 145) & ((red - green) >= 45) & ((red - blue) >= 40)
    ys, xs = np.nonzero(mask)
    minimum_pixels = max(35, int(pixmap.width * pixmap.height * 0.00004))
    if len(xs) < minimum_pixels:
        return None

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    region_width = x1 - x0
    region_height = y1 - y0
    region_area = max(1, region_width * region_height)
    density = float(mask[y0:y1, x0:x1].sum() / region_area)
    page_area = pixmap.width * pixmap.height

    # Reject red page borders, backgrounds, and isolated annotations.
    if (
        region_width < 7
        or region_height < 7
        or region_width > pixmap.width * 0.48
        or region_height > pixmap.height * 0.48
        or region_area > page_area * 0.18
        or density < 0.035
    ):
        return None

    bbox = BoundingBox(
        x0=x0 * page_width / pixmap.width,
        y0=y0 * page_height / pixmap.height,
        x1=x1 * page_width / pixmap.width,
        y1=y1 * page_height / pixmap.height,
    )
    confidence = min(0.98, 0.65 + density * 0.5)
    return bbox, confidence, density


def _raster_table_regions(
    pixmap: Any, *, page_width: float, page_height: float
) -> list[BoundingBox]:
    """Locate ruled table grids on scanned pages using inexpensive morphology."""

    try:
        import cv2
        import numpy as np
    except ImportError:  # pragma: no cover - RapidOCR installs both in normal builds
        return []

    rgb = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width, pixmap.n
    )[..., :3]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    binary = cv2.threshold(gray, 205, 255, cv2.THRESH_BINARY_INV)[1]
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(20, pixmap.width // 24), 1)
    )
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(20, pixmap.height // 36))
    )
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    grid = cv2.bitwise_or(horizontal, vertical)
    contours, _hierarchy = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions: list[BoundingBox] = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if width < pixmap.width * 0.35 or height < pixmap.height * 0.06:
            continue
        if width > pixmap.width * 0.96 or height > pixmap.height * 0.75:
            continue
        crop = grid[y : y + height, x : x + width]
        if not crop.size or float(np.count_nonzero(crop) / crop.size) < 0.015:
            continue
        region = _clipped_bbox(
            x * page_width / pixmap.width,
            y * page_height / pixmap.height,
            (x + width) * page_width / pixmap.width,
            (y + height) * page_height / pixmap.height,
            width=page_width,
            height=page_height,
        )
        if region is not None:
            regions.append(region)
    regions.sort(key=lambda item: (item.y0, item.x0))
    return regions


def _merge_table_regions(
    blocks: list[BlockCandidate], regions: list[BoundingBox]
) -> list[BlockCandidate]:
    """Represent each ruled table as one row-major, layout-preserving block."""

    remaining = list(blocks)
    merged: list[BlockCandidate] = []
    for table_index, region in enumerate(regions, start=1):
        cells = [
            block
            for block in remaining
            if region.x0 <= (block.bbox.x0 + block.bbox.x1) / 2 <= region.x1
            and region.y0 <= (block.bbox.y0 + block.bbox.y1) / 2 <= region.y1
        ]
        if not cells:
            continue
        for cell in cells:
            remaining.remove(cell)
        if len(cells) == 1:
            cell = cells[0]
            cell.kind = "table"
            cell.metadata["table_id"] = table_index
            merged.append(cell)
            continue

        cells.sort(key=lambda item: (item.bbox.y0, item.bbox.x0))
        typical_height = sorted(cell.bbox.height for cell in cells)[len(cells) // 2]
        tolerance = max(3.0, typical_height * 0.75)
        rows: list[list[BlockCandidate]] = []
        row_centers: list[float] = []
        for cell in cells:
            center = (cell.bbox.y0 + cell.bbox.y1) / 2
            row_index = next(
                (
                    index
                    for index, row_center in enumerate(row_centers)
                    if abs(center - row_center) <= tolerance
                ),
                None,
            )
            if row_index is None:
                rows.append([cell])
                row_centers.append(center)
            else:
                rows[row_index].append(cell)
                row_centers[row_index] = sum(
                    (item.bbox.y0 + item.bbox.y1) / 2 for item in rows[row_index]
                ) / len(rows[row_index])
        ordered_rows = [
            sorted(row, key=lambda item: item.bbox.x0)
            for _center, row in sorted(zip(row_centers, rows, strict=True))
        ]
        text = "\n".join(" | ".join(cell.text for cell in row) for row in ordered_rows)
        merged.append(
            BlockCandidate(
                text=text,
                bbox=region,
                confidence=sum(cell.confidence for cell in cells) / len(cells),
                source=cells[0].source,
                kind="table",
                metadata={
                    "table_id": table_index,
                    "row_count": len(ordered_rows),
                    "column_count": max(len(row) for row in ordered_rows),
                    "cell_count": len(cells),
                    "format": "pipe-delimited",
                },
            )
        )
    return [*remaining, *merged]


def _intersection_fraction(left: BoundingBox, right: BoundingBox) -> float:
    width = max(0.0, min(left.x1, right.x1) - max(left.x0, right.x0))
    height = max(0.0, min(left.y1, right.y1) - max(left.y0, right.y0))
    intersection = width * height
    smaller_area = max(1.0, min(left.width * left.height, right.width * right.height))
    return intersection / smaller_area


def _clipped_bbox(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    width: float,
    height: float,
) -> BoundingBox | None:
    x0, x1 = sorted((float(x0), float(x1)))
    y0, y1 = sorted((float(y0), float(y1)))
    x0, x1 = max(0.0, x0), min(width, x1)
    y0, y1 = max(0.0, y0), min(height, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)


def _to_document_block(
    block: BlockCandidate,
    *,
    digest: str,
    page_number: int,
    reading_order: int,
) -> DocumentBlock:
    bbox_signature = ",".join(
        f"{value:.3f}"
        for value in (block.bbox.x0, block.bbox.y0, block.bbox.x1, block.bbox.y1)
    )
    identity = "|".join(
        (
            digest,
            str(page_number),
            str(reading_order),
            block.source,
            bbox_signature,
            block.text,
        )
    )
    suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return DocumentBlock(
        block_id=f"p{page_number:04d}-b{reading_order:04d}-{suffix}",
        page=page_number,
        kind=block.kind,
        text=block.text,
        bbox=block.bbox,
        confidence=block.confidence,
        source=block.source,
        reading_order=reading_order,
        metadata=block.metadata,
    )
