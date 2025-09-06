import os
import json
import cv2
import yaml
import argparse
import numpy as np
from PIL import Image  # for saving PNG with DPI


# ---------- utilities ----------
def mm_to_px(mm: float, dpi: int) -> int:
    """Convert millimeters to pixels using DPI."""
    return int(round((mm / 25.4) * dpi))


def put_multiline_text(img, lines, org, color=(0, 0, 0), font_size=0.7, thickness=1):
    """
    Draw multi-line left-aligned text using OpenCV.
    Args:
        img: target image (BGR or Gray)
        lines: list[str]
        org: (x, y) top-left anchor for first line
    """
    x, y = org
    # Simple one-line join; adjust as needed for real multiline
    line = "".join([f"{l}, " for l in lines])
    cv2.putText(
        img,
        line,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_size,
        color,
        thickness,
        cv2.LINE_AA,
    )


def aruco_dict_name(dict_id: int) -> str:
    """Return a human-readable name for a cv2.aruco dict id."""
    for k in dir(cv2.aruco):
        if k.startswith("DICT_") and getattr(cv2.aruco, k) == dict_id:
            return k
    return f"DICT({dict_id})"


def get_aruco_dict_by_name(name: str):
    """Resolve cv2.aruco predefined dictionary from its name string."""
    if not hasattr(cv2.aruco, name):
        raise ValueError(
            f"Unknown ArUco dictionary name '{name}'. "
            f"Check cv2.aruco.* constants (e.g., DICT_5X5_100)."
        )
    dict_id = getattr(cv2.aruco, name)
    return cv2.aruco.getPredefinedDictionary(dict_id), dict_id


# ---------- main generator with margin details ----------
def generate_charuco_board_with_details(
    out_path: str,
    dict_id: int,
    squares_x: int,
    squares_y: int,
    square_len_mm: float,
    marker_len_mm: float,
    dpi: int,
    margin_mm: float,
    show_ids_on_board: bool = False,
    show_preview: bool = False,
):
    """
    Generate a ChArUco board PNG with exact physical size and print-friendly details
    (dictionary name, geometry, DPI) in the margin.

    Print at 100% scale (no fit-to-page) to preserve dimensions.
    """
    # 1) Build board (OpenCV expects meters)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    try:
        board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y),
            square_len_mm / 1000.0,
            marker_len_mm / 1000.0,
            aruco_dict,
        )
        use_generate = True
    except Exception:
        # Fallback for older OpenCV versions
        board = cv2.aruco.CharucoBoard_create(
            squares_x,
            squares_y,
            square_len_mm / 1000.0,
            marker_len_mm / 1000.0,
            aruco_dict,
        )
        use_generate = False

    # 2) Compute pixel sizes
    px_per_mm = dpi / 25.4
    board_w_mm = squares_x * square_len_mm
    board_h_mm = squares_y * square_len_mm
    img_w_px = mm_to_px(board_w_mm + 2 * margin_mm, dpi)
    img_h_px = mm_to_px(board_h_mm + 2 * margin_mm, dpi)
    margin_px = mm_to_px(margin_mm, dpi)

    # 3) Render board (white background)
    if use_generate:
        board_img = cv2.aruco.CharucoBoard.generateImage(
            board, (img_w_px, img_h_px), marginSize=margin_px, borderBits=1
        )
    else:
        # If generateImage is not available, render into a blank image manually
        board_img = 255 * np.ones((img_h_px, img_w_px), dtype=np.uint8)
        # Note: Older APIs may require cv2.aruco.drawPlanarBoard or similar.
        try:
            cv2.aruco.drawPlanarBoard(
                board, (img_w_px - 2 * margin_px, img_h_px - 2 * margin_px), board_img[margin_px:-margin_px, margin_px:-margin_px], 1
            )
        except Exception as e:
            raise RuntimeError(
                "Your OpenCV version lacks CharucoBoard.generateImage and drawPlanarBoard. "
                "Please upgrade OpenCV (opencv-contrib-python)."
            ) from e

    # Ensure 3-channel BGR for drawing colored text/graphics (even if original is grayscale)
    if board_img.ndim == 2:
        board_img = cv2.cvtColor(board_img, cv2.COLOR_GRAY2BGR)

    # 4) Optional marker IDs on the board (only for visual checking)
    if show_ids_on_board:
        try:
            params = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(aruco_dict, params)
            corners, ids, _ = detector.detectMarkers(board_img)
        except Exception:
            corners, ids, _ = cv2.aruco.detectMarkers(board_img, aruco_dict)
        if ids is not None:
            for c, i in zip(corners, ids.flatten()):
                c = c.reshape(4, 2).astype(int)
                center = c.mean(axis=0).astype(int)
                cv2.putText(
                    board_img,
                    str(int(i)),
                    tuple(center),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (128, 128, 128),
                    1,
                    cv2.LINE_AA,
                )

    # 5) Details in margin (top-left area inside margin)
    text_x = margin_px
    text_y = int(margin_px / 2)
    details = [
        f"ChArUco: {squares_x} x {squares_y}",
        f"Square: {square_len_mm:.1f} mm, Marker: {marker_len_mm:.1f} mm",
        f"Dict: {aruco_dict_name(dict_id)}",
        f"DPI: {dpi}, Margin: {margin_mm:.1f} mm",
        "PRINT AT 100% SCALE (no fit-to-page)",
    ]
    put_multiline_text(board_img, details, (text_x, text_y), font_size=0.3)

    # 6) Save with DPI metadata via Pillow
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    Image.fromarray(cv2.cvtColor(board_img, cv2.COLOR_BGR2RGB)).save(
        out_path, format="PNG", dpi=(dpi, dpi)
    )

    # 7) Save metadata JSON next to the image
    base, _ = os.path.splitext(out_path)
    json_path = base + ".json"
    meta = {
        "aruco_dict": aruco_dict_name(dict_id),
        "squares_x": squares_x,
        "squares_y": squares_y,
        "square_len_mm": square_len_mm,
        "marker_len_mm": marker_len_mm,
        "dpi": dpi,
        "margin_mm": margin_mm,
        "notes": "Print at 100% scale (no fit-to-page). Details are placed in the margin.",
    }
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] Saved board -> {out_path} (DPI={dpi})")
    print(f"[OK] Saved meta  -> {json_path}")

    if show_preview:
        cv2.imshow("ChArUco Board", board_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def load_config(path: str) -> dict:
    """Load YAML config from file path and apply defaults."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    board = cfg.get("board", {})
    pr = cfg.get("print", {})

    dict_name = board.get("dict_name", "DICT_5X5_100")
    aruco_dict, dict_id = get_aruco_dict_by_name(dict_name)

    return {
        "output_root_dir": cfg.get("output_root_dir", "./output"),
        "dict_id": dict_id,
        "squares_x": int(board.get("squares_x", 5)),
        "squares_y": int(board.get("squares_y", 5)),
        "square_len_mm": float(board.get("square_len_mm", 35.0)),
        "marker_len_mm": float(board.get("marker_len_mm", 28.0)),
        "dpi": int(pr.get("dpi", 96)),
        "margin_mm": float(pr.get("margin_mm", 10.0)),
        "show_ids_on_board": bool(pr.get("show_ids_on_board", False)),
        "show_preview": bool(pr.get("show_preview", False)),
    }


def parse_args():
    """Parse CLI arguments (config path)."""
    p = argparse.ArgumentParser(
        description="Generate a ChArUco board from a YAML config."
    )
    p.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to YAML config file (e.g., config.yaml)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    params = load_config(args.config)

    os.makedirs(params["output_root_dir"], exist_ok=True)
    print(f"[Info] Output directory: {params['output_root_dir']}")

    # Construct output filename
    output_path = os.path.join(
        params["output_root_dir"],
        f"charuco_{params['squares_x']}x{params['squares_y']}_"
        f"{int(params['square_len_mm'])}mm_"
        f"{int(params['marker_len_mm'])}mm_"
        f"{params['dpi']}dpi_"
        f"{aruco_dict_name(params['dict_id'])}.png",
    )

    generate_charuco_board_with_details(
        out_path=output_path,
        dict_id=params["dict_id"],
        squares_x=params["squares_x"],
        squares_y=params["squares_y"],
        square_len_mm=params["square_len_mm"],
        marker_len_mm=params["marker_len_mm"],
        dpi=params["dpi"],
        margin_mm=params["margin_mm"],
        show_ids_on_board=params["show_ids_on_board"],
        show_preview=params["show_preview"],
    )
