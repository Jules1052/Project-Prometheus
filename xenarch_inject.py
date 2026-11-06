"""
Xenarch synthetic injections: natural-terrain images with a space-hardware object added, plus the untouched twin
and a pixel mask, for injection-recovery tests (how often does the detector find an object of this size, at this
sun angle, on this terrain?). Test data only: Xenarch itself trains on natural terrain alone.

Backgrounds: 512x512 crops of the natural-terrain tiles (LROC NAC Moon, HiRISE Mars), never hardware sites.
Objects land on flat-ish lit ground, not on crater walls or in shadow. Two kinds, one object per image:

  rendered      past and new hardware built from parts at real dimensions (HARDWARE below): bodies raised on legs,
                tapering upper structures, thin masts and panels, wheels. Lit by the tile's own sun.
  real-sprite   real hardware cut out of the LROC / HiRISE crops (high-confidence boxes of boxes_coco.json): its
                bright body and the black shadow touching it, without the halo, rotated by its own body-to-shadow
                direction onto the tile's sun, only shrunk, only into tiles whose incidence is within 12 deg.

Calibrated on the real hardware (`--realism` renders each design at the sun and m/px of every real frame and
compares; report in realism_report.csv, side by side in contact-sheets/real-vs-model.jpg):

  radiometry    tiles are stretched 8-bit, so one unit of ground radiance is far more than the DN of the ground.
                `radiance_scale` estimates it from the tile's contrast and sun: the shadow of real hardware is DN 0
                and its body saturates at DN 255, and so do these. Everything is composited in radiance, blurred,
                then clipped, like a camera.
  shadows       swept from each part at its own height, so they taper and detach like real ones, and bent over the terrain
                (relief from the tile's shading). Soft edge, dark centre, a little terrain and noise inside: edge width and inside
                DN are measured on real hardware and natural shadows, and reported by --realism.
  bodies        parts of different albedo with fine facets and glints, brightness rolling off below saturation, edges as soft
                as real ones (saturated share, body DN and edge width are reported by --realism).
  camera        blur measured on each tile's sharpest edges (LROC ~0.69 px, HiRISE ~0.52 px), the tile's noise.
  sun           azimuth from the tile's latitude and incidence (+ Mars season), up to the four mirror images a
                raw frame allows, picked with the tile's shading. Shading and geometry agree: median 5 deg (Moon),
                16 deg (Mars). `sun_azimuth_source` says which was used.

Variability per image (all recorded in the table): object size x0.92-1.08 (hardware has fixed dimensions; impact
craters 10-35 m); target resolution (the tile's own, and on the Moon a coarser camera by cutting 1-2x larger and
resampling); sun incidence and azimuth (the tile's); brightness (albedo x0.7-1.3, sprite contrast x0.8-1.2); gain /
gamma on the whole image; terrain (12 landform folders); context: blast zone around landers, impact scar around heat
shields and backshells, wheel tracks behind rovers, footpaths around crewed sites, ejecta around impact craters.
A design only goes into tiles where its shadow stays under MAX_SHADOW_PX (the longest shadows common on real hardware),
so tall designs appear only at high sun, and Starship not at all on HiRISE tiles.

Writes to data/synthetic-injections/: natural/<body>/<landform>/ the terrain alone (with the same gain, gamma and resampling as
its injected twin), injected/<body>/<class>/ the same terrain with the object, masks/<body>/<class>/ (1 object, 2 its cast shadow,
3 context), annotation-synthetic-injection/<body>/<class>/ one COCO file per image; metadata.csv pairs them; boxes_coco.json (the
whole set), realism_report.csv, contact-sheets/.

  .venv/bin/python xenarch_inject.py [--per-tile 1] [--sprite-frac 0.6] [--val-extra 2] [--limit N] [--seed 7] [--workers 6] [--only DESIGN]
  .venv/bin/python xenarch_inject.py --galleries      only rebuild the zoomed sheets contact-sheets/gallery_*.jpg
  .venv/bin/python xenarch_inject.py --realism        only the check against real hardware
  .venv/bin/python xenarch_inject.py --dashboard      only rebuild dashboard/injections.json + thumbnails (page: ./train.sh --injections)

Evaluate on `split` = validation (backgrounds never in Mk18's memory bank); the train-split images serve other models
and visual checks.
  .venv/bin/python xenarch_hf.py injections     publish to synthetic-injections_lroc-nac_hirise/ of the HF dataset
"""
import csv, json, math, multiprocessing, sys
from collections import Counter, defaultdict
from pathlib import Path

import contourpy
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

HERE = Path(__file__).resolve().parent
OUT = HERE / "data/synthetic-injections"
SIZE, FINE = 512, 3                        # output image side; render oversampling (fine px per image px)
TILES = {"Moon": HERE / "data/lroc-nac-set", "Mars": HERE / "data/hirise-set"}
CROPS = HERE / "data/artifact-site-images"
COCO = HERE / "artifact-site-catalog/annotations/boxes_coco.json"
SITES = HERE / "artifact-site-catalog/sites.csv"
CAMERA = {"Moon": "LROC NAC", "Mars": "HiRISE RED"}
# measured on real hardware and natural shadows: shadow edge 10-90 % Moon 3.9 / 3.2 px, Mars 2.8 / 3.0 px; body edge Moon 1.9, Mars 2.4 px;
# body brightness over the ground Moon ~90, Mars ~120 DN; shadow tips p90 Moon 31, Mars 28 px
SHADOW_FLOOR_MIN = {"Moon": 4.0, "Mars": 2.0}                     # DN: added shadows are never darker than this, see natural_floor
SHADOW_SOFT_PX = {"Moon": (1.4, 2.2), "Mars": (0.9, 1.6)}         # edge blur beyond the camera's: rough ground, scattered light
SHADOW_CORE = 0.65                                                # blurred coverage above this counts as full shadow: soft edge, dark centre
SHADOW_TEXTURE = (0.1, 0.25)                                      # how much of the terrain still shows inside a shadow
BODY_SOFT_PX = {"Moon": (0.3, 0.6), "Mars": (0.6, 1.0)}           # the same for the bright body
BODY_CONTRAST_DN = {"Moon": (110.0, 200.0), "Mars": (80.0, 170.0)} # how far above the ground bright parts can go, rolling off
ANNOTATION = "annotation-synthetic-injection"                     # folder and file suffix of the per-image COCO files
CATEGORIES = ["lander", "rover", "habitat", "power", "surface-experiment", "landing-hardware", "impact-crater"]
ALBEDO_SD, SLOPE_SD = {"Moon": 0.05, "Mars": 0.08}, {"Moon": 0.10, "Mars": 0.08}   # natural terrain, for radiance_scale
MAX_INCIDENCE = 86.0                       # the sun model stops here
MAX_SHADOW_PX = 24                         # a design only goes where its shadow stays under this: real p50-p75 is 14-27 px
SHADOW_DN = 20                             # 8-bit stretched tiles: real shadows sit at 0-15
PIXEL_SIGMA = 0.29                         # blur of integrating over one pixel, in px


# ── hardware catalogue: metres above ground, albedo relative to the regolith ─────────────────────────────────────
def poly(n, r, z1, alb, z0=0.0):
    return dict(k="poly", n=n, r=r, z1=z1, alb=alb, z0=z0)
def cyl(r, z1, alb, z0=0.0, dx=0.0, dy=0.0):
    return dict(k="cyl", r=r, z1=z1, alb=alb, z0=z0, dx=dx, dy=dy)
def box(w, l, z1, alb, z0=0.0, dx=0.0, dy=0.0, rot=0.0):
    return dict(k="box", w=w, l=l, z1=z1, alb=alb, z0=z0, dx=dx, dy=dy, rot=rot)
def dome(r, h, alb):                                 # covered in regolith: matte, shaded like a mound of the same ground
    return dict(k="dome", r=r, z1=h, alb=alb, glint=0.25)
def cone(r0, r1, z0, z1, alb):
    return dict(k="cone", r0=r0, r1=r1, z0=z0, z1=z1, alb=alb)
def hcyl(r, length, alb, z0=0.0):
    return dict(k="hcyl", r=r, length=length, alb=alb, z0=z0, z1=z0 + 2 * r)
def legs(n, r, z1, alb):
    return dict(k="legs", n=n, r=r, z1=z1, alb=alb)
def disk(r, z1, alb, dx=0.0, dy=0.0):                # thin panel or dish on a mast: its shadow detaches from the body
    return dict(k="cyl", r=r, z1=z1, alb=alb, z0=z1 - 0.12, dx=dx, dy=dy)
def dish(r, h, alb):                                 # shield lying open side up: bright rim, shadow inside
    return dict(k="dish", r=r, z1=h, alb=alb)
def canopy(r, alb, dx=0.0, dy=0.0):                  # parachute collapsed on the ground: irregular, no gores visible
    return dict(k="canopy", r=r, z1=0.35, alb=alb, dx=dx, dy=dy, glint=0.3)
def crater(diameters):
    return dict(k="crater", sizes=diameters, z1=0.0)
def wheels(per_side, half_w, half_l, w, l, h, alb):
    ys = [0.0] if per_side == 1 else [(-half_l + 2 * half_l * k / (per_side - 1)) for k in range(per_side)]
    return [box(w, l, h, alb, dx=sx * half_w, dy=y) for sx in (-1, 1) for y in ys]


HARDWARE = {
    # Moon, past
    "apollo-lm-descent-stage": dict(era="past", cls="lander", body="Moon", name="Apollo LM descent stage", context="footpaths",
        shapes=[poly(8, 2.1, 2.4, 3.0, z0=1.0), box(1.6, 1.6, 3.5, 3.5, z0=2.4), cyl(0.7, 1.0, 1.2), legs(4, 4.7, 2.0, 1.8)]),
    "surveyor": dict(era="past", cls="lander", body="Moon", name="Surveyor lander",
        shapes=[poly(3, 0.9, 1.3, 2.5, z0=0.4), legs(3, 2.1, 0.9, 1.6), cyl(0.06, 3.0, 1.5), disk(0.6, 2.6, 1.8, 0.5, 0.4)]),
    "luna-sample-return": dict(era="past", cls="lander", body="Moon", name="Luna 16-24 descent stage",
        shapes=[cyl(1.2, 1.9, 2.4, z0=0.6), legs(4, 2.1, 0.9, 1.6), box(1.0, 1.0, 2.4, 1.4, z0=1.9)]),
    "lunokhod": dict(era="past", cls="rover", body="Moon", name="Lunokhod rover",
        shapes=[box(1.3, 2.0, 1.35, 2.2, z0=0.45), box(1.5, 1.2, 1.45, 0.7, z0=1.25, dy=1.45), *wheels(4, 0.9, 0.85, 0.2, 0.5, 0.5, 1.0)], tracks=1.6),
    "apollo-lrv": dict(era="past", cls="rover", body="Moon", name="Apollo Lunar Roving Vehicle",
        shapes=[box(1.1, 2.4, 0.75, 0.9, z0=0.35), disk(0.45, 1.6, 2.6, 0.0, 0.9), *wheels(2, 1.0, 1.25, 0.23, 0.8, 0.8, 0.8)], tracks=1.8),
    "change3-lander": dict(era="past", cls="lander", body="Moon", name="Chang'e 3/4 lander",
        shapes=[box(2.4, 2.4, 3.0, 2.8, z0=1.0), box(1.0, 2.6, 2.6, 0.7, z0=2.4, dx=2.1), box(1.0, 2.6, 2.6, 0.7, z0=2.4, dx=-2.1), legs(4, 2.6, 1.2, 1.6)]),
    "yutu": dict(era="past", cls="rover", body="Moon", name="Yutu rover",
        shapes=[box(0.8, 1.3, 0.85, 2.2, z0=0.3), box(2.0, 0.7, 1.0, 0.7, z0=0.9), cyl(0.05, 1.5, 1.5, dy=0.5), *wheels(3, 0.5, 0.55, 0.15, 0.3, 0.3, 1.0)], tracks=1.0),
    "spacecraft-impact": dict(era="past", cls="impact-crater", body="Moon", name="spacecraft / rocket-stage impact crater",
        shapes=[crater((10.0, 14.0, 18.0, 25.0, 35.0))]),
    # Moon, new
    "nova-c": dict(era="new", cls="lander", body="Moon", name="Intuitive Machines Nova-C",
        shapes=[poly(6, 0.8, 4.3, 2.6, z0=0.7), legs(6, 2.3, 1.3, 1.6)]),
    "blue-ghost": dict(era="new", cls="lander", body="Moon", name="Firefly Blue Ghost",
        shapes=[poly(8, 1.6, 2.0, 2.4, z0=0.8), legs(4, 1.9, 0.9, 1.6)]),
    "blue-moon-mk1": dict(era="new", cls="lander", body="Moon", name="Blue Origin Blue Moon MK1",
        shapes=[cyl(1.5, 7.4, 2.6, z0=1.5), cone(1.5, 0.6, 7.4, 8.0, 2.6), legs(4, 3.2, 1.8, 1.6)]),
    "starship-hls": dict(era="new", cls="lander", body="both", name="SpaceX Starship HLS",
        shapes=[cyl(4.5, 44.0, 3.0, z0=3.0), cone(4.5, 0.8, 44.0, 50.0, 3.0), legs(6, 6.5, 3.0, 1.8)]),
    "surface-habitat": dict(era="new", cls="habitat", body="Moon", name="Artemis surface habitat (lying cylinder)", context="footpaths",
        shapes=[hcyl(2.0, 10.0, 1.6, z0=1.0), box(2.0, 2.0, 2.2, 1.4, dy=6.0), legs(4, 4.5, 1.2, 1.2)]),
    "lunar-terrain-vehicle": dict(era="new", cls="rover", body="Moon", name="Lunar Terrain Vehicle",
        shapes=[box(1.7, 3.2, 1.4, 2.2, z0=0.6), disk(0.5, 2.0, 1.6, 0.0, -1.1), *wheels(2, 1.1, 1.3, 0.35, 0.9, 0.9, 1.0)], tracks=2.4),
    "vertical-solar-array": dict(era="new", cls="power", body="Moon", name="vertical solar array tower",
        shapes=[box(3.0, 0.25, 16.0, 0.8, z0=3.0), cyl(0.25, 18.0, 1.4), poly(4, 1.2, 1.0, 1.8)]),
    "fission-power-unit": dict(era="new", cls="power", body="Moon", name="fission surface power unit with radiator",
        shapes=[cyl(1.1, 4.0, 2.4), box(6.0, 0.3, 3.2, 1.8, z0=0.6, dy=2.0)]),
    # Mars, past
    "viking-lander": dict(era="past", cls="lander", body="Mars", name="Viking lander",
        shapes=[poly(6, 0.6, 1.2, 2.8, z0=0.45), legs(3, 1.5, 0.7, 1.2), cyl(0.05, 2.0, 1.5), disk(0.4, 2.0, 1.4, 0.5, 0.3)]),
    "pathfinder": dict(era="past", cls="lander", body="Mars", name="Mars Pathfinder with open petals",
        shapes=[poly(3, 0.6, 0.9, 2.4, z0=0.2), cyl(1.2, 0.3, 1.8), box(1.0, 1.0, 0.3, 0.8, dx=1.0), box(1.0, 1.0, 0.3, 0.8, dx=-0.5, dy=0.87, rot=60), box(1.0, 1.0, 0.3, 0.8, dx=-0.5, dy=-0.87, rot=-60)]),
    "phoenix-insight": dict(era="past", cls="lander", body="Mars", name="Phoenix / InSight lander with round arrays",
        shapes=[cyl(0.75, 1.05, 3.0, z0=0.55), disk(1.05, 1.15, 0.9, 1.85, 0.0), disk(1.05, 1.15, 0.9, -1.85, 0.0), legs(3, 1.2, 0.7, 1.2)]),
    "mer-rover": dict(era="past", cls="rover", body="Mars", name="Spirit / Opportunity rover",
        shapes=[box(1.5, 1.5, 1.05, 2.2, z0=0.55), box(2.3, 0.5, 1.05, 1.0, z0=0.9), cyl(0.08, 1.5, 2.0, dy=0.5), *wheels(3, 0.95, 0.55, 0.16, 0.3, 0.3, 0.8)], tracks=1.2),
    "msl-rover": dict(era="past", cls="rover", body="Mars", name="Curiosity / Perseverance rover",
        shapes=[box(2.2, 2.7, 1.25, 2.6, z0=0.5), box(0.9, 0.8, 1.6, 3.0, z0=0.9, dy=-1.5), cyl(0.15, 2.2, 2.2, dx=0.5, dy=1.0), *wheels(3, 1.2, 1.2, 0.4, 0.5, 0.5, 0.9)], tracks=2.8),
    "heat-shield": dict(era="past", cls="landing-hardware", body="Mars", name="heat shield (4.5 m)", context="impact-scar",
        shapes=[dish(2.25, 0.9, 2.6)]),
    "backshell-parachute": dict(era="past", cls="landing-hardware", body="Mars", name="backshell with parachute lying flat", context="impact-scar",
        shapes=[cone(2.25, 1.0, 0.0, 1.8, 2.2), canopy(7.5, 2.4, 0.0, 16.0)]),
    # Mars, new
    "habitat-dome": dict(era="new", cls="habitat", body="Mars", name="habitat dome (10 m, regolith-shielded)", context="footpaths",
        shapes=[dome(5.0, 4.5, 1.05), box(1.5, 3.0, 2.0, 1.4, dy=6.0)]),
    "sample-retrieval-lander": dict(era="new", cls="lander", body="Mars", name="Mars Sample Retrieval Lander",
        shapes=[poly(6, 1.8, 2.5, 2.6, z0=1.0), legs(4, 3.2, 1.2, 1.6), disk(1.6, 2.7, 1.6, 3.6, 0.0), disk(1.6, 2.7, 1.6, -3.6, 0.0)]),
    "pressurised-rover": dict(era="new", cls="rover", body="Mars", name="pressurised crew rover",
        shapes=[hcyl(1.5, 6.0, 1.9, z0=0.8), *wheels(3, 1.7, 2.2, 0.5, 1.0, 1.0, 0.8)], tracks=3.0),
}
SPRITE_TYPES = {"crewed lander": "lander", "robotic lander": "lander", "rover": "rover", "landing hardware": "landing-hardware",
                "surface experiment": "surface-experiment"}
FIELDS = ["file_name", "natural_file", "mask_file", "annotation_file", "split", "body", "instrument", "kind", "era", "object_class",
          "hardware_id", "hardware_name", "sprite_source", "x_center", "y_center", "box_x_min", "box_y_min", "box_x_max",
          "box_y_max", "object_px", "shadow_length_px", "shadow_area_px", "shadow_length_m", "shadow_soft_px", "shadow_floor_dn", "body_contrast_dn", "context", "scale_factor", "width_m", "height_m", "resolution_m_px",
          "camera_downscale", "incidence_deg", "sun_azimuth_image_deg", "sun_azimuth_source", "sun_azimuth_confidence", "albedo_factor",
          "contrast_factor", "radiance_scale_dn", "psf_sigma_px", "noise_sigma", "gain", "gamma", "rotation_deg", "mirrored",
          "landform", "shadowed_fraction", "background_file", "background_x0", "background_y0", "background_side_px",
          "frame_id", "seed"]


# ── sun direction, camera and radiometry of a tile ────────────────────────────────────────────────────────────
def shading_azimuth(a, body, r0=2, r1=14):
    """(azimuth toward the sun in image degrees, 0 = +x, 90 = +y (down); strength) from the mean offset between
    dark and bright blobs of the high-passed image: on the Moon crater shadows sit on the sun side, so the offset
    points away from the sun; on Mars relief dominates and it points toward it."""
    a = np.asarray(a, np.float32)
    g = ndimage.gaussian_filter(a, 1.0); hp = g - ndimage.gaussian_filter(g, 12)
    z = hp / (np.median(np.abs(hp - np.median(hp))) * 1.4826 + 1e-6)
    D = (z < -2.0).astype(np.float32); B = (z > 2.0).astype(np.float32)
    H, W = a.shape
    C = np.fft.fftshift(np.fft.irfft2(np.conj(np.fft.rfft2(D)) * np.fft.rfft2(B), (H, W)))
    cy, cx = H // 2, W // 2
    yy, xx = np.mgrid[-r1:r1 + 1, -r1:r1 + 1]; rr = np.maximum(np.hypot(yy, xx), 1)
    win = C[cy - r1:cy + r1 + 1, cx - r1:cx + r1 + 1]
    anti, ring = win - win[::-1, ::-1], (rr >= r0) & (rr <= r1)
    vx, vy = (anti * xx / rr)[ring].sum(), (anti * yy / rr)[ring].sum()
    strength = math.hypot(vx, vy) / max((win + win[::-1, ::-1])[ring].sum() / 2, 1e-6)
    dark_to_bright = math.degrees(math.atan2(vy, vx))
    toward_sun = dark_to_bright + 180 if body == "Moon" else dark_to_bright
    return (toward_sun + 180) % 360 - 180, float(strength)


def geometric_azimuths(lat, incidence, ls=None):
    """The four image azimuths the sun can have in a raw frame (line axis along a near-polar orbit, north up or down,
    east left or right), from latitude, incidence and, on Mars, the season (subsolar latitude from Ls)."""
    dec = 0.0 if ls is None else math.asin(math.sin(math.radians(25.19)) * math.sin(math.radians(ls)))
    ph = math.radians(lat)
    cos_h = (math.cos(math.radians(incidence)) - math.sin(ph) * math.sin(dec)) / max(math.cos(ph) * math.cos(dec), 1e-3)
    cos_h = max(-1.0, min(1.0, cos_h)); sin_h = math.sqrt(1 - cos_h * cos_h)
    east, north = math.cos(dec) * sin_h, math.sin(dec) * math.cos(ph) - math.cos(dec) * cos_h * math.sin(ph)
    return [math.degrees(math.atan2(sy * north, sx * east)) for sx in (1, -1) for sy in (1, -1)]


def sun_azimuth(a, body, lat=None, incidence=None, ls=None):
    """(azimuth, confidence, source). The geometric candidate nearest the shading estimate, unless the frame is polar
    (the line axis is not north there) or, on Mars, strong shading disagrees with every candidate."""
    az, conf = shading_azimuth(a, body)
    if lat is None or incidence is None or abs(lat) > 75:
        return az, conf, "shading"
    near = min(geometric_azimuths(lat, incidence, ls), key=lambda k: abs((az - k + 180) % 360 - 180))
    if body == "Mars" and conf > 0.3 and abs((az - near + 180) % 360 - 180) > 30:
        return az, conf, "shading"
    return near, conf, "geometry"


def tile_psf(a, s0=1.0):
    """Camera blur (Gaussian sigma, px) from the sharpest unclipped edges: their gradient against a re-blurred copy."""
    a = np.asarray(a, np.float32)
    ok = ndimage.binary_erosion((a > 3) & (a < 252), iterations=2)
    if ok.sum() < 200:
        return 0.7
    g1 = np.hypot(*np.gradient(a)); g2 = np.hypot(*np.gradient(ndimage.gaussian_filter(a, s0)))
    sel = ok & (g1 >= np.percentile(g1[ok], 99.5))
    r = g1[sel] / np.maximum(g2[sel], 1e-3); r = r[r > 1.02]
    return float(np.clip(np.median(s0 / np.sqrt(r ** 2 - 1)), 0.35, 1.5)) if len(r) else 0.7


def radiance_scale(a, incidence, body):
    """(DN of flat lit ground, DN per unit of ground radiance). The tiles are stretched: their contrast against what
    natural terrain has at this sun (albedo spread + slopes x tan i) tells how many DN one unit of radiance became."""
    a = np.asarray(a, np.float32); lit = a[a > SHADOW_DN]
    if lit.size < 200:
        lit = a.ravel()
    level = float(np.median(lit)); sd = float(np.median(np.abs(lit - level)) * 1.4826)
    contrast = math.hypot(ALBEDO_SD[body], SLOPE_SD[body] * math.tan(math.radians(min(incidence, 85.0))))
    return level, max(level * 1.05, sd / contrast)


def to_dn(rad, base, floor, u, top):
    """Image DN of a relative radiance. Below the ground level: a straight line down to the shadow floor, with no overshoot to
    be clipped (which is what makes edges harsh). Above it: the tile's stretch rolling off toward `top` DN over the ground, so
    bright parts keep their structure instead of turning into a flat white shape."""
    return np.where(rad < 1, floor + (base - floor) * np.clip(rad, 0, None), base + top * (1 - np.exp(-u * np.maximum(rad - 1, 0) / top)))


def body_top(body, level, rng):
    """How far above the ground bright parts can go: they roll off before saturating, even on bright terrain."""
    return min(rng.uniform(*BODY_CONTRAST_DN[body]), max(40.0, 250.0 - level))


def natural_floor(bg, body, rng):
    """DN inside an added shadow: as dark as the darkest natural shadows of this crop, so it does not stand out from them."""
    low = SHADOW_FLOOR_MIN[body]
    dark = float(np.percentile(ndimage.gaussian_filter(bg, 1.0), 1.0))
    return float(np.clip(dark * rng.uniform(0.7, 1.0), low, max(low, min(26.0, 0.25 * float(np.median(bg))))))   # natural shadows: DN 16-18


def noise_sigma(a):
    d = a[:, 1:] - a[:, :-1]
    return float(np.median(np.abs(d - np.median(d))) * 1.4826 / math.sqrt(2))


def pick_spot(bg, rng, margin=96):
    """A place an object could stand and be seen: lit, not on a steep wall. Random among those, inside the margin."""
    dark = ndimage.uniform_filter((bg < SHADOW_DN).astype(np.float32), 31)
    rough = ndimage.uniform_filter(np.hypot(*np.gradient(ndimage.gaussian_filter(bg, 2.0))), 21)
    ok = (dark < 0.01) & (rough <= np.percentile(rough, 70))
    inner = np.zeros(bg.shape, bool); inner[margin:-margin, margin:-margin] = True
    ys, xs = np.nonzero(ok & inner)
    if len(xs) == 0:
        ys, xs = np.nonzero(inner)
    k = int(rng.integers(len(xs)))
    return float(xs[k] + rng.random()), float(ys[k] + rng.random())


# ── rendering ─────────────────────────────────────────────────────────────────────────────────────────────────
def _poly_mask(X, Y, n, r, rot=0.0):
    phi = np.arctan2(Y, X) - rot
    seg = 2 * np.pi / n
    return np.hypot(X, Y) <= r * np.cos(np.pi / n) / np.cos((phi % seg) - seg / 2)


def _field(shape, cell, rng, scale_m):
    """Unit-variance noise with features of about scale_m metres."""
    n = ndimage.gaussian_filter(rng.normal(0, 1, shape), max(scale_m / cell, 0.5))
    return n / (n.std() + 1e-6)


def _part(sh, X, Y, rot, scale, rng, cell):
    """(mask, top height, bottom height, albedo) of one part on the grid X, Y (metres, around the object)."""
    c, s = math.cos(rot), math.sin(rot)
    dx, dy = sh.get("dx", 0.0) * scale, sh.get("dy", 0.0) * scale
    u0, v0 = X - (dx * c - dy * s), Y - (dx * s + dy * c)
    cr, sr = math.cos(rot + math.radians(sh.get("rot", 0.0))), math.sin(rot + math.radians(sh.get("rot", 0.0)))
    u, v = u0 * cr + v0 * sr, -u0 * sr + v0 * cr
    k, z0, alb = sh["k"], sh.get("z0", 0.0) * scale, sh["alb"]
    z1 = sh["z1"] * scale
    if k == "poly":
        m = _poly_mask(u, v, sh["n"], sh["r"] * scale); z = np.where(m, z1, 0.0)
    elif k == "cyl":
        m = np.hypot(u, v) <= sh["r"] * scale; z = np.where(m, z1, 0.0)
    elif k == "box":
        m = (np.abs(u) <= sh["w"] * scale / 2) & (np.abs(v) <= sh["l"] * scale / 2); z = np.where(m, z1, 0.0)
    elif k == "dome":
        rho = np.hypot(u, v); m = rho <= sh["r"] * scale
        z = np.where(m, z1 * np.sqrt(np.clip(1 - (rho / (sh["r"] * scale)) ** 2, 0, 1)), 0.0)
    elif k == "cone":
        rho = np.hypot(u, v); r0, r1 = sh["r0"] * scale, sh["r1"] * scale; m = rho <= r0
        z = np.where(m, z0 + (z1 - z0) * np.clip((r0 - rho) / max(r0 - r1, 1e-6), 0, 1), 0.0)
        alb = np.where(_field(X.shape, cell, rng, 0.5) > 0.6, 0.3 * alb, alb)   # dark patches: hatches, torn thermal cover
    elif k == "hcyl":
        r = sh["r"] * scale; m = (np.abs(v) <= sh["length"] * scale / 2) & (np.abs(u) <= r)
        z = np.where(m, z0 + r + np.sqrt(np.clip(r * r - u * u, 0, None)), 0.0)
    elif k == "dish":
        rho = np.hypot(u, v); m = rho <= sh["r"] * scale
        z = np.where(m, 0.15 * scale + z1 * (rho / (sh["r"] * scale)) ** 2, 0.0)
        alb = np.where(rho > 0.72 * sh["r"] * scale, alb, 0.25 * alb)            # the thermal coat inside is dark
    elif k == "canopy":
        th, rho = np.arctan2(v, u), np.hypot(u, v * rng.uniform(1.0, 1.9))
        edge = 1 + sum(rng.uniform(0.04, 0.16) / math.sqrt(q) * np.cos(q * th + rng.uniform(0, 6.3)) for q in range(2, 8))
        m = rho <= sh["r"] * scale * rng.uniform(0.65, 1.1) * edge
        folds = _field(X.shape, cell, rng, 0.5)
        fade = ndimage.gaussian_filter(m.astype(np.float32), 0.5 * scale / cell)   # thin cloth: the edge fades into the ground
        z = np.where(m, np.clip(0.15 + 0.1 * folds, 0.03, z1), 0.0); alb = np.where(m, 1 + (alb * np.exp(0.2 * folds) - 1) * fade, 0.0)
    elif k == "legs":
        z = np.zeros_like(X); m = np.zeros(X.shape, bool)
        for q in range(int(sh["n"])):
            a = rot + 2 * np.pi * q / sh["n"] + np.pi / sh["n"]
            fx, fy = sh["r"] * scale * math.cos(a), sh["r"] * scale * math.sin(a); ax, ay = 0.45 * fx, 0.45 * fy
            t = np.clip(((X - fx) * (ax - fx) + (Y - fy) * (ay - fy)) / ((ax - fx) ** 2 + (ay - fy) ** 2 + 1e-9), 0, 1)
            near = np.hypot(X - fx - t * (ax - fx), Y - fy - t * (ay - fy)) <= 0.1 * scale
            pad = np.hypot(X - fx, Y - fy) <= 0.35 * scale                          # strut + footpad
            z = np.maximum(z, np.where(near, z1 * t, 0.0)); z = np.maximum(z, np.where(pad, 0.12 * scale, 0.0)); m |= near | pad
        alb = 0.65 * alb                                                        # struts barely show from orbit
    else:
        raise ValueError(k)
    return m, z, z0, alb


def _crater(D, X, Y, rng, cell, incidence):
    """Fresh impact crater, from the real S-IVB and Ranger frames: height field, and the ground factor of what the impact
    threw out. Irregular rim, bumpy floor, blocky rim; at high sun bright rays around a dark blotchy centre, at low sun
    mostly its shadow."""
    r = D / 2; rho, th = np.hypot(X, Y), np.arctan2(Y, X)
    a, e = rng.uniform(0, np.pi), rng.uniform(1.0, 1.35)                         # oblique impacts are elongated
    rho = rho * np.sqrt((np.cos(th - a) / e) ** 2 + (np.sin(th - a) * e) ** 2)
    wob = sum(rng.uniform(0.012, 0.04) / math.sqrt(q) * np.cos(q * th + rng.uniform(0, 6.3)) for q in range(2, 11))
    rr = rho / (r * (1 + wob))
    depth = rng.uniform(0.12, 0.2) * D
    H = np.where(rr < 1, -depth * (1 - rr ** 2), 0.0) + 0.035 * D * np.exp(-((rr - 1) / 0.22) ** 2)
    H = H + 0.02 * D * _field(X.shape, cell, rng, max(0.07 * D, 0.5)) * np.clip(1.5 - rr, 0, 1)
    high_sun = max(0.0, (60 - incidence) / 60)                                  # albedo shows at high sun, relief at low
    # rays: noise in azimuth at several scales, streaks reaching out to 3 radii
    k = np.arange(2, 40); amp = rng.uniform(0.3, 1.0, len(k)) / np.sqrt(k); ph = rng.uniform(0, 6.3, len(k))
    th1 = np.linspace(-np.pi, np.pi, 720); ang = (amp[:, None] * np.cos(k[:, None] * th1 + ph[:, None])).sum(0); ang = (ang - ang.mean()) / ang.std()
    rays = np.clip(np.interp(th, th1, ang) + 0.5, 0, None)
    reach = np.clip(1 - (rr - 1) / (1.5 + 0.5 * rays), 0, 1) ** 1.2 * np.clip((rr - 0.4) / 0.3, 0, 1)   # walls too: fresh, bright
    blocks = np.exp(0.5 * _field(X.shape, cell, rng, max(0.04 * D, 0.5)))
    G = 1 + (rng.uniform(0.08, 0.2) + rng.uniform(0.2, 0.7) * high_sun) * reach * (0.8 + 0.3 * rays) * blocks    # a blanket with streaks
    core = np.clip((0.6 - rr) / 0.3, 0, 1) * np.clip(0.5 + 0.8 * _field(X.shape, cell, rng, max(0.05 * D, 0.5)), 0, 1)
    G = G * (1 - rng.uniform(0.3, 0.8) * high_sun * core)                       # dark blotches on the floor at high sun
    G = G * np.exp(0.13 * np.clip((1.25 - rr) / 0.3, 0, 1) * _field(X.shape, cell, rng, max(0.02 * D, 0.4)))   # boulders in and on the rim
    return H, G, D


def cast_shadow(H, cell, toward_sun_deg, incidence, offset=0.0, warp=None):
    """Shadow fraction of the height field H (cell = metres per cell) on flat ground, for the sun at `toward_sun_deg`
    (image deg) and `incidence` from vertical: resampled along the sun direction, running horizon maximum, sampled
    back. `offset` moves it away from the sun (cells: a part raised above the ground); `warp` (cells, per cell) bends
    it over the terrain: higher ground catches the ray earlier."""
    ny, nx = H.shape
    t = math.radians(toward_sun_deg); sx, sy = math.cos(t), math.sin(t)
    L = int(math.hypot(ny, nx)) + 4; u = np.arange(L) - L / 2
    U, V = np.meshgrid(u, u)                                   # U along the sun direction, V across
    cx, cy = (nx - 1) / 2, (ny - 1) / 2
    Hr = ndimage.map_coordinates(H, [cy + U * sy + V * sx, cx + U * sx - V * sy], order=1, cval=0.0)
    drop = cell * math.tan(math.radians(90 - min(incidence, MAX_INCIDENCE)))   # horizon fall per cell
    M = np.full(L, -np.inf, np.float32); sh = np.zeros_like(Hr, bool)
    for j in range(L - 2, -1, -1):                             # walk away from the sun; M = highest horizon so far
        M = np.maximum(M, Hr[:, j + 1]) - drop
        sh[:, j] = M > Hr[:, j] + 1e-3
    yy, xx = np.mgrid[0:ny, 0:nx]
    du, dv = (xx - cx) * sx + (yy - cy) * sy, -(xx - cx) * sy + (yy - cy) * sx
    du = du + offset - (warp if warp is not None else 0.0)
    return ndimage.map_coordinates(sh.astype(np.float32), [dv + L / 2, du + L / 2], order=1, cval=0.0)


def terrain_warp(bg, X, Y, cx, cy, res, cell, sun_az, incidence, scale_dn, rng, limit_m):
    """How far (cells) the terrain moves a shadow, per cell of the window: elevation from the tile's shading along the
    sun direction (photoclinometry: brighter = facing the sun = falling toward it), relative to the object's foot,
    plus fine roughness. Elevation is capped: this is shading, not a terrain model."""
    i = math.radians(min(incidence, MAX_INCIDENCE)); tan_i = math.tan(i)
    hp = ndimage.gaussian_filter(bg, 1.2) - ndimage.gaussian_filter(bg, 25)
    slope = np.clip(hp / scale_dn / max(tan_i, 0.4), -0.35, 0.35)            # rising away from the sun, positive
    t = math.radians(sun_az); sx, sy = math.cos(t), math.sin(t)
    L = int(math.hypot(*X.shape)) + 4; u = (np.arange(L) - L / 2) * cell     # metres along / across the sun, from the object
    U, V = np.meshgrid(u, u)
    S = ndimage.map_coordinates(slope, [cy + (U * sy + V * sx) / res, cx + (U * sx - V * sy) / res], order=1, mode="nearest")
    G = -np.cumsum(S, axis=1) * cell; G = G - G[:, L // 2][:, None]           # elevation relative to the foot
    du, dv = X * sx + Y * sy, -X * sy + Y * sx
    g = ndimage.map_coordinates(G, [dv / cell + L / 2, du / cell + L / 2], order=1, mode="nearest")
    g = np.clip(g, -limit_m, limit_m) + 0.15 * _field(X.shape, cell, rng, 0.6)
    return g * tan_i / cell


def height_m(spec):
    """Tallest point of the object in metres (crater: 0)."""
    return max(sh["z1"] for sh in spec["shapes"])


def footprint_m(spec, scale=1.0):
    """Diameter of the object's footprint in metres (craters: the largest crater)."""
    r = 0.0
    for sh in spec["shapes"]:
        k = sh["k"]
        if k == "crater":
            return max(sh["sizes"]) * scale
        ext = {"poly": lambda: sh["r"], "cyl": lambda: sh["r"], "dome": lambda: sh["r"], "dish": lambda: sh["r"], "cone": lambda: sh["r0"],
               "legs": lambda: sh["r"] + 0.35, "canopy": lambda: sh["r"] * 1.3, "hcyl": lambda: math.hypot(sh["length"] / 2, sh["r"]),
               "box": lambda: math.hypot(sh["w"], sh["l"]) / 2}[k]()
        r = max(r, ext + math.hypot(sh.get("dx", 0.0), sh.get("dy", 0.0)))
    return 2 * r * scale


def shadow_px(spec, incidence, res, scale=1.0):
    """Longest shadow of the design in image pixels at this sun and resolution."""
    return height_m(spec) * scale * math.tan(math.radians(min(incidence, MAX_INCIDENCE))) / res


def render(spec, bg, res, cx, cy, sun_az, incidence, body, rng, scale, alb_k, level, scale_dn):
    """Add a rendered object at (cx, cy) of the background `bg` (float, DN). Returns (unclipped image, mask, info).
    Rendered on a FINE x oversampled window around the object, padded by its longest possible shadow."""
    cell = res / FINE
    rot = rng.uniform(0, 2 * np.pi)
    is_crater = spec["cls"] == "impact-crater"
    tan_i = math.tan(math.radians(min(incidence, MAX_INCIDENCE)))
    if is_crater:
        D = float(rng.choice(spec["shapes"][0]["sizes"])) * scale; reach = 2.2 * D
    else:
        reach = footprint_m(spec, scale) * 0.5 + 1.0
    pad = int(math.ceil((reach + height_m(spec) * scale * tan_i) / res)) + 3
    x0, x1 = max(0, int(cx) - pad), min(SIZE, int(cx) + pad + 1)
    y0, y1 = max(0, int(cy) - pad), min(SIZE, int(cy) + pad + 1)
    fx = (x0 - 0.5 + (np.arange((x1 - x0) * FINE) + 0.5) / FINE - cx) * res      # cell centres, metres from the object
    fy = (y0 - 0.5 + (np.arange((y1 - y0) * FINE) + 0.5) / FINE - cy) * res
    X, Y = np.meshgrid(fx, fy)
    floor, soft, texture = natural_floor(bg, body, rng), rng.uniform(*SHADOW_SOFT_PX[body]), rng.uniform(*SHADOW_TEXTURE)
    body_soft, top = rng.uniform(*BODY_SOFT_PX[body]), rng.uniform(*BODY_CONTRAST_DN[body]) * 1.6 if is_crater else body_top(body, level, rng)
    H, A, glint = np.zeros_like(X), np.zeros_like(X), np.ones_like(X)
    G, ground = np.ones_like(X), np.zeros_like(X)
    solid, raised = np.zeros_like(X), []
    if is_crater:
        ground, G, _ = _crater(D, X, Y, rng, cell, incidence)
    else:
        for sh in spec["shapes"]:
            m, z, z0, alb = _part(sh, X, Y, rot, scale, rng, cell)
            A = np.where(m & (z >= H), alb, A); glint = np.where(m & (z >= H), sh.get("glint", 1.0), glint); H = np.maximum(H, z)
            if z0 <= 0.05:
                solid = np.maximum(solid, z)
            else:
                raised.append((np.where(m, np.maximum(z - z0, 0.0), 0.0), z0))
    obj = A > 0
    # shadows: every part from its own height, bent over the terrain, softened by the sun's width (0.53 deg)
    S = np.zeros_like(X)
    if not is_crater:
        hmax = float(H.max())
        warp = terrain_warp(bg, X, Y, cx, cy, res, cell, sun_az, incidence, scale_dn, rng, min(3.0, 0.5 * hmax + 0.5)) if incidence > 25 else None
        S = np.maximum(S, cast_shadow(solid, cell, sun_az, incidence, warp=warp))
        for h, z0 in raised:
            S = np.maximum(S, cast_shadow(h, cell, sun_az, incidence, offset=z0 * tan_i / cell, warp=warp))
        pen = 0.0046 * hmax * tan_i / cell
        if pen > 0.5:
            S = ndimage.gaussian_filter(S, pen)
    else:
        S = cast_shadow(ground, cell, sun_az, incidence)
    # shading: facets and glints on hardware, slopes on the crater; walls are not seen from orbit
    t, i = math.radians(sun_az), math.radians(min(incidence, MAX_INCIDENCE))
    sv = (math.sin(i) * math.cos(t), math.sin(i) * math.sin(t), math.cos(i))
    surf = H if not is_crater else ground
    gy, gx = np.gradient(surf, cell)
    lam = np.minimum(np.clip((-gx * sv[0] - gy * sv[1] + sv[2]) / np.sqrt(gx ** 2 + gy ** 2 + 1), 0, None) / max(sv[2], 0.05), 6.0)
    lam = np.where(obj & (gx ** 2 + gy ** 2 > 7.5), 1.0, lam)
    lit_ground = (lam if is_crater else 1.0) * G                               # the shadow is blended in below, softly
    if obj.any():
        self_shadow = cast_shadow(H, cell, sun_az, incidence) > 0.5
        glints = np.exp(glint * (0.35 * _field(X.shape, cell, rng, 0.35) + 0.3 * _field(X.shape, cell, rng, 1.0)))
        rad = A * alb_k * glints * np.where(self_shadow, 0.06, lam)
    else:
        rad = np.zeros_like(X)
    wy, wx = y1 - y0, x1 - x0
    down = lambda a: np.asarray(a, np.float32).reshape(wy, FINE, wx, FINE).mean(axis=(1, 3))
    win = (slice(y0, y1), slice(x0, x1))
    cov, gmean, orad = np.zeros((SIZE, SIZE), np.float32), np.ones((SIZE, SIZE), np.float32), np.zeros((SIZE, SIZE), np.float32)
    shc, bowl, ejecta, shf = (np.zeros((SIZE, SIZE), np.float32) for _ in range(4))
    cov[win] = down(obj)
    gmean[win] = np.where(cov[win] < 0.999, down(np.where(obj, 0.0, lit_ground)) / np.maximum(1 - cov[win], 1e-6), 1.0)
    shf[win] = np.where(cov[win] < 0.999, down(np.where(obj, 0.0, S)) / np.maximum(1 - cov[win], 1e-6), 0.0)
    orad[win] = down(np.where(obj, rad, 0.0))
    shc[win] = down((S > 0.5) & ~obj)
    bowl[win] = down((ground < -0.02) & ~obj); ejecta[win] = down((np.abs(G - 1) > 0.02) & ~obj)
    soft_shadow = np.clip(ndimage.gaussian_filter(shf, soft) / SHADOW_CORE, 0, 1)   # soft edge (rough ground, scattered light), dark centre
    ground_dn = to_dn(gmean, bg, floor, scale_dn, top)    # the terrain keeps its texture
    in_shadow = floor + texture * (bg - level)            # a little of it still shows inside the shadow
    ground_dn = ground_dn * (1 - soft_shadow) + in_shadow * soft_shadow
    obj_dn = to_dn(orad / np.maximum(cov, 1e-6), level, floor, scale_dn, top)
    soft_cov = ndimage.gaussian_filter(cov, body_soft)    # no body has a clean outline either
    soft_obj = ndimage.gaussian_filter(obj_dn * cov, body_soft) / np.maximum(soft_cov, 1e-3)
    out = ground_dn * (1 - soft_cov) + soft_obj * soft_cov
    mask = np.zeros((SIZE, SIZE), np.uint8)
    mask[ejecta > 0.3] = 3; mask[(shc > 0.3) | (soft_shadow > 0.3)] = 2; mask[(cov > 0.3) | (bowl > 0.3)] = 1
    if is_crater:
        info = dict(width_m=round(D, 2), height_m=0.0, shadow_length_m=0.0)
    else:
        info = dict(width_m=round(float(2 * np.hypot(X[obj], Y[obj]).max()), 2) if obj.any() else 0.0, height_m=round(float(H.max()), 2),
                    shadow_length_m=round(float(H.max()) * tan_i, 1))
    info.update(rotation_deg=round(math.degrees(rot), 1), shadow_soft_px=round(soft, 2), shadow_floor_dn=round(floor, 1), body_contrast_dn=round(top, 1))
    return out, mask, info


# ── context traces ────────────────────────────────────────────────────────────────────────────────────────────
def _path(rng, x, y, n, step, heading, wiggle):
    pts = [(x, y)]
    for _ in range(n):
        heading += rng.normal(0, wiggle); x += step * math.cos(heading); y += step * math.sin(heading); pts.append((x, y))
    return pts


def context_layer(kind, spec, cx, cy, res, size_m, rng, body, incidence, scale_dn):
    """Radiance factor (SIZE x SIZE) of a blast zone, impact scar, wheel tracks or footpaths."""
    f = np.ones((SIZE, SIZE), np.float32)
    if kind in ("blast-zone", "impact-scar"):
        yy, xx = np.mgrid[0:SIZE, 0:SIZE]; rho = np.hypot(xx - cx, yy - cy)
        if kind == "blast-zone":
            R = size_m / 2 * rng.uniform(3, 8) / res
            amp = rng.uniform(0.03, 0.18) if body == "Moon" else rng.uniform(-0.15, 0.06)
        else:
            R = rng.uniform(5, 11) / res; amp = rng.uniform(-0.35, -0.1)
        noise = ndimage.gaussian_filter(rng.normal(0, 1, (SIZE, SIZE)), max(R / 5, 1)); noise /= noise.std() + 1e-6
        f = f * (1 + amp * np.clip(1 - rho / (R * (1 + 0.3 * noise)), 0, 1) ** 1.6)   # fades out: no visible outline
        if kind == "blast-zone" and incidence < 60 and rng.random() < 0.6:       # high sun: the disturbed ground right around
            R1 = rng.uniform(4, 9) / res; fine = ndimage.gaussian_filter(rng.normal(0, 1, (SIZE, SIZE)), max(R1 / 3, 1)); fine /= fine.std() + 1e-6
            f = f * (1 - rng.uniform(0.1, 0.35) * min(1.0, (60 - incidence) / 40) * np.exp(-(rho / (R1 * (1 + 0.25 * fine))) ** 2))   # goes dark (LM at i < 50)
        return f
    up = 4; big = Image.new("F", (SIZE * up, SIZE * up), 0.0); d = ImageDraw.Draw(big)
    if kind == "tracks":
        gauge = spec["tracks"] / res * up; width = max(rng.uniform(0.3, 0.5) / res * up, 1)
        length = rng.uniform(15, 120) / res; step = 2.0
        pts = _path(rng, cx, cy, int(length / step), step, rng.uniform(0, 2 * np.pi), 0.06)
        for side in (-0.5, 0.5):
            line = []
            for k in range(len(pts) - 1):
                (x0, y0), (x1, y1) = pts[k], pts[k + 1]; hd = math.atan2(y1 - y0, x1 - x0)
                line.append((x0 * up - side * gauge * math.sin(hd), y0 * up + side * gauge * math.cos(hd)))
            d.line(line, fill=1.0, width=int(round(width)))
        depth = min(rng.uniform(0.05, 0.16), rng.uniform(15, 45) / scale_dn) * (1 if body == "Moon" or rng.random() < 0.7 else -0.6)
    else:                                                   # footpaths: thin winding trails out of the lander
        width = max(0.5 / res * up, 1)
        for _ in range(rng.integers(1, 5)):
            pts = _path(rng, cx, cy, int(rng.uniform(20, 150) / res / 2), 2.0, rng.uniform(0, 2 * np.pi), 0.12)
            d.line([(x * up, y * up) for x, y in pts], fill=1.0, width=int(round(width)))
        depth = min(rng.uniform(0.04, 0.12), rng.uniform(10, 30) / scale_dn)
    cov = np.asarray(big, np.float32).reshape(SIZE, up, SIZE, up).mean(axis=(1, 3))
    return f * (1 - depth * cov)


def add_context(work, ctx, spec, cx, cy, res, size_m, rng, body, incidence, level, u):
    """Add blast zones, scars, tracks and footpaths to the ground, in DN, within what the ground level allows."""
    for c in ctx:
        work = work + np.clip(u * (context_layer(c, spec, cx, cy, res, size_m, rng, body, incidence, u) - 1), -0.45 * level, 0.6 * level)
    return work


# ── real sprites ──────────────────────────────────────────────────────────────────────────────────────────────
def build_sprites():
    """Cut-outs of real hardware: relative radiance of the bright body and the black shadow touching it (no halo), a soft
    alpha, the source's m/px and incidence, and the sun direction taken from its own body-to-shadow offset."""
    sites = {r["site_id"]: r for r in csv.DictReader(SITES.open())}
    meta = {r["file_name"]: r for r in csv.DictReader((CROPS / "metadata.csv").open())}
    coco = json.load(COCO.open()); cat = {c["id"]: c["name"] for c in coco["categories"]}; ims = {i["id"]: i for i in coco["images"]}
    sprites, seen, crops = [], set(), {}
    for a in sorted(coco["annotations"], key=lambda a: a["id"]):
        oid = cat[a["category_id"]]; f = ims[a["image_id"]]["file_name"]; m = meta[f]
        if a["confidence"] != "high" or oid not in sites or sites[oid]["object_type"] not in SPRITE_TYPES:
            continue
        x, y, w, h = a["bbox"]
        key = (m["frame_id"], oid, int(m["crop_x0"]) + x // 4, int(m["crop_y0"]) + y // 4)
        if key in seen or max(w, h) > 120:
            continue
        seen.add(key)
        inc = float(m["incidence_deg"])
        if f not in crops:
            crops.clear()
            img = np.asarray(Image.open(CROPS / f).convert("L"), np.float32)
            crops[f] = (img, radiance_scale(img, inc, m["body"]), shading_azimuth(img, m["body"])[0], float(np.percentile(img, 0.5)))
        img, (_, scale_dn), shade_az, black = crops[f]
        mg = max(6, round(0.35 * max(w, h)))
        x0, y0, x1, y1 = max(0, x - mg), max(0, y - mg), min(2048, x + w + mg), min(2048, y + h + mg)
        p = img[y0:y1, x0:x1]
        inside = np.zeros(p.shape, bool); inside[y - y0:y - y0 + h, x - x0:x - x0 + w] = True
        ring = ~ndimage.binary_dilation(inside, iterations=2)
        if ring.sum() < 20:
            continue
        g = float(np.median(p[ring])); n = float(np.median(np.abs(p[ring] - g)) * 1.4826) + 1e-3
        bright = (p > g + max(2.5 * n, 0.2 * g)) & inside
        dark = (p < g - max(2.5 * n, 0.45 * g)) & inside
        lab, k = ndimage.label(bright | dark, structure=np.ones((3, 3)))
        if k == 0 or not bright.any():
            continue
        counts = ndimage.sum(bright, lab, range(1, k + 1))
        core = lab == (int(np.argmax(counts)) + 1)
        if core.sum() < 4 or not (core & bright).any() or (core & dark).sum() > 8 * (core & bright).sum() + 20:
            continue                                                             # nothing bright, or mostly a dark halo
        yb, xb = np.nonzero(core & bright); yd, xd = np.nonzero(core & dark)
        sun = math.degrees(math.atan2(yb.mean() - yd.mean(), xb.mean() - xd.mean())) if len(xd) >= 3 else shade_az
        rad = np.where(p < g, (p - black) / max(g - black, 3.0), 1 + (p - g) / scale_dn)
        rad = np.where(p >= 253, np.maximum(rad, 3.0), rad)
        alpha = np.clip(ndimage.gaussian_filter(ndimage.binary_dilation(core).astype(np.float32), 0.7) * 1.3, 0, 1)
        sprites.append(dict(rad=rad.astype(np.float32), alpha=alpha.astype(np.float32), body=m["body"], oid=oid,
                            cls=SPRITE_TYPES[sites[oid]["object_type"]], res=float(m["resolution_m_px"]),
                            incidence=inc, sun=sun, frame=m["frame_id"], source=f"{f}#{x},{y},{w},{h}", name=sites[oid]["object"]))
    return sprites


def paste(bg, sp, cx, cy, res, sun_az, contrast, mirror, level, scale_dn, floor, top):
    """Paste sprite `sp` at (cx, cy): rotated so its sun azimuth becomes `sun_az`, scaled to `res`, optionally mirrored
    across the sun axis, in relative radiance (see to_dn). Returns the unclipped image."""
    s = sp["res"] / res
    d = math.radians(sun_az - sp["sun"])
    a2 = math.radians(sun_az)
    h, w = sp["rad"].shape
    half = int(math.ceil(max(h, w) * s * 0.75)) + 2
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float32)
    if mirror:                                                 # reflect across the sun axis (keeps the sun direction)
        c2, s2 = math.cos(2 * a2), math.sin(2 * a2)
        xx, yy = c2 * xx + s2 * yy, s2 * xx - c2 * yy
    cd, sd = math.cos(-d), math.sin(-d)
    sx, sy = (cd * xx - sd * yy) / s + (w - 1) / 2, (sd * xx + cd * yy) / s + (h - 1) / 2
    rad = ndimage.map_coordinates(sp["rad"], [sy, sx], order=1, cval=1.0)
    al = ndimage.map_coordinates(sp["alpha"], [sy, sx], order=1, cval=0.0)
    out = bg.copy(); mask = np.zeros(bg.shape, np.uint8)
    y0, x0 = int(round(cy)) - half, int(round(cx)) - half
    ys, xs = slice(max(0, y0), min(SIZE, y0 + 2 * half + 1)), slice(max(0, x0), min(SIZE, x0 + 2 * half + 1))
    sub = (slice(ys.start - y0, ys.stop - y0), slice(xs.start - x0, xs.stop - x0))
    rad, al = rad[sub], al[sub]
    out[ys, xs] = bg[ys, xs] * (1 - al) + to_dn(1 + contrast * (rad - 1), level, floor, scale_dn, top) * al
    m = np.zeros(rad.shape, np.uint8); m[(al > 0.3) & (rad >= 1)] = 1; m[(al > 0.3) & (rad < 1)] = 2
    mask[ys, xs] = m
    return out, mask, s, round(math.degrees(d), 1)


def camera(bg, out, mask, psf, rng, already=PIXEL_SIGMA):
    """What the camera does to what was added: its blur (minus what the rendering already has), the tile's noise on the
    object and inside its shadow. Clipping to 8 bits comes after."""
    extra = math.sqrt(max(psf ** 2 - already ** 2, 0.05 ** 2))
    diff = ndimage.gaussian_filter(out - bg, extra)
    cover = ndimage.gaussian_filter(((mask == 1) | (mask == 2)).astype(np.float32), 0.5)
    shot = np.sqrt(np.clip((bg + diff) / max(float(np.median(bg)), 1.0), 0.02, 1.0))   # shot noise: weaker in the dark of a shadow
    return bg + diff + rng.normal(0, noise_sigma(bg), bg.shape) * shot * cover


# ── one image ─────────────────────────────────────────────────────────────────────────────────────────────────
MIN_DN_LEVELS = 20                         # tiles with fewer grey levels (faint HiRISE EDRs) are blocky: no background


def load_tiles():
    tiles = {}
    for body, d in TILES.items():
        rows = list(csv.DictReader((d / "metadata.csv").open()))
        tiles[body] = []
        for r in rows:
            a = np.asarray(Image.open(d / r["file_name"]).convert("L"))
            if len(np.unique(a)) < MIN_DN_LEVELS or ((a > 3) & (a < 252)).mean() < 0.5:
                continue
            tiles[body].append(dict(rel=r["file_name"], path=d / r["file_name"], split=r["split"], landform=r["landform"], body=body,
                                    frame=r["frame_id"], res=float(r["resolution_m_px"]), incidence=float(r["incidence_deg"] or 45),
                                    lat=float(r["lat"]) if r.get("lat") else None,
                                    ls=float(r["solar_longitude_ls"]) if r.get("solar_longitude_ls") else None))
        print(f"  {body}: {len(tiles[body])} of {len(rows)} tiles usable as backgrounds", flush=True)
    return tiles


def background(t, rng, allow_coarser):
    """512x512 crop of tile t (on the Moon sometimes 1.5-2x larger and resampled: a coarser camera). Up to six
    draws to avoid crops that are mostly shadow (grazing sun), where no object could show."""
    a = np.asarray(Image.open(t["path"]).convert("L"), np.float32)
    H, W = a.shape
    f = 1.0
    if allow_coarser and min(H, W) >= 2 * SIZE:
        f = float(rng.choice([1.0, 1.0, 1.0, 1.5, 2.0]))
    side = int(SIZE * f)
    for _ in range(6):
        y0, x0 = int(rng.integers(0, H - side + 1)), int(rng.integers(0, W - side + 1))
        crop = a[y0:y0 + side, x0:x0 + side]
        if (crop < SHADOW_DN).mean() < 0.4:
            break
    if f != 1.0:
        crop = np.asarray(Image.fromarray(crop).resize((SIZE, SIZE), Image.BOX), np.float32)
    return crop, f, x0, y0, side


def contexts(spec, rng):
    """What comes with the object. Real landers nearly always show a blast zone, heat shields and backshells a scar."""
    out = []
    if spec["cls"] == "lander" and rng.random() < 0.7:
        out.append("blast-zone")
    if spec.get("tracks") and rng.random() < 0.8:
        out.append("tracks")
    if spec.get("context") == "footpaths" and rng.random() < 0.6:
        out.append("footpaths")
    if spec.get("context") == "impact-scar" and rng.random() < 0.6:
        out.append("impact-scar")
    return out


def shadow_extent(mask, az):
    """How far the largest shadow reaches along the sun, in pixels. Measured on the mask, so real cut-outs have it too."""
    lab, n = ndimage.label(mask == 2)
    if not n:
        return 0
    ys, xs = np.nonzero(lab == 1 + int(np.argmax(np.bincount(lab[lab > 0])[1:])))
    return int(np.ptp(xs * math.cos(math.radians(az)) + ys * math.sin(math.radians(az))) + 1)


def make(item, sprites, info_of, rng):
    t, body, kind = item["tile"], item["body"], item["kind"]
    bg, f, bx0, by0, side = background(t, rng, body == "Moon")
    res = t["res"] * f
    ti = info_of(t)
    az, inc = ti["az"], t["incidence"]
    cx, cy = pick_spot(bg, rng)
    level, scale_dn = radiance_scale(bg, inc, body)
    scale_dn *= rng.uniform(0.85, 1.15)
    psf = math.sqrt((ti["psf"] / f) ** 2 + PIXEL_SIGMA ** 2) if f != 1.0 else ti["psf"]
    info = dict(sprite_source="", rotation_deg="", mirrored="", contrast_factor="", albedo_factor="", context="none",
                scale_factor="", width_m="", height_m="", shadow_length_m="", shadow_soft_px="", shadow_floor_dn="", body_contrast_dn="")
    if kind == "rendered":
        hid = item["hardware"]; spec = HARDWARE[hid]
        scale, alb_k = rng.uniform(0.92, 1.08), rng.uniform(0.7, 1.3)
        ctx = contexts(spec, rng)
        work = add_context(bg.copy(), ctx, spec, cx, cy, res, footprint_m(spec, scale), rng, body, inc, level, scale_dn)
        out, mask, r = render(spec, work, res, cx, cy, az, inc, body, rng, scale, alb_k, level, scale_dn)
        if ctx:
            mask[(mask == 0) & (np.abs(work - bg) > 2)] = 3
        info.update(r, scale_factor=round(scale, 3), albedo_factor=round(alb_k, 3), context="+".join(ctx) or "none")
        era, cls, name, hw = spec["era"], spec["cls"], spec["name"], hid
        out = camera(bg, out, mask, psf, rng)
    else:
        cands = [s for s in sprites if s["body"] == body and abs(s["incidence"] - inc) <= 12 and s["res"] <= res * 1.02]
        if not cands:
            return None
        sp = cands[rng.integers(len(cands))]
        era, cls, name, hw = "past", sp["cls"], sp["name"], sp["oid"]
        contrast, mirror = rng.uniform(0.8, 1.2), bool(rng.random() < 0.5)
        floor, top = natural_floor(bg, body, rng), body_top(body, level, rng)
        out, mask, s, rot = paste(bg, sp, cx, cy, res, az, contrast, mirror, level, scale_dn, floor, top)
        info.update(sprite_source=sp["source"], rotation_deg=rot, mirrored="yes" if mirror else "no",
                    contrast_factor=round(contrast, 3), scale_factor=round(s, 3), shadow_floor_dn=round(floor, 1), body_contrast_dn=round(top, 1))
        # the cut-out already has its source camera's blur, shrunk by s
        out = camera(bg, out, mask, psf, rng, already=min(0.65 * s, psf))
    gain, gamma = rng.uniform(0.95, 1.05), rng.uniform(0.92, 1.08)
    cam = lambda a: np.clip(255 * (np.clip(a, 0, 255) / 255) ** gamma * gain, 0, 255).astype(np.uint8)
    mask[(mask == 0) & (np.abs(cam(out).astype(np.int16) - cam(bg).astype(np.int16)) > 2)] = 3   # nothing changes outside the mask
    ys, xs = np.nonzero((mask == 1) | (mask == 2))
    if len(xs) == 0:
        return None
    oy, ox = np.nonzero(mask == 1)
    obj_px = int(max(np.ptp(ox), np.ptp(oy)) + 1) if len(ox) else 0
    if kind != "rendered":
        info["width_m"] = round(obj_px * res, 2)
    row = dict(split=t["split"], body=body, instrument=CAMERA[body], kind=kind, era=era, object_class=cls,
               hardware_id=hw, hardware_name=name, x_center=round(cx, 1), y_center=round(cy, 1),
               box_x_min=int(xs.min()), box_y_min=int(ys.min()), box_x_max=int(xs.max()), box_y_max=int(ys.max()),
               object_px=obj_px, shadow_length_px=shadow_extent(mask, az), shadow_area_px=int((mask == 2).sum()), resolution_m_px=round(res, 4),
               camera_downscale=f, incidence_deg=round(inc, 2), sun_azimuth_image_deg=round(az, 1),
               sun_azimuth_source=ti["source"], sun_azimuth_confidence=round(ti["conf"], 3), radiance_scale_dn=round(scale_dn, 1),
               psf_sigma_px=round(psf, 2), noise_sigma=round(noise_sigma(bg), 2),
               gain=round(gain, 3), gamma=round(gamma, 3), landform=t["landform"], background_file=t["rel"],
               background_x0=bx0, background_y0=by0, background_side_px=side, frame_id=t["frame"], seed=item["seed"],
               shadowed_fraction=round(float((bg < SHADOW_DN).mean()), 3), **info)
    return cam(out), cam(bg), mask, row


# ── check against the real hardware ───────────────────────────────────────────────────────────────────────────
REAL_DESIGN = {"apollo11_lm": "apollo-lm-descent-stage", "apollo12_lm": "apollo-lm-descent-stage", "apollo14_lm": "apollo-lm-descent-stage",
               "apollo15_lm": "apollo-lm-descent-stage", "apollo16_lm": "apollo-lm-descent-stage", "apollo17_lm": "apollo-lm-descent-stage",
               "surveyor1": "surveyor", "surveyor3": "surveyor", "surveyor6": "surveyor", "surveyor7": "surveyor", "change3": "change3-lander",
               "blue_ghost1": "blue-ghost", "lunokhod1": "lunokhod", "apollo15_lrv": "apollo-lrv", "apollo16_lrv": "apollo-lrv", "apollo17_lrv": "apollo-lrv",
               "yutu": "yutu", "viking1": "viking-lander", "viking2": "viking-lander", "pathfinder": "pathfinder", "phoenix": "phoenix-insight",
               "insight": "phoenix-insight", "spirit": "mer-rover", "opportunity": "mer-rover", "curiosity_rover": "msl-rover", "perseverance_rover": "msl-rover",
               "insight_heatshield": "heat-shield", "perseverance_heatshield": "heat-shield", "insight_backshell": "backshell-parachute",
               "perseverance_backshell": "backshell-parachute", "apollo13_sivb": "spacecraft-impact", "apollo14_sivb": "spacecraft-impact",
               "apollo15_sivb": "spacecraft-impact", "apollo16_sivb": "spacecraft-impact", "apollo17_sivb": "spacecraft-impact",
               "ranger7": "spacecraft-impact", "ranger8": "spacecraft-impact", "ranger9": "spacecraft-impact"}


def shadow_softness(p, region, g):
    """Edge width (10-90 % rise, px) of the largest dark blob in `region`, and its inside: median DN, share at zero. None if
    there is no real shadow. The bright body touching the shadow does not count as edge."""
    if not region.any():
        return None
    floor = float(np.percentile(p[region], 2))
    if g - floor < 25:
        return None
    lab, k = ndimage.label((p < (g + floor) / 2) & region)
    if k == 0:
        return None
    sizes = ndimage.sum(np.ones_like(p), lab, range(1, k + 1)); i = int(np.argmax(sizes)) + 1
    if sizes[i - 1] < 8:
        return None
    dark = lab == i
    edge = dark & ~ndimage.binary_erosion(dark) & ~ndimage.binary_dilation(p > g + 0.15 * max(g, 30), iterations=2)
    if edge.sum() < 5:
        return None
    width = (g - floor) / np.maximum(np.hypot(*np.gradient(p))[edge], 1e-3) * 0.8
    inner = ndimage.binary_erosion(dark)
    q = p[inner] if inner.sum() >= 3 else p[dark]
    return dict(edge_px=float(np.median(width)), inside_dn=float(np.median(q)), at_zero=float((q <= 1).mean()))


def body_softness(p, region, g):
    """Of the largest bright blob in `region`: share of its pixels at saturation, its median DN, its edge width (px)."""
    lab, k = ndimage.label((p > g + max(0.25 * g, 20)) & region)
    if k == 0:
        return None
    sizes = ndimage.sum(np.ones_like(p), lab, range(1, k + 1)); i = int(np.argmax(sizes)) + 1
    if sizes[i - 1] < 3:
        return None
    b = lab == i; edge = b & ~ndimage.binary_erosion(b)
    top = float(np.percentile(p[b], 90))
    width = (top - g) / np.maximum(np.hypot(*np.gradient(p))[edge], 1e-3) * 0.8
    return dict(sat_share=float((p[b] >= 250).mean()), body_dn=float(np.median(p[b])), body_edge_px=float(np.median(width)))


def measure_object(a, box, res):
    """Peak DN, shadow floor DN, bright body and shadow size of the object inside box (x, y, w, h). The shadow is the dark
    run along the body-to-shadow ray, so a halo does not count. The same code measures real and rendered objects."""
    x, y, w, h = box; pad = 40
    x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(a.shape[1], x + w + pad), min(a.shape[0], y + h + pad)
    p = a[y0:y1, x0:x1]; ins = np.zeros(p.shape, bool); ins[y - y0:y - y0 + h, x - x0:x - x0 + w] = True
    g = float(np.median(p))
    out = dict(ground_dn=g, peak_dn=float(p[ins].max()), floor_dn=float(p[ins].min()), body_m=0.0, tip_m=0.0, shadow_w_m=0.0, edge_px="", inside_dn="", at_zero="", sat_share="", body_dn="", body_edge_px="")
    for soft in (shadow_softness(p, ins, g), body_softness(p, ins, g)):
        if soft:
            out.update(soft)
    hi = ins & (p >= max(float(np.percentile(p[ins], 92)), g * 1.15)); lo = ins & (p <= min(float(np.percentile(p[ins], 15)), g * 0.6))
    if hi.any():
        lab, k = ndimage.label(hi); big = lab == (int(np.argmax(ndimage.sum(hi, lab, range(1, k + 1)))) + 1)
        ys, xs = np.nonzero(big); out["body_m"] = float(max(np.ptp(xs), np.ptp(ys)) + 1) * res
    if not hi.any() or lo.sum() < 2:
        return out
    yh, xh = np.nonzero(hi); yl, xl = np.nonzero(lo)
    t = math.atan2(yl.mean() - yh.mean(), xl.mean() - xh.mean()); ux, uy = math.cos(t), math.sin(t)
    d = np.arange(0, 120, 0.5); cx, cy = xh.mean(), yh.mean()
    vals = np.min([ndimage.map_coordinates(p, [cy + d * uy + k * ux, cx + d * ux - k * uy], order=1, mode="nearest") for k in (-1, 0, 1)], axis=0)
    dark = vals < g * 0.55
    if dark.any() and np.argmax(dark) * 0.5 <= 10:
        en = int(np.argmax(dark))
        while en + 1 < len(dark) and (dark[en + 1] or (en + 3 < len(dark) and dark[en + 3])):
            en += 1
        out["tip_m"] = en * 0.5 * res
        st = int(np.argmax(dark)); out["shadow_w_m"] = float(lo.sum()) * res * res / max((en - st) * 0.5 * res, res)
    return out


def realism(sheet=True):
    """Every real hardware frame against its design rendered at the same sun and m/px on terrain of the same crop.
    Writes realism_report.csv and contact-sheets/real-vs-model.jpg; prints medians per design."""
    sites = {r["site_id"]: r for r in csv.DictReader(SITES.open())}
    meta = {r["file_name"]: r for r in csv.DictReader((CROPS / "metadata.csv").open())}
    coco = json.load(COCO.open()); cat = {c["id"]: c["name"] for c in coco["categories"]}; ims = {i["id"]: i for i in coco["images"]}
    rows, seen, tiles_ = [], set(), defaultdict(list)
    for an in sorted(coco["annotations"], key=lambda q: q["id"]):
        oid = cat[an["category_id"]]
        if oid not in REAL_DESIGN or an["confidence"] != "high":
            continue
        f = ims[an["image_id"]]["file_name"]; m = meta[f]
        if (m["frame_id"], oid) in seen:
            continue
        seen.add((m["frame_id"], oid))
        img = np.asarray(Image.open(CROPS / f).convert("L"), np.float32)
        res, inc, body = float(m["resolution_m_px"]), float(m["incidence_deg"]), m["body"]
        x, y, w, h = an["bbox"]; hx, hy = x + w / 2, y + h / 2
        windows = [(bx, by) for bx in range(0, 1537, 512) for by in range(0, 1537, 512) if math.hypot(bx + 256 - hx, by + 256 - hy) > 450]
        windows = [q for q in windows if (img[q[1]:q[1] + 512, q[0]:q[0] + 512] < SHADOW_DN).mean() < 0.1]
        if not windows:
            continue
        rng = np.random.default_rng(an["id"])
        bx, by = windows[int(rng.integers(len(windows)))]
        bg = img[by:by + 512, bx:bx + 512].copy()
        az = sun_azimuth(img, body, float(sites[oid]["lat"]), inc)[0]
        spec = HARDWARE[REAL_DESIGN[oid]]
        cx, cy = pick_spot(bg, rng); level, scale_dn = radiance_scale(bg, inc, body)
        work = add_context(bg.copy(), contexts(spec, rng), spec, cx, cy, res, footprint_m(spec), rng, body, inc, level, scale_dn)
        out, mask, _ = render(spec, work, res, cx, cy, az, inc, body, rng, 1.0, 1.0, level, scale_dn)
        model = np.clip(camera(bg, out, mask, tile_psf(img), rng), 0, 255)
        ys, xs = np.nonzero((mask == 1) | (mask == 2))
        if len(xs) == 0:
            continue
        box = (int(xs.min()), int(ys.min()), int(np.ptp(xs)) + 1, int(np.ptp(ys)) + 1)
        a, b = measure_object(img, an["bbox"], res), measure_object(model, box, res)
        rows.append(dict(design=REAL_DESIGN[oid], object=oid, frame=m["frame_id"], body=body, resolution_m_px=res, incidence_deg=inc,
                         **{f"real_{k}": round(v, 2) if v != "" else "" for k, v in a.items()}, **{f"model_{k}": round(v, 2) if v != "" else "" for k, v in b.items()}))
        z = 40
        cut = lambda q, ax, ay: q[int(np.clip(ay - z, 0, q.shape[0] - 2 * z)):int(np.clip(ay - z, 0, q.shape[0] - 2 * z)) + 2 * z, int(np.clip(ax - z, 0, q.shape[1] - 2 * z)):int(np.clip(ax - z, 0, q.shape[1] - 2 * z)) + 2 * z]
        tiles_[REAL_DESIGN[oid]].append((cut(img, hx, hy), cut(model, cx, cy), oid, m["frame_id"], res, inc))
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "realism_report.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]), lineterminator="\n"); w.writeheader(); w.writerows(rows)
    print(f"{'design':<26}{'n':>3}{'sun':>5}{'m/px':>6} | shadow tip m  real/model | shadow width m | peak DN | floor DN | bright body m")
    by = defaultdict(list)
    for r in rows:
        by[r["design"]].append(r)
    for d, v in sorted(by.items()):
        med = lambda k: float(np.median([q[k] for q in v]))
        print(f"{d:<26}{len(v):3d}{med('incidence_deg'):5.0f}{med('resolution_m_px'):6.2f} | {med('real_tip_m'):6.1f} /{med('model_tip_m'):5.1f}      | {med('real_shadow_w_m'):4.1f} /{med('model_shadow_w_m'):4.1f}    | "
              f"{med('real_peak_dn'):3.0f}/{med('model_peak_dn'):3.0f} | {med('real_floor_dn'):3.0f}/{med('model_floor_dn'):3.0f}  | {med('real_body_m'):4.1f} /{med('model_body_m'):4.1f}")
    print("shadows, hardware only (not craters), real / model:")
    for body in ("Moon", "Mars"):
        v = [r for r in rows if r["body"] == body and r["design"] != "spacecraft-impact"]
        med = lambda k: float(np.median([q[k] for q in v if q[k] != ""] or [float("nan")]))
        print(f"  {body}: edge width {med('real_edge_px'):.2f} / {med('model_edge_px'):.2f} px · inside DN {med('real_inside_dn'):.0f} / {med('model_inside_dn'):.0f} · at zero {med('real_at_zero'):.2f} / {med('model_at_zero'):.2f}")
        print(f"        bodies: saturated {med('real_sat_share'):.2f} / {med('model_sat_share'):.2f} · median DN {med('real_body_dn'):.0f} / {med('model_body_dn'):.0f} · edge {med('real_body_edge_px'):.2f} / {med('model_body_edge_px'):.2f} px")
    if sheet:
        f12 = _font(12); s = 3; P = 80 * s; items = []
        for d, v in sorted(tiles_.items()):
            v = sorted(v, key=lambda q: q[5]); items += [v[k] for k in sorted({0, len(v) // 2, len(v) - 1})]
        cols = 4; nr = math.ceil(len(items) / cols)
        im = Image.new("RGB", (cols * (2 * P + 18) + 8, 40 + nr * (P + 34)), (12, 22, 40)); dr = ImageDraw.Draw(im)
        dr.text((8, 12), "XENARCH · REAL HARDWARE | RENDERED DESIGN, SAME SUN AND M/PX, ON TERRAIN OF THE SAME CROP (ZOOM X3)", fill=(240, 136, 78), font=_font(14))
        for k, (a, b, oid, fr, res, inc) in enumerate(items):
            x, y = 8 + (k % cols) * (2 * P + 18), 40 + (k // cols) * (P + 34)
            for j, q in enumerate((a, b)):
                im.paste(Image.fromarray(np.clip(q, 0, 255).astype(np.uint8)).resize((P, P), Image.NEAREST), (x + j * (P + 2), y))
            dr.text((x, y + P + 3), f"REAL | MODEL  {oid}", fill=(240, 136, 78), font=f12); dr.text((x, y + P + 17), f"{fr} · {res} m/px · sun {inc:.0f} deg", fill=(140, 150, 170), font=f12)
        (OUT / "contact-sheets").mkdir(exist_ok=True); im.save(OUT / "contact-sheets/real-vs-model.jpg", quality=90)
    return rows


# ── plan, run, sheets ─────────────────────────────────────────────────────────────────────────────────────────
def plan(tiles, sprites, per_tile, sprite_frac, seed, val_extra=2):
    """Per tile: `per_tile` renders (the designs whose shadow fits this sun, each about equally often) and a real sprite with
    probability `sprite_frac`. Validation tiles (never in Mk18's memory) get `val_extra` more renders and always a sprite: the
    pool to evaluate on."""
    rng = np.random.default_rng(seed); items = []
    for body, ts in tiles.items():
        hw = [k for k, v in HARDWARE.items() if v["body"] in (body, "both")]
        bal = Counter()
        for t in ts:
            val = t["split"] == "validation"
            fits = [k for k in hw if shadow_px(HARDWARE[k], t["incidence"], t["res"]) <= MAX_SHADOW_PX]
            for _ in range(per_tile + (val_extra if val else 0)):
                if fits:
                    h = min(fits, key=lambda k: (bal[k], rng.random())); bal[h] += 1
                    items.append(dict(body=body, tile=t, kind="rendered", hardware=h))
            ok = [s for s in sprites if s["body"] == body and abs(s["incidence"] - t["incidence"]) <= 12 and s["res"] <= t["res"] * 1.02]
            if ok and (val or rng.random() < sprite_frac):
                items.append(dict(body=body, tile=t, kind="real-sprite"))
    for k, it in enumerate(items):
        it["seed"] = seed * 100003 + k
    return items


def polygons(mask):
    """COCO polygons [[x0, y0, x1, y1, ...], ...] around a boolean mask, in image coordinates."""
    z = np.pad(mask.astype(np.float32), 1)
    out = []
    for line in contourpy.contour_generator(z=z).lines(0.5):
        if len(line) >= 4:
            out.append([round(float(v), 1) for v in (line[:-1] - 1).ravel()])
    return out


def coco_document(row, mask):
    """One complete COCO file for one injected image: the object with its cast shadow, as bbox and polygons."""
    both = (mask == 1) | (mask == 2)
    oy, ox = np.nonzero(mask == 1)
    obj_box = [int(ox.min()), int(oy.min()), int(np.ptp(ox)) + 1, int(np.ptp(oy)) + 1] if len(ox) else None
    x0, y0, x1, y1 = row["box_x_min"], row["box_y_min"], row["box_x_max"], row["box_y_max"]
    return {"info": {"description": "Xenarch synthetic injection: annotation of the object added to this natural-terrain image", "version": "3.0"},
            "licenses": [{"id": 1, "name": "CC0-1.0"}],
            "images": [{"id": 1, "file_name": row["file_name"], "width": SIZE, "height": SIZE, "license": 1, "natural_file": row["natural_file"], "mask_file": row["mask_file"],
                        **{k: row[k] for k in ("body", "instrument", "kind", "split", "landform", "frame_id", "resolution_m_px", "incidence_deg", "sun_azimuth_image_deg")}}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": CATEGORIES.index(row["object_class"]) + 1, "bbox": [x0, y0, x1 - x0 + 1, y1 - y0 + 1], "area": int(both.sum()),
                             "segmentation": polygons(both), "iscrowd": 0, "includes_shadow": True, "bbox_object_only": obj_box,
                             **{k: row[k] for k in ("hardware_id", "hardware_name", "era", "context", "object_px", "shadow_length_px", "shadow_area_px", "width_m", "height_m", "shadow_length_m")}}],
            "categories": coco_categories()}


def coco_categories():
    return [{"id": k + 1, "name": c, "supercategory": "synthetic-hardware"} for k, c in enumerate(CATEGORIES)]


_W = {}                                     # worker state: tiles, sprites, plan (shared by fork), tile info cache


def _info_of(t):
    if t["rel"] not in _W["cache"]:
        a = np.asarray(Image.open(t["path"]).convert("L"), np.float32)
        az, conf, source = sun_azimuth(a, t["body"], t["lat"], t["incidence"], t["ls"])
        _W["cache"][t["rel"]] = dict(az=az, conf=conf, source=source, psf=tile_psf(a))
    return _W["cache"][t["rel"]]


def _one(k):
    """Make and save image k of the plan (its seed is fixed, so the result does not depend on the worker)."""
    it = _W["items"][k]
    r = make(it, _W["sprites"], _info_of, np.random.default_rng(it["seed"]))
    if r is None:
        return None
    inj, natural, mask, row = r
    b, cls, land = row["body"].lower(), row["object_class"], row["landform"]
    stem = f"{b}_{row['kind']}_{row['hardware_id']}_{k:05d}"
    twin = f"{b}_{land.replace('_', '-')}_natural_{k:05d}"
    row.update(file_name=f"injected/{b}/{cls}/{stem}.png", mask_file=f"masks/{b}/{cls}/{stem}.png",
               natural_file=f"natural/{b}/{land}/{twin}.png", annotation_file=f"{ANNOTATION}/{b}/{cls}/{stem}_{ANNOTATION}.json")
    for rel, arr in ((row["file_name"], inj), (row["natural_file"], natural), (row["mask_file"], mask)):
        p = OUT / rel; p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).save(p, optimize=True)
    p = OUT / row["annotation_file"]; p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(coco_document(row, mask), p.open("w"), indent=1)
    return row


NUMERIC = {"x_center", "y_center", "box_x_min", "box_y_min", "box_x_max", "box_y_max", "object_px", "shadow_length_px", "shadow_area_px", "resolution_m_px", "camera_downscale",
           "incidence_deg", "sun_azimuth_image_deg", "sun_azimuth_confidence", "radiance_scale_dn", "psf_sigma_px", "noise_sigma", "gain", "gamma",
           "shadowed_fraction", "background_x0", "background_y0", "background_side_px", "seed"}


def _load_rows():
    rows = list(csv.DictReader((OUT / "metadata.csv").open()))
    for r in rows:
        for k in NUMERIC:
            r[k] = float(r[k]) if "." in r[k] else int(r[k])
    return rows


def run(per_tile=1, sprite_frac=0.6, limit=None, seed=7, val_extra=2, workers=6, only=None):
    """The whole set, or with `only` a single rendered design: same plan and seeds, so the other images stay as they are."""
    tiles = load_tiles()
    print("cutting real sprites ...", flush=True)
    sprites = build_sprites()
    print(f"  {len(sprites)} sprites: {dict(Counter((s['body'], s['cls']) for s in sprites))}")
    items = plan(tiles, sprites, per_tile, sprite_frac, seed, val_extra)
    if limit:
        items = items[:: max(1, len(items) // limit)][:limit]
    todo = [k for k, it in enumerate(items) if not only or it.get("hardware") == only]
    if only:
        old = {r["file_name"]: r for r in _load_rows()}
        print(f"  regenerating {len(todo)} images of {only}", flush=True)
    else:
        old = {}
        for sub in ("injected", "clean", "natural", "masks", ANNOTATION):
            for f in (OUT / sub).glob("*/*/*"):
                f.unlink()
            for d in (OUT / sub).glob("*/*"):
                d.rmdir()
    for f in (OUT / "contact-sheets").glob("*.jpg"):
        f.unlink()
    _W.update(sprites=sprites, items=items, cache={})
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        for n, row in enumerate(pool.imap(_one, todo, chunksize=4)):
            if row is not None:
                old[row["file_name"]] = row
            if (n + 1) % 100 == 0:
                print(f"  {n + 1}/{len(todo)}", flush=True)
    rows = sorted(old.values(), key=lambda r: (r["body"] != "Moon", r["file_name"].rsplit("_", 1)[1])) if only else [old[k] for k in old]
    coco = {"info": {"description": "Xenarch synthetic injections: one space-hardware object added to natural terrain (LROC NAC, HiRISE)", "version": "3.0"},
            "licenses": [{"id": 1, "name": "CC0-1.0"}], "images": [], "annotations": [], "categories": coco_categories()}
    for n, row in enumerate(rows, 1):                     # the whole set = the per-image files, renumbered
        doc = json.load((OUT / row["annotation_file"]).open())
        coco["images"].append({**doc["images"][0], "id": n}); coco["annotations"].append({**doc["annotations"][0], "id": n, "image_id": n})
    with (OUT / "metadata.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS, lineterminator="\n"); w.writeheader(); w.writerows(rows)
    json.dump(coco, (OUT / "boxes_coco.json").open("w"), indent=1)
    sheets(rows)
    galleries(rows)
    realism(); design_sheet(OUT / "contact-sheets/designs.jpg")
    dashboard(rows)
    n = Counter((r["body"], r["kind"]) for r in rows)
    print(f"{len(rows)} injected images in {OUT}: " + ", ".join(f"{b} {k} {v}" for (b, k), v in sorted(n.items())))


def _font(size):
    for f in ("/System/Library/Fonts/Menlo.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def sheets(rows, t=200, cols=4, per=16):
    """contact-sheets/<body>_<kind>.jpg: natural | injected pairs with the box outlined, and overview.jpg."""
    d = OUT / "contact-sheets"; d.mkdir(parents=True, exist_ok=True)
    navy, orange, muted = (12, 22, 40), (240, 136, 78), (140, 150, 170)
    f12, f11 = _font(13), _font(11)
    by = defaultdict(list)
    for r in rows:
        by[(r["body"], r["kind"])].append(r)
    def pair(r):
        a = Image.open(OUT / r["natural_file"]).convert("RGB").resize((t, t), Image.LANCZOS)
        b = Image.open(OUT / r["file_name"]).convert("RGB").resize((t, t), Image.LANCZOS)
        k = t / SIZE; ImageDraw.Draw(b).rectangle([r["box_x_min"] * k - 3, r["box_y_min"] * k - 3, r["box_x_max"] * k + 3, r["box_y_max"] * k + 3], outline=orange, width=1)
        return a, b
    for (body, kind), rs in by.items():
        rs = sorted(rs, key=lambda r: r["hardware_id"])[:: max(1, len(rs) // per)][:per]
        nr = math.ceil(len(rs) / cols)
        im = Image.new("RGB", (cols * (2 * t + 18) + 8, 36 + nr * (t + 34)), navy); dr = ImageDraw.Draw(im)
        dr.text((8, 10), f"{body.upper()} · {kind.upper()} · NATURAL | INJECTED", fill=orange, font=f12)
        for i, r in enumerate(rs):
            x, y = 8 + (i % cols) * (2 * t + 18), 36 + (i // cols) * (t + 34)
            a, b = pair(r); im.paste(a, (x, y)); im.paste(b, (x + t + 2, y))
            dr.text((x, y + t + 4), f"{r['hardware_id'][:28]} · {r['resolution_m_px']} m/px · i={r['incidence_deg']:.0f}", fill=muted, font=f11)
            dr.text((x, y + t + 18), f"context {r['context']} · scale {r['scale_factor']}", fill=muted, font=f11)
        im.save(d / f"{body.lower()}_{kind}.jpg", quality=88)
    pick = []
    for (body, kind), rs in sorted(by.items()):
        pick += sorted(rs, key=lambda r: r["hardware_id"])[:: max(1, len(rs) // 4)][:4]
    nr = math.ceil(len(pick) / cols)
    im = Image.new("RGB", (cols * (2 * t + 18) + 8, 36 + nr * (t + 34)), navy); dr = ImageDraw.Draw(im)
    dr.text((8, 10), "XENARCH SYNTHETIC INJECTIONS · NATURAL | INJECTED", fill=orange, font=f12)
    for i, r in enumerate(pick):
        x, y = 8 + (i % cols) * (2 * t + 18), 36 + (i // cols) * (t + 34)
        a, b = pair(r); im.paste(a, (x, y)); im.paste(b, (x + t + 2, y))
        dr.text((x, y + t + 4), f"{r['body']} · {r['kind']} · {r['hardware_id'][:22]}", fill=muted, font=f11)
        dr.text((x, y + t + 18), f"{r['resolution_m_px']} m/px · i={r['incidence_deg']:.0f} · {r['context']}", fill=muted, font=f11)
    im.save(d / "overview.jpg", quality=88)


def galleries(rows=None, z=56, s=3, cols=3):
    """contact-sheets/gallery_*.jpg: one zoomed before | after per rendered design, and samples of the real cut-outs.
    The objects are too small to judge in the whole-image sheets."""
    rows = rows or list(csv.DictReader((OUT / "metadata.csv").open()))
    d = OUT / "contact-sheets"; navy, orange, muted, ink = (12, 22, 40), (240, 136, 78), (124, 137, 163), (232, 237, 246)
    def pick(kind, key, per=1):
        by = defaultdict(list)
        for r in rows:
            if r["kind"] == kind:
                by[r[key]].append(r)
        out = []
        for k in sorted(by):
            rs = by[k]
            good = [r for r in rs if 4 <= int(r["object_px"]) <= 70 and float(r["shadowed_fraction"]) < 0.2] or rs
            good = sorted(good, key=lambda r: (r["split"] != "validation", -min(int(r["object_px"]), 24), r["file_name"]))
            out += good[:per]
        return out
    def sheet(items, title, name, label=lambda r: r["hardware_name"]):
        if not items:
            return
        P = 2 * z * s; cw = 2 * P + 38; nr = math.ceil(len(items) / cols)
        im = Image.new("RGB", (cols * cw + 28, 70 + nr * (P + 64)), navy); dr = ImageDraw.Draw(im)
        dr.text((28, 18), title, fill=orange, font=_font(20)); dr.text((28, 44), "each pair: natural terrain | same terrain with the injected object (zoom x3)", fill=muted, font=_font(13))
        for i, r in enumerate(items):
            x, y = 28 + (i % cols) * cw, 70 + (i // cols) * (P + 64)
            cx, cy = int(float(r["x_center"])), int(float(r["y_center"]))
            x0, y0 = min(max(0, cx - z), SIZE - 2 * z), min(max(0, cy - z), SIZE - 2 * z)
            for j, f in enumerate(("natural_file", "file_name")):
                im.paste(Image.open(OUT / r[f]).crop((x0, y0, x0 + 2 * z, y0 + 2 * z)).resize((P, P), Image.NEAREST).convert("RGB"), (x + j * (P + 10), y))
            era = (r["era"].upper() + " · ") if r["era"] else ""
            dr.text((x, y + P + 6), f"{era}{label(r)[:44]}", fill=ink, font=_font(13))
            dr.text((x, y + P + 24), f"{r['body']} · {r['resolution_m_px']} m/px · sun {float(r['incidence_deg']):.0f} deg · {r['landform'].replace('_', ' ')}", fill=muted, font=_font(12))
            dr.text((x, y + P + 40), f"{r['object_px']} px object · shadow {r['shadow_length_px']} px long · {r['context']}", fill=muted, font=_font(12))
        im.save(d / name, quality=90)
    order = list(HARDWARE)
    ren = sorted(pick("rendered", "hardware_id"), key=lambda r: (HARDWARE[r["hardware_id"]]["era"] == "new", order.index(r["hardware_id"])))
    sheet([r for r in ren if r["body"] == "Moon"], "XENARCH · RENDERED HARDWARE ON THE MOON (LROC NAC TERRAIN)", "gallery_rendered_moon.jpg")
    sheet([r for r in ren if r["body"] == "Mars"], "XENARCH · RENDERED HARDWARE ON MARS (HIRISE TERRAIN)", "gallery_rendered_mars.jpg")
    sheet(sorted(pick("real-sprite", "hardware_id"), key=lambda r: (r["body"], r["hardware_id"]))[:18], "XENARCH · REAL HARDWARE CUT OUT AND PASTED INTO NEW TERRAIN", "gallery_real-sprites.jpg",
          label=lambda r: r["hardware_name"] + " (real)")


def dashboard(rows=None, z=56):
    """dashboard/injections.json (every row) + thumbnails in data/synthetic-injections/thumbs/: the zoom around
    the object (injected and clean, 2x) and both whole images at half size (before / after on the page). The page reads them through the
    dashboard/injections-full link (data/ is not in git)."""
    rows = rows or list(csv.DictReader((OUT / "metadata.csv").open()))
    th = OUT / "thumbs"; th.mkdir(exist_ok=True)
    keep, out = set(), []
    for r in rows:
        stem = Path(r["file_name"]).stem
        cx, cy = int(float(r["x_center"])), int(float(r["y_center"]))
        x0, y0 = min(max(0, cx - z), SIZE - 2 * z), min(max(0, cy - z), SIZE - 2 * z)
        for tag, f in (("inj", r["file_name"]), ("clean", r["natural_file"])):
            im = Image.open(OUT / f)
            im.crop((x0, y0, x0 + 2 * z, y0 + 2 * z)).resize((4 * z, 4 * z), Image.NEAREST).save(th / f"{stem}_{tag}.jpg", quality=86)
            keep.add(f"{stem}_{tag}.jpg")
        for tag, f in (("full", r["file_name"]), ("cleanfull", r["natural_file"])):
            Image.open(OUT / f).resize((SIZE // 2, SIZE // 2), Image.LANCZOS).save(th / f"{stem}_{tag}.jpg", quality=84)
            keep.add(f"{stem}_{tag}.jpg")
        b = [int(r[k]) for k in ("box_x_min", "box_y_min", "box_x_max", "box_y_max")]
        out.append({"id": stem, "kind": r["kind"], "body": r["body"], "era": r["era"], "cls": r["object_class"],
                    "hw": r["hardware_id"], "name": r["hardware_name"], "split": r["split"], "res": float(r["resolution_m_px"]),
                    "inc": float(r["incidence_deg"]), "az": float(r["sun_azimuth_image_deg"]), "conf": float(r["sun_azimuth_confidence"]),
                    "ctx": r["context"], "px": int(r["object_px"]), "shadow": int(r["shadow_length_px"]), "landform": r["landform"],
                    "shadowed": float(r["shadowed_fraction"]),
                    "scale": r["scale_factor"], "albedo": r["albedo_factor"], "contrast": r["contrast_factor"], "psf": float(r["psf_sigma_px"]),
                    "noise": float(r["noise_sigma"]), "gamma": float(r["gamma"]), "gain": float(r["gain"]), "down": float(r["camera_downscale"]),
                    "w": r["width_m"], "h": r["height_m"], "frame": r["frame_id"], "sprite": r["sprite_source"], "bg": r["background_file"],
                    "box": b, "zoom": [x0, y0, 2 * z], "img": r["file_name"], "natural": r["natural_file"], "mask": r["mask_file"], "ann": r["annotation_file"]})
    for old in th.glob("*.jpg"):
        if old.name not in keep:
            old.unlink()
    bodies = {k: " / ".join(sorted({r["body"] for r in rows if r["hardware_id"] == k})) or v["body"] for k, v in HARDWARE.items()}
    designs = {k: {"name": v["name"], "era": v["era"], "cls": v["cls"], "body": bodies[k],
                   "footprint_m": round(footprint_m(v), 1), "height_m": round(height_m(v), 1)} for k, v in HARDWARE.items()}
    check = []
    if (OUT / "realism_report.csv").exists():
        by = defaultdict(list)
        for q in csv.DictReader((OUT / "realism_report.csv").open()):
            by[q["design"]].append(q)
        for d, v in sorted(by.items()):
            med = lambda k: round(float(np.median([float(q[k]) for q in v])), 1)
            med2 = lambda k: (lambda w: round(float(np.median(w)), 1) if w else None)([float(q[k]) for q in v if q[k] != ""])
            check.append(dict(design=designs[d]["name"], n=len(v), inc=med("incidence_deg"), res=round(float(np.median([float(q["resolution_m_px"]) for q in v])), 2),
                              **{k: med2(k) for k in ("real_edge_px", "model_edge_px", "real_inside_dn", "model_inside_dn", "real_body_dn", "model_body_dn")},
                              **{k: med(k) for k in ("real_tip_m", "model_tip_m", "real_shadow_w_m", "model_shadow_w_m", "real_peak_dn", "model_peak_dn",
                                                     "real_floor_dn", "model_floor_dn", "real_body_m", "model_body_m")}))
    dash = HERE / "dashboard"
    json.dump({"generated": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
               "size": SIZE, "designs": designs, "check": check, "rows": out}, (dash / "injections.json").open("w"), separators=(",", ":"))
    link = dash / "injections-full"
    if not link.exists():
        link.symlink_to(Path("..") / OUT.relative_to(HERE))
    print(f"dashboard/injections.json: {len(out)} images; thumbnails in {th}")


def design_sheet(path, z=48, s=3, cols=4):
    """Every rendered design at a mid sun on real terrain, zoomed: before | after. For checking the designs by eye."""
    tiles = load_tiles(); items = []
    for hid, spec in HARDWARE.items():
        for body in (("Moon", "Mars") if spec["body"] == "both" else (spec["body"],)):
            rng = np.random.default_rng(abs(hash_str(hid + body)) % 2 ** 31)
            cand = [t for t in tiles[body] if 45 <= t["incidence"] <= 70 and t["res"] <= 0.75 and shadow_px(spec, t["incidence"], t["res"]) <= MAX_SHADOW_PX]
            cand = cand or [t for t in tiles[body] if shadow_px(spec, t["incidence"], t["res"]) <= MAX_SHADOW_PX]
            if not cand:
                continue
            t = cand[int(rng.integers(len(cand)))]
            a = np.asarray(Image.open(t["path"]).convert("L"), np.float32)
            az = sun_azimuth(a, body, t["lat"], t["incidence"], t["ls"])[0]
            bg = background(t, rng, False)[0]; cx, cy = pick_spot(bg, rng); level, u = radiance_scale(bg, t["incidence"], body)
            ctx = contexts(spec, rng)
            work = add_context(bg.copy(), ctx, spec, cx, cy, t["res"], footprint_m(spec), rng, body, t["incidence"], level, u)
            out, mask, r = render(spec, work, t["res"], cx, cy, az, t["incidence"], body, rng, 1.0, 1.0, level, u)
            out = np.clip(camera(bg, out, mask, tile_psf(a), rng), 0, 255)
            x0, y0 = int(np.clip(cx - z, 0, SIZE - 2 * z)), int(np.clip(cy - z, 0, SIZE - 2 * z))
            items.append((bg[y0:y0 + 2 * z, x0:x0 + 2 * z], out[y0:y0 + 2 * z, x0:x0 + 2 * z], spec, body, t, r, ctx))
    P = 2 * z * s; nr = math.ceil(len(items) / cols)
    im = Image.new("RGB", (cols * (2 * P + 18) + 8, 40 + nr * (P + 50)), (12, 22, 40)); dr = ImageDraw.Draw(im); f12 = _font(12)
    dr.text((8, 12), "XENARCH · EVERY RENDERED DESIGN, BEFORE | AFTER (ZOOM X3)", fill=(240, 136, 78), font=_font(14))
    for k, (a, b, spec, body, t, r, ctx) in enumerate(items):
        x, y = 8 + (k % cols) * (2 * P + 18), 40 + (k // cols) * (P + 50)
        for j, q in enumerate((a, b)):
            im.paste(Image.fromarray(np.clip(q, 0, 255).astype(np.uint8)).resize((P, P), Image.NEAREST), (x + j * (P + 2), y))
        dr.text((x, y + P + 3), f"{spec['era'].upper()} · {spec['name'][:44]}", fill=(240, 136, 78), font=f12)
        dr.text((x, y + P + 17), f"{body} · {t['res']} m/px · sun {t['incidence']:.0f} deg · {t['landform']}", fill=(140, 150, 170), font=f12)
        dr.text((x, y + P + 31), f"{r['width_m']} m wide · {r['height_m']} m tall · shadow {r['shadow_length_m']} m · {'+'.join(ctx) or 'no context'}", fill=(140, 150, 170), font=f12)
    im.save(path, quality=90); print(path, im.size)


def hash_str(t):
    return sum((k + 1) * ord(c) for k, c in enumerate(t))


def _arg(name, default, cast=str):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


if __name__ == "__main__":
    if "--dashboard" in sys.argv:                 # page data and thumbnails only, from the existing set
        dashboard(); sys.exit()
    if "--galleries" in sys.argv:                 # zoomed sheets only, from the existing set
        galleries(); sys.exit()
    if "--realism" in sys.argv:                   # only the check against the real hardware
        realism(); design_sheet(OUT / "contact-sheets/designs.jpg"); sys.exit()
    run(per_tile=_arg("--per-tile", 1, int), sprite_frac=_arg("--sprite-frac", 0.6, float), limit=_arg("--limit", None, int), seed=_arg("--seed", 7, int),
        val_extra=_arg("--val-extra", 2, int), workers=_arg("--workers", 6, int), only=_arg("--only", None))
