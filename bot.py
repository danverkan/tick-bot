"""Telegram-бот «Клещи: потенциал местообитаний» (Беларусь). Один файл. Запуск: python bot.py"""

import datetime as dt
import glob
import html as _html
import io
import json
import logging
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from html import escape
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlencode

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont


# ======================================================================
# КОНФИГУРАЦИЯ
# ======================================================================


def env(name, default=None):
    v = os.environ.get(name)
    return v if v not in (None, "") else default


BOT_TOKEN = env("BOT_TOKEN", "")
MAP_URL = env("MAP_URL", "https://tick-habitat.nextgis.com/resource/21/display")
CONTACT = env("CONTACT", "contact-not-set@example.com")
USER_AGENT = f"TickHabitatBelarusBot/1.0 ({CONTACT})"

DATA_DIR = env("DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
_found = sorted(glob.glob(os.path.join(DATA_DIR, "tick_index_*m.npy")))
GRID_NPY = env("GRID_NPY", _found[0] if _found else os.path.join(DATA_DIR, "tick_index_30m.npy"))
GRID_META = env("GRID_META", GRID_NPY[:-4] + ".json")
DATA_URL_NPY = env("DATA_URL_NPY")
DATA_URL_META = env("DATA_URL_META")

# Радиусы анализа окружения, м (500 м ~ «двор/ближайший лес», 1 км ~ «прогулка из НП»)
RADII = (500, 1000)
HEADLINE_RADIUS = 500          # по этому радиусу определяется итоговая категория
VIEW_RADIUS_M = 1500           # половина стороны картинки-фрагмента

# Границы Беларуси (грубый bbox) для отсева точек за пределами страны
BY_BBOX = (51.2, 23.1, 56.2, 32.8)  # lat_min, lon_min, lat_max, lon_max

# Подложка для картинки. Для постоянной нагрузки замените на свой провайдер.
BASEMAP_URL = env("BASEMAP_URL", "https://tile.openstreetmap.org/{z}/{x}/{y}.png")
BASEMAP_ATTRIBUTION = env("BASEMAP_ATTRIBUTION", "© OpenStreetMap contributors")
TILE_CACHE_DIR = env("TILE_CACHE_DIR", os.path.join(DATA_DIR, "tile_cache"))

# Необязательно: отрисовка фрагмента самим NextGIS Web (id СТИЛЯ растра)
NGW_BASE = env("NGW_BASE")
NGW_RENDER_RESOURCE = env("NGW_RENDER_RESOURCE")

NOMINATIM_URL = env("NOMINATIM_URL", "https://nominatim.openstreetmap.org/search")
USER_COOLDOWN_S = float(env("USER_COOLDOWN_S", "3"))


# ======================================================================
# ГЕО
# ======================================================================
"""Проекции и вспомогательные геофункции (без внешних зависимостей, только numpy)."""

_A = 6378137.0
_F = 1 / 298.257223563
_N = _F / (2 - _F)
_AA = _A / (1 + _N) * (1 + _N**2 / 4 + _N**4 / 64)
_ALPHA = (
    _N / 2 - 2 * _N**2 / 3 + 5 * _N**3 / 16,
    13 * _N**2 / 48 - 3 * _N**3 / 5,
    61 * _N**3 / 240,
)
_K0 = 0.9996


def lonlat_to_utm(lon, lat, zone=35):
    """WGS84 -> UTM (северное полушарие), ряды Крюгера. Принимает числа или массивы.
    zone=35 соответствует EPSG:32635, в котором посчитан растр."""
    lon = np.asarray(lon, dtype="float64")
    lat = np.asarray(lat, dtype="float64")
    lam = np.radians(lon - (6 * zone - 183))
    phi = np.radians(lat)
    c = 2 * math.sqrt(_N) / (1 + _N)
    t = np.sinh(np.arctanh(np.sin(phi)) - c * np.arctanh(c * np.sin(phi)))
    xi = np.arctan2(t, np.cos(lam))
    eta = np.arctanh(np.sin(lam) / np.sqrt(1 + t * t))
    x = eta.copy()
    y = xi.copy()
    for j, a in enumerate(_ALPHA, start=1):
        x = x + a * np.cos(2 * j * xi) * np.sinh(2 * j * eta)
        y = y + a * np.sin(2 * j * xi) * np.cosh(2 * j * eta)
    return 500000.0 + _K0 * _AA * x, _K0 * _AA * y


# --- Web Mercator / тайлы -------------------------------------------------
def lonlat_to_world_px(lon, lat, z):
    n = 256 * (2**z)
    x = (np.asarray(lon, dtype="float64") + 180.0) / 360.0 * n
    s = np.sin(np.radians(lat))
    y = (0.5 - np.log((1 + s) / (1 - s)) / (4 * math.pi)) * n
    return x, y


def world_px_to_lonlat(x, y, z):
    n = 256 * (2**z)
    lon = np.asarray(x, dtype="float64") / n * 360.0 - 180.0
    lat = np.degrees(np.arctan(np.sinh(math.pi * (1 - 2 * np.asarray(y, dtype="float64") / n))))
    return lon, lat


def meters_per_px(lat, z):
    return 156543.03392 * math.cos(math.radians(lat)) / (2**z)


def merc_xy(lon, lat):
    """EPSG:3857 (для запроса render/image в NextGIS Web)."""
    R = 6378137.0
    x = math.radians(lon) * R
    y = math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)) * R
    return x, y


def in_belarus_bbox(lat, lon, bbox):
    return bbox[0] <= lat <= bbox[2] and bbox[1] <= lon <= bbox[3]


# ======================================================================
# РАСТР
# ======================================================================
"""Доступ к растру индекса пригодности (uint8, 0–100; 0 = неблагоприятно; 255 = вне данных).

Растр хранится как .npy (читается через memory-map: почти не занимает ОЗУ) + .json с метаданными,
которые создаёт prepare_grid.py (Colab) из ваших GeoTIFF-тайлов.
"""



NODATA = 255
CLASS_NAMES = {
    0: "неблагоприятные условия",
    1: "очень низкий потенциал",
    2: "низкий потенциал",
    3: "средний потенциал",
    4: "высокий потенциал",
    5: "очень высокий потенциал",
}


class Grid:
    def __init__(self, npy_path, meta_path):
        with open(meta_path, encoding="utf-8") as f:
            self.meta = json.load(f)
        self.npy_path = npy_path
        self.a = np.load(npy_path, mmap_mode="r")
        m = self.meta
        self.x0, self.y0, self.cs = m["x0"], m["y0"], m["cellsize"]
        self.h, self.w = self.a.shape
        assert (self.h, self.w) == (m["height"], m["width"]), "Размер .npy не совпадает с .json"
        self.zone = int(m.get("utm_zone", 35))
        self.breaks = list(m["breaks"])  # 4 порога между классами 1|2|3|4|5 (квинтили ненулевых значений)
        hist = np.asarray(m["hist"], dtype="float64")  # hist[v] — число ячеек со значением v (1..100)
        hist[0] = 0
        self._cum = np.cumsum(hist)
        self._total = float(self._cum[-1]) or 1.0
        self.p90_breaks = None   # пороги классов для статистики «p90 в круге» (см. calibrate_p90)
        self.p90_info = {}

    # ---- индексы ----
    def xy(self, lon, lat):
        return lonlat_to_utm(lon, lat, self.zone)

    def rowcol(self, x, y):
        col = np.floor((np.asarray(x) - self.x0) / self.cs).astype("int64")
        row = np.floor((self.y0 - np.asarray(y)) / self.cs).astype("int64")
        return row, col

    def covers(self, lon, lat):
        x, y = self.xy(lon, lat)
        r, c = self.rowcol(x, y)
        return bool(0 <= r < self.h and 0 <= c < self.w)

    # ---- классификация ----
    def classify(self, v):
        """Класс 0–5. Классы 1–5 — квинтили распределения ненулевых значений по всей территории."""
        if v is None or v == NODATA or v <= 0:
            return 0
        return 1 + sum(v > b for b in self.breaks)

    def classify_array(self, arr):
        out = np.zeros(arr.shape, dtype="uint8")
        valid = (arr > 0) & (arr != NODATA)
        cls = 1 + sum((arr > b).astype("uint8") for b in self.breaks)
        out[valid] = cls[valid]
        return out

    def percentile_rank(self, v):
        """Доля (%) ячеек с ненулевым индексом, значение которых не выше v."""
        if v is None or v <= 0:
            return 0.0
        i = int(min(100, max(1, round(v))))
        below = self._cum[i - 1]
        here = self._cum[i] - below
        return 100.0 * (below + 0.5 * here) / self._total

    # ---- выборка ----
    def value_at(self, lon, lat):
        x, y = self.xy(lon, lat)
        r, c = self.rowcol(x, y)
        if not (0 <= r < self.h and 0 <= c < self.w):
            return None
        v = int(self.a[r, c])
        return None if v == NODATA else v

    def _window_vals(self, x, y, radius_m):
        """Значения ячеек (без NoData) в круге радиуса radius_m вокруг точки (x, y) в UTM."""
        r0, c0 = self.rowcol(x, y)
        k = int(np.ceil(radius_m / self.cs)) + 1
        ra, rb = max(0, r0 - k), min(self.h, r0 + k + 1)
        ca, cb = max(0, c0 - k), min(self.w, c0 + k + 1)
        if ra >= rb or ca >= cb:
            return None
        win = np.asarray(self.a[ra:rb, ca:cb])
        cy = self.y0 - (np.arange(ra, rb) + 0.5) * self.cs
        cx = self.x0 + (np.arange(ca, cb) + 0.5) * self.cs
        d2 = (cx[None, :] - x) ** 2 + (cy[:, None] - y) ** 2
        mask = (d2 <= radius_m**2) & (win != NODATA)
        vals = win[mask]
        return vals if vals.size else None

    def window_stats(self, lon, lat, radius_m):
        """Статистика индекса в круге радиуса radius_m вокруг точки."""
        x, y = self.xy(lon, lat)
        raw = self._window_vals(x, y, radius_m)
        if raw is None:
            return None
        vals = raw.astype("float64")
        return {
            "radius": radius_m,
            "n": int(vals.size),
            "mean": float(vals.mean()),
            "p90": float(np.percentile(vals, 90)),
            "max": float(vals.max()),
            "share_suitable": float(np.mean(vals > 0)) * 100,
            "share_high": float(np.mean(self.classify_array(raw) >= 4)) * 100,
        }

    # ---- калибровка порогов для статистики окружения ----
    def calibrate_p90(self, radius_m=500, n=10000, seed=1):
        """Пороги классов 1–5 для величины «90-й перцентиль индекса в круге radius_m».
        Считаются как квинтили этой же величины по случайной выборке n точек с ненулевым индексом.
        Нужно, потому что пороги по значениям отдельных пикселей (meta['breaks']) для статистики
        окружения не подходят: p90 в круге систематически выше значения в пикселе."""
        cache = self.npy_path[:-4] + f"_p90_{radius_m}.json"
        try:
            with open(cache, encoding="utf-8") as f:
                c = json.load(f)
            if c.get("grid_created") == self.meta.get("created") and c.get("n") == n and c.get("seed") == seed:
                self.p90_breaks, self.p90_info = c["breaks"], c
                return self.p90_breaks
        except Exception:
            pass
        rng = np.random.default_rng(seed)
        rows = np.sort(rng.integers(0, self.h, size=6 * n))   # сортировка — только для скорости чтения
        cols = rng.integers(0, self.w, size=6 * n)
        v = np.asarray(self.a[rows, cols])
        ok = (v > 0) & (v != NODATA)
        rows, cols = rows[ok], cols[ok]
        pick = np.sort(rng.permutation(len(rows))[:n])        # случайный отбор n из ВСЕХ допустимых точек
        rows, cols = rows[pick], cols[pick]
        out = []
        for r, c0 in zip(rows, cols):
            vals = self._window_vals(self.x0 + (c0 + 0.5) * self.cs, self.y0 - (r + 0.5) * self.cs, radius_m)
            if vals is not None:
                out.append(np.percentile(vals, 90))
        if len(out) < 50:
            return None
        br = [float(np.quantile(out, q)) for q in (0.2, 0.4, 0.6, 0.8)]
        self.p90_breaks = br
        self.p90_info = {"breaks": br, "n": len(out), "seed": seed, "radius": radius_m,
                         "grid_created": self.meta.get("created")}
        try:
            with open(cache, "w", encoding="utf-8") as f:
                json.dump(self.p90_info, f, ensure_ascii=False)
        except Exception:
            pass
        return br

    def classify_stat(self, v):
        """Класс 0–5 для p90 в круге: по порогам этой же статистики (если нет калибровки — по пикселям)."""
        br = self.p90_breaks or self.breaks
        if v is None or v <= 0:
            return 0
        return 1 + sum(v > b for b in br)


def ensure_data(npy_path, meta_path, url_npy=None, url_meta=None):
    """Скачивает растр при первом запуске, если задан URL и файлов нет."""
    import requests

    for path, url in ((meta_path, url_meta), (npy_path, url_npy)):
        if os.path.exists(path) or not url:
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".part"
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        os.replace(tmp, path)


# ======================================================================
# ГЕОКОДЕР
# ======================================================================
"""Поиск населённого пункта по названию (Nominatim/OSM, бесплатно, ≤1 запрос/с)."""



_gc_lock = threading.Lock()
_gc_last = [0.0]
_gc_cache = {}

_COORD_RE = re.compile(r"^\s*(-?\d{1,3}(?:[.,]\d+)?)\s*[,;\s]\s*(-?\d{1,3}(?:[.,]\d+)?)\s*$")


def parse_coords(text):
    """'53.9006, 27.5590' -> (lat, lon). Допускает запятую как десятичный разделитель."""
    m = _COORD_RE.match(text or "")
    if not m:
        return None
    a, b = (float(g.replace(",", ".")) for g in m.groups())
    lat, lon = (a, b) if abs(a) <= 90 and a > 40 else (b, a)  # допускаем «lon, lat»
    if -90 <= lat <= 90 and -180 <= lon <= 180:
        return lat, lon
    return None


def _label(item):
    addr = item.get("address", {})
    name = item.get("name") or item.get("display_name", "").split(",")[0]
    parts = [name]
    for key in ("county", "state"):
        v = addr.get(key)
        if v and v != name and v not in parts:
            parts.append(v)
    return ", ".join(parts)


def _gc_query(params):
    with _gc_lock:  # политика Nominatim: не чаще 1 запроса в секунду
        wait = 1.1 - (time.time() - _gc_last[0])
        if wait > 0:
            time.sleep(wait)
        r = requests.get(
            NOMINATIM_URL,
            params=params,
            headers={"User-Agent": USER_AGENT, "Accept-Language": "ru"},
            timeout=15,
        )
        _gc_last[0] = time.time()
    r.raise_for_status()
    return r.json()


def geocode_search(text, limit=5):
    """Возвращает список {'label','lat','lon'} (только Беларусь), лучшие совпадения первыми."""
    key = text.strip().lower()
    if key in _gc_cache:
        return _gc_cache[key]
    base = {"q": text, "format": "jsonv2", "countrycodes": "by", "limit": 10,
            "addressdetails": 1, "accept-language": "ru"}
    data = _gc_query({**base, "featureType": "settlement"})
    if not data:
        data = [d for d in _gc_query(base) if d.get("category") in ("place", "boundary", "landuse", "natural", "leisure")]
    out, seen = [], set()
    for d in data:
        lat, lon = float(d["lat"]), float(d["lon"])
        lab = _label(d)
        k = (lab, round(lat, 2), round(lon, 2))
        if k in seen:
            continue
        seen.add(k)
        out.append({"label": lab, "lat": lat, "lon": lon})
        if len(out) >= limit:
            break
    _gc_cache[key] = out
    if len(_gc_cache) > 2000:
        _gc_cache.pop(next(iter(_gc_cache)))
    return out


# ======================================================================
# КАРТИНКА
# ======================================================================
"""Картинка-фрагмент: подложка OSM + раскраска индекса из растра + круги 500 м / 1 км."""



# Цвета классов 1–5 (спокойная последовательная шкала); класс 0 — прозрачно
CLASS_RGBA = {
    1: (255, 237, 160, 120),
    2: (254, 204, 92, 140),
    3: (253, 141, 60, 150),
    4: (227, 74, 51, 165),
    5: (153, 0, 13, 185),
}
SIZE = 640


def _font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans.ttf",
              "/Library/Fonts/Arial.ttf", "C:/Windows/Fonts/arial.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _fetch_tile(z, x, y):
    path = os.path.join(TILE_CACHE_DIR, str(z), str(x), f"{y}.png")
    if os.path.exists(path):
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            pass
    url = BASEMAP_URL.format(z=z, x=x, y=y)
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=10)
    r.raise_for_status()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(r.content)
    return Image.open(io.BytesIO(r.content)).convert("RGB")


def _basemap(lon, lat, z):
    """Подложка SIZE×SIZE, центрированная на точке. При сбое сети — нейтральный фон."""
    cx, cy = lonlat_to_world_px(lon, lat, z)
    left, top = cx - SIZE / 2, cy - SIZE / 2
    img = Image.new("RGB", (SIZE, SIZE), (236, 236, 232))
    tx0, tx1 = int(left // 256), int((left + SIZE - 1) // 256)
    ty0, ty1 = int(top // 256), int((top + SIZE - 1) // 256)
    jobs = [(tx, ty) for tx in range(tx0, tx1 + 1) for ty in range(ty0, ty1 + 1)]
    ok = False

    def work(j):
        try:
            return j, _fetch_tile(z, *j)
        except Exception:
            return j, None

    with ThreadPoolExecutor(max_workers=6) as ex:
        for (tx, ty), tile in ex.map(work, jobs):
            if tile is not None:
                ok = True
                img.paste(tile, (int(round(tx * 256 - left)), int(round(ty * 256 - top))))
    return img, ok, (left, top)


def _overlay_local(grid, left, top, z):
    """RGBA-слой индекса из локального растра (ближайший сосед — без выдуманной детальности)."""
    px = left + np.arange(SIZE) + 0.5
    py = top + np.arange(SIZE) + 0.5
    X, Y = np.meshgrid(px, py)
    lon, lat = world_px_to_lonlat(X, Y, z)
    ux, uy = grid.xy(lon, lat)
    r, c = grid.rowcol(ux, uy)
    inside = (r >= 0) & (r < grid.h) & (c >= 0) & (c < grid.w)
    vals = np.full(r.shape, 255, dtype="uint8")
    rr, cc = np.clip(r, 0, grid.h - 1), np.clip(c, 0, grid.w - 1)
    vals[inside] = grid.a[rr, cc][inside]
    cls = grid.classify_array(vals)
    rgba = np.zeros((SIZE, SIZE, 4), dtype="uint8")
    for k, col in CLASS_RGBA.items():
        rgba[cls == k] = col
    return Image.fromarray(rgba, "RGBA")


def _overlay_ngw(lon, lat, z, left, top):
    """Необязательно: фрагмент растра, отрисованный самим NextGIS Web (идентичен веб-карте)."""
    if not (NGW_BASE and NGW_RENDER_RESOURCE):
        return None
    try:
        lo0, la0 = world_px_to_lonlat(left, top + SIZE, z)
        lo1, la1 = world_px_to_lonlat(left + SIZE, top, z)
        x0, y0 = merc_xy(float(lo0), float(la0))
        x1, y1 = merc_xy(float(lo1), float(la1))
        r = requests.get(
            f"{NGW_BASE.rstrip('/')}/api/component/render/image",
            params={"resource": NGW_RENDER_RESOURCE, "extent": f"{x0},{y0},{x1},{y1}",
                    "size": f"{SIZE},{SIZE}"},
            headers={"User-Agent": USER_AGENT}, timeout=20)
        r.raise_for_status()
        im = Image.open(io.BytesIO(r.content)).convert("RGBA")
        return im if im.size == (SIZE, SIZE) else im.resize((SIZE, SIZE))
    except Exception:
        return None


def render_map(grid, lon, lat):
    """Возвращает PNG (bytes)."""
    half = VIEW_RADIUS_M
    # масштаб подбираем так, чтобы SIZE пикселей ≈ 2*half метров
    z = 14
    for cand in (16, 15, 14, 13, 12):
        if meters_per_px(lat, cand) * SIZE >= 2 * half:
            z = cand
            break
    base, base_ok, (left, top) = _basemap(lon, lat, z)
    ov = _overlay_ngw(lon, lat, z, left, top) or _overlay_local(grid, left, top, z)
    img = Image.alpha_composite(base.convert("RGBA"), ov)
    d = ImageDraw.Draw(img, "RGBA")
    c = SIZE / 2
    mpp = meters_per_px(lat, z)
    for rad, dash in ((RADII[0], (255, 255, 255, 255)), (RADII[-1], (255, 255, 255, 255))):
        rp = rad / mpp
        d.ellipse([c - rp, c - rp, c + rp, c + rp], outline=(20, 20, 20, 230), width=3)
        d.ellipse([c - rp, c - rp, c + rp, c + rp], outline=dash, width=1)
    d.ellipse([c - 7, c - 7, c + 7, c + 7], fill=(0, 90, 200, 255), outline=(255, 255, 255, 255), width=3)

    # легенда
    f, fs = _font(13), _font(11)
    d.rectangle([6, SIZE - 56, 232, SIZE - 6], fill=(255, 255, 255, 215))
    d.text((12, SIZE - 53), "Класс пригодности (отн.)", fill=(20, 20, 20, 255), font=f)
    for i, k in enumerate(range(1, 6)):
        x = 12 + i * 42
        d.rectangle([x, SIZE - 32, x + 36, SIZE - 12], fill=CLASS_RGBA[k][:3] + (255,), outline=(60, 60, 60, 255))
        d.text((x + 18, SIZE - 22), str(k), fill=(255, 255, 255, 255) if k >= 4 else (20, 20, 20, 255),
               font=f, anchor="mm")
    d.text((SIZE - 6, SIZE - 4), BASEMAP_ATTRIBUTION if base_ok else "нет подложки",
           fill=(40, 40, 40, 255), font=fs, anchor="rd")
    d.text((SIZE - 6, 6), "круги: 500 м и 1 км", fill=(20, 20, 20, 255), font=fs, anchor="ra")
    out = io.BytesIO()
    img.convert("RGB").save(out, "PNG", optimize=True)
    return out.getvalue()


# ======================================================================
# ТЕКСТЫ
# ======================================================================
"""Тексты сообщений бота. Формулировки — только про ПРИГОДНОСТЬ МЕСТООБИТАНИЙ, не про численность/риск."""


START = (
    "<b>Клещи: потенциал местообитаний</b>\n"
    "Научно-информационный бот по карте пригодности территории Беларуси для иксодовых клещей "
    "(спутниковые данные Sentinel-1/2, Landsat 8/9, ESA WorldCover, Copernicus DEM; модель из 9 факторов).\n\n"
    "Отправьте:\n"
    "• <b>геолокацию</b> (кнопка ниже или 📎 → «Геопозиция»), или\n"
    "• <b>название населённого пункта</b>, например «Заславль», или\n"
    "• <b>координаты</b> «53.9006, 27.5590».\n\n"
    "Я покажу индекс пригодности в точке и в радиусе 500 м и 1 км, фрагмент карты и ссылку на интерактивную карту.\n\n"
    "⚠️ Это <b>не вероятность встречи с клещом</b> и не оценка заболеваемости — см. /about"
)

HELP = (
    "Как пользоваться: пришлите геолокацию, название населённого пункта или координаты.\n"
    "/about — что показывает карта и каковы её ограничения\n"
    "/start — начать сначала"
)

ABOUT = (
    "<b>Что показывает карта</b>\n"
    "Индекс пригодности местообитаний иксодовых клещей (0–100) — потенциал территории поддерживать "
    "популяцию переносчика по ландшафтным признакам: влажность и зелёная масса растительности (NDMI, NDVI), "
    "температура поверхности (LST), лесистость, опушечный эффект, близость к воде и переувлажнённым землям, "
    "пониженность рельефа, весенний радарный сигнал Sentinel-1, доля листопадных пород (ΔNDVI). "
    "Формула: I = 100 · lcScore · Σ(wₖ·fₖ). Разрешение расчёта и растра в боте — 30 м.\n\n"
    "<b>Классы 1–5</b> — относительные (квинтили распределения ненулевых значений по территории), "
    "класс 0 — условия неблагоприятны (застройка, открытые поверхности, вода).\n\n"
    "<b>Ограничения</b>\n"
    "1. Индекс ≠ численность клещей ≠ риск заражения. Риск = опасность × экспозиция × уязвимость.\n"
    "2. Веса факторов экспертные, модель не откалибрована и не валидирована по полевым сборам или данным заболеваемости.\n"
    "3. Растр 30 м (масштаб использования ≈1:60 000–1:100 000): надёжно видны объекты от 60–90 м (≈0,5–1 га); "
    "сад, газон или опушка у дома могут не отображаться.\n"
    "4. Индекс не учитывает сезонную и погодную динамику активности клещей.\n"
    "5. Отсутствие повышенных значений не гарантирует отсутствие клещей.\n\n"
    "<b>Допущения шкалы</b>\n"
    "• Итоговая категория — по 90-му перцентилю индекса в радиусе 500 м (оценка «лучших» условий рядом). "
    "Радиусы 500 м и 1 км выбраны по соглашению, а не выведены из данных.\n"
    "• Классы 1–5 для категории — квинтили этой же статистики по случайной выборке точек Беларуси с ненулевым индексом, "
    "то есть относительные.\n"
    "• Часть факторов имеет собственное разрешение грубее 30 м (температура поверхности ≈100 м), а модель уже включает "
    "окрестности 150–300 м, поэтому значения соседних ячеек и радиусов не независимы.\n"
    "• По названию населённого пункта расчёт выполняется для его центра (точка OSM), по геолокации — для вашей точки "
    "(погрешность GPS обычно 5–20 м).\n\n"
    "<b>Как цитировать</b>: Полюхович А. Н., Архипова Н. В. Интерактивная карта зон активности иксодовых клещей (БГУ), "
    f"{dt.date.today().year}. Интерактивная карта: {MAP_URL}"
)

OUT_OF_BELARUS = (
    "Эта точка находится за пределами Беларуси — карта покрывает только территорию республики. "
    "Пришлите другую геолокацию или название населённого пункта."
)
NOT_FOUND = (
    "Не нашёл такой населённый пункт в Беларуси. Попробуйте уточнить написание "
    "(например, «Новая Боровая») или отправьте геолокацию."
)
NO_DATA = "Для этой точки в растре нет данных (вне контура анализа). Пришлите другую точку."
BUSY = "Слишком часто — подождите пару секунд и повторите."
ERROR = "Не удалось обработать запрос. Попробуйте ещё раз позже."


def map_link(lat, lon, zoom=15):
    q = urlencode({"zoom": zoom, "angle": 0, "lon": f"{lon:.5f}", "lat": f"{lat:.5f}", "panel": "layers"})
    return f"{MAP_URL}?{q}"


def season_note(today=None):
    """Сезонная справка из описания проекта: два пика активности — май–июнь и август–сентябрь."""
    m = (today or dt.date.today()).month
    if m in (5, 6):
        return "🕒 Сейчас — весенне-летний пик активности клещей (май–июнь)."
    if m in (8, 9):
        return "🕒 Сейчас — летне-осенний пик активности клещей (август–сентябрь)."
    if m in (4, 7, 10):
        return "🕒 Сезон активности клещей (апрель–октябрь), но вне основных пиков."
    return "🕒 Сейчас вне сезона активности клещей (ноябрь–март); карта показывает многолетний ландшафтный потенциал."


MEANING = {
    0: "В радиусе анализа преобладают условия, неблагоприятные для клещей по модели (застройка, открытые "
       "поверхности, вода). Отдельные мелкие местообитания (парки, сады, газоны с высокой травой) при "
       "разрешении 30 м могут не отображаться.",
    1: "Ландшафтные условия для клещей по модели заметно хуже, чем на большей части территории.",
    2: "Условия ниже среднего по территории Беларуси.",
    3: "Умеренный потенциал: условия близки к среднему по территории Беларуси.",
    4: "Условия выше среднего (4-й квинтиль): рядом вероятны влажные лесные или кустарниковые участки и опушки.",
    5: "Условия среди наиболее благоприятных (верхний квинтиль) — типично влажные лесные массивы с развитым "
       "подлеском и опушками.",
}


def build_report(label, lat, lon, point_val, point_cls, point_pct, stats, headline_cls, grid):
    lines = [f"<b>{escape(label)}</b>", f"<i>{lat:.4f}° с. ш., {lon:.4f}° в. д.</i>", ""]
    lines.append(f"<b>Потенциал местообитаний клещей: {CLASS_NAMES[headline_cls]}</b> (класс {headline_cls} из 5)")
    lines.append("<i>категория — по 90-му перцентилю индекса в радиусе "
                 f"{HEADLINE_RADIUS} м («лучшие» условия рядом); классы — квинтили этой же статистики по Беларуси</i>")
    lines.append("")
    if point_val is None:
        lines.append("• В ячейке 30 м под точкой: нет данных")
    elif point_val == 0:
        lines.append("• В ячейке 30 м под точкой: 0 (условия неблагоприятны)")
    else:
        lines.append(f"• В ячейке 30 м под точкой: <b>{point_val}</b> из 100 (класс {point_cls}; выше, чем у ≈{point_pct:.0f} % "
                     "ячеек с ненулевым индексом). Значение одной ячейки менее надёжно, чем статистика окружения.")
    for s in stats:
        if s is None:
            continue
        r = s["radius"]
        rr = f"{r} м" if r < 1000 else f"{r/1000:g} км"
        lines.append(
            f"• Радиус {rr}: среднее {s['mean']:.0f}, 90-й перцентиль {s['p90']:.0f}, максимум {s['max']:.0f}; "
            f"классы 4–5 занимают {s['share_high']:.0f} % площади"
        )
    lines += ["", MEANING[headline_cls], "", season_note(), ""]
    lines.append(
        "⚠️ <b>Это индекс пригодности местообитаний, а не вероятность встречи с клещом и не оценка риска заражения.</b> "
        "Модель не валидирована полевыми данными (см. /about)."
    )
    lines.append("")
    lines.append(
        "Профилактика в любой точке в сезон: закрытая одежда, репелленты, осмотр тела после прогулки; "
        "при присасывании клеща — обратитесь в организацию здравоохранения."
    )
    return "\n".join(lines)


def build_short(label, lat, lon, point_val, stats, headline_cls, place=False):
    """Короткий ответ (для телефона). Подробности — по кнопке «Подробнее»."""
    head = next((x for x in stats if x and x["radius"] == HEADLINE_RADIUS), None)
    lines = [f"<b>{escape(label)}</b>", "",
             f"<b>{CLASS_NAMES[headline_cls].capitalize()}</b> (класс {headline_cls} из 5)"]
    if head:
        lines.append(f"Радиус {HEADLINE_RADIUS} м: индекс (0–100) в среднем {head['mean']:.0f}, "
                     f"90-й перцентиль {head['p90']:.0f}; классы 4–5 занимают {head['share_high']:.0f} % площади.")
    if place:
        lines.append("📌 Расчёт по центру населённого пункта; для конкретного места пришлите геолокацию.")
    lines += ["", "⚠️ Это пригодность местообитаний, а не вероятность встречи с клещом и не риск заражения."]
    return "\n".join(lines)


# ======================================================================
# TELEGRAM
# ======================================================================
"""Минимальный клиент Telegram Bot API (long polling) — только requests."""


class TelegramError(RuntimeError):
    pass


class Bot:
    def __init__(self, token, session=None):
        self.base = f"https://api.telegram.org/bot{token}/"
        self.s = session or requests.Session()

    def call(self, method, files=None, timeout=60, **params):
        data = {k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in params.items() if v is not None}
        r = self.s.post(self.base + method, data=data, files=files, timeout=timeout)
        try:
            j = r.json()
        except ValueError:
            raise TelegramError(f"{method}: HTTP {r.status_code}")
        if not j.get("ok"):
            raise TelegramError(f"{method}: {j.get('description')}")
        return j["result"]

    def get_updates(self, offset=None, timeout=50):
        return self.call("getUpdates", offset=offset, timeout=timeout,
                         allowed_updates=["message", "callback_query"])

    def send_message(self, chat_id, text, reply_markup=None):
        return self.call("sendMessage", chat_id=chat_id, text=text, parse_mode="HTML",
                         disable_web_page_preview=True, reply_markup=reply_markup)

    def send_photo(self, chat_id, png_bytes, caption=None):
        return self.call("sendPhoto", files={"photo": ("map.png", png_bytes, "image/png")},
                         chat_id=chat_id, caption=caption, parse_mode="HTML")

    def answer_callback(self, cb_id, text=None):
        return self.call("answerCallbackQuery", callback_query_id=cb_id, text=text)

    def send_chat_action(self, chat_id, action="upload_photo"):
        try:
            self.call("sendChatAction", chat_id=chat_id, action=action, timeout=10)
        except Exception:
            pass


# ======================================================================
# БОТ
# ======================================================================
"""Telegram-бот «Клещи: потенциал местообитаний». Запуск: python bot.py"""


log = logging.getLogger("tickbot")

KEYBOARD = {
    "keyboard": [[{"text": "📍 Отправить мою геолокацию", "request_location": True}]],
    "resize_keyboard": True,
}


class App:
    def __init__(self, bot, grid):
        self.bot, self.grid = bot, grid
        self._last = {}

    # ---------- анализ точки ----------
    def analyze(self, chat_id, lat, lon, label=None, place=False):
        g = self.grid
        if not in_belarus_bbox(lat, lon, BY_BBOX):
            return self.bot.send_message(chat_id, OUT_OF_BELARUS)
        if not g.covers(lon, lat):
            return self.bot.send_message(chat_id, NO_DATA)
        label = label or "Выбранная точка"
        pv = g.value_at(lon, lat)
        pc = g.classify(pv)
        pp = g.percentile_rank(pv)
        stats = [g.window_stats(lon, lat, r) for r in RADII]
        head = next((s for s in stats if s and s["radius"] == HEADLINE_RADIUS), None)
        hc = g.classify_stat(head["p90"]) if head else 0
        text = build_short(label, lat, lon, pv, stats, hc, place=place)
        link = map_link(lat, lon)
        self.bot.send_message(chat_id, text, reply_markup={"inline_keyboard": [
            [{"text": "🗺 Открыть на интерактивной карте", "url": link}],
            [{"text": "ℹ️ Подробнее", "callback_data": f"d|{lat:.5f}|{lon:.5f}"}]]})
        self.bot.send_chat_action(chat_id, "upload_photo")
        try:
            png = render_map(g, lon, lat)
            self.bot.send_photo(chat_id, png, caption=f"Фрагмент карты: {_esc(label)} · круги 500 м и 1 км")
        except Exception:
            log.exception("render failed")

    def details(self, chat_id, lat, lon, label):
        g = self.grid
        pv = g.value_at(lon, lat)
        stats = [g.window_stats(lon, lat, r) for r in RADII]
        head = next((s for s in stats if s and s["radius"] == HEADLINE_RADIUS), None)
        hc = g.classify_stat(head["p90"]) if head else 0
        self.bot.send_message(chat_id, build_report(
            label, lat, lon, pv, g.classify(pv), g.percentile_rank(pv), stats, hc, g))

    # ---------- сообщения ----------
    def on_message(self, m):
        chat_id = m["chat"]["id"]
        uid = m.get("from", {}).get("id", chat_id)
        text = (m.get("text") or "").strip()
        if text.startswith("/"):
            cmd = text.split()[0].split("@")[0].lower()
            if cmd in ("/start",):
                return self.bot.send_message(chat_id, START, reply_markup=KEYBOARD)
            if cmd == "/about":
                return self.bot.send_message(chat_id, ABOUT)
            return self.bot.send_message(chat_id, HELP)
        if not self._allow(uid):
            return self.bot.send_message(chat_id, BUSY)
        if "location" in m:
            loc = m["location"]
            return self.analyze(chat_id, loc["latitude"], loc["longitude"], "Ваша геолокация")
        if not text:
            return self.bot.send_message(chat_id, HELP)
        c = parse_coords(text)
        if c:
            return self.analyze(chat_id, c[0], c[1], "Заданные координаты")
        self.bot.send_chat_action(chat_id, "typing")
        found = geocode_search(text)
        if not found:
            return self.bot.send_message(chat_id, NOT_FOUND)
        if len(found) == 1:
            f = found[0]
            return self.analyze(chat_id, f["lat"], f["lon"], f["label"], place=True)
        kb = [[{"text": f["label"][:60], "callback_data": f"g|{f['lat']:.5f}|{f['lon']:.5f}"}] for f in found]
        return self.bot.send_message(chat_id, "Найдено несколько вариантов — выберите:",
                                     reply_markup={"inline_keyboard": kb})

    def on_callback(self, cb):
        self.bot.answer_callback(cb["id"])
        data = cb.get("data", "")
        if not data.startswith(("g|", "d|")):
            return
        kind, lat, lon = data.split("|")
        chat_id = cb["message"]["chat"]["id"]
        if kind == "d":
            first = (cb["message"].get("text") or "").split("\n")[0] or "Выбранная точка"
            return self.details(chat_id, float(lat), float(lon), first)
        if not self._allow(cb["from"]["id"]):
            return self.bot.send_message(chat_id, BUSY)
        label = None  # подпись берём из текста нажатой кнопки — состояние на сервере не нужно
        for row in cb["message"].get("reply_markup", {}).get("inline_keyboard", []):
            for b in row:
                if b.get("callback_data") == data:
                    label = b["text"]
        self.analyze(chat_id, float(lat), float(lon), label, place=True)

    def _allow(self, uid):
        now = time.time()
        if now - self._last.get(uid, 0) < USER_COOLDOWN_S:
            return False
        self._last[uid] = now
        if len(self._last) > 5000:
            self._last.clear()
        return True

    def handle(self, upd):
        try:
            if "message" in upd:
                self.on_message(upd["message"])
            elif "callback_query" in upd:
                self.on_callback(upd["callback_query"])
        except Exception:
            log.exception("update failed")
            try:
                chat = (upd.get("message") or upd["callback_query"]["message"])["chat"]["id"]
                self.bot.send_message(chat, ERROR)
            except Exception:
                pass


def _esc(s):
    from html import escape
    return escape(s)


def _health_server():
    port = os.environ.get("PORT")
    if not port:
        return

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"tickbot ok")

        def log_message(self, *a):
            pass

    threading.Thread(target=HTTPServer(("0.0.0.0", int(port)), H).serve_forever, daemon=True).start()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not BOT_TOKEN:
        raise SystemExit("Не задан BOT_TOKEN (токен от @BotFather)")
    _health_server()  # порт открываем СРАЗУ, до скачивания растра (иначе Render/пингер не видят сервис)
    ensure_data(GRID_NPY, GRID_META, DATA_URL_NPY, DATA_URL_META)
    grid = Grid(GRID_NPY, GRID_META)
    log.info("Калибровка порогов категории (одноразово, несколько секунд)…")
    log.info("Пороги p90@%s м: %s", HEADLINE_RADIUS, grid.calibrate_p90(HEADLINE_RADIUS))
    log.info("Растр: %sx%s, шаг %s м", grid.w, grid.h, grid.cs)
    bot = Bot(BOT_TOKEN)
    app = App(bot, grid)
    pool = ThreadPoolExecutor(max_workers=4)
    offset = None
    log.info("Бот запущен")
    while True:
        try:
            for upd in bot.get_updates(offset):
                offset = upd["update_id"] + 1
                pool.submit(app.handle, upd)
        except KeyboardInterrupt:
            break
        except Exception:
            log.exception("polling error")
            time.sleep(5)


if __name__ == "__main__":
    main()
