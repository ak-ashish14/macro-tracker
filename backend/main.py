"""
MacroTrack – FastAPI backend

Endpoints
---------
GET  /health                liveness check
POST /auth/register         create account (step 1: username+password, step 2: profile)
POST /auth/login            returns JWT token
GET  /auth/me               current user profile
PUT  /auth/profile          update name/age/gender/weight/height
PUT  /auth/password         change password
POST /api/suggest           ingredient → BM25 → macro filter → Gemini → top 5
POST /api/lookup            food name → Gemini macro estimate → add to index
GET  /api/workouts          static workout + exercise list with stable UUIDs
POST /api/exercise/save     persist exercise sets + calories to DB
POST /api/activity/log      persist custom activity to DB
GET  /api/workout/today     return full workout log for user+date
"""

import json
import os
import re
import sys
import time
import uuid as _uuid_mod
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests
import psycopg2
import psycopg2.extras
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from rank_bm25 import BM25Okapi

# ── Paths ─────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(__file__)

# ── Static workout seed data (mirrors WORKOUTS_DATA in frontend) ───────────
_WORKOUT_SEED = [
    {
        'name': 'Workout 1 - Push',
        'exercises': ['Dumbbell Bench Press', 'Incline Dumbbell Bench Press', 'Dip', 'Seated Triceps Press'],
    },
    {
        'name': 'Workout 2 - Pull',
        'exercises': ['Dumbbell Deadlift', 'One-Arm Dumbbell Row', 'Chin-up', 'Alternating Dumbbell Curl'],
    },
    {
        'name': 'Workout 3 - Upper Body A',
        'exercises': ['Seated Dumbbell Press', 'Dumbbell Side Lateral Raise',
                      'Dumbbell Rear Lateral Raise (Seated)', 'Seated Triceps Press'],
    },
    {
        'name': 'Workout 4 - Legs',
        'exercises': ['Dumbbell Goblet Squat', 'Single Leg Split Squat (Dumbbell)',
                      'Dumbbell Romanian Deadlift', 'Dumbbell Lunge (In-Place)'],
    },
    {
        'name': 'Workout 5 - Upper Body B',
        'exercises': ['Close-Grip Dumbbell Bench Press', 'Chin-up',
                      'Two-Arm Dumbbell Row', 'Alternating Dumbbell Curl'],
    },
]

# Deterministic UUIDs derived from names — stable across restarts
_UUID_NS = _uuid_mod.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')  # URL namespace

def _workout_uuid(name: str) -> str:
    return str(_uuid_mod.uuid5(_UUID_NS, f'workout::{name}'))

def _exercise_uuid(workout_name: str, ex_name: str) -> str:
    return str(_uuid_mod.uuid5(_UUID_NS, f'exercise::{workout_name}::{ex_name}'))
META_PATH = os.path.join(BASE, 'foods_meta.json')

# ── Auth config ────────────────────────────────────────────────────────────
SECRET_KEY        = os.environ.get('SECRET_KEY', 'macrotrack-dev-secret-change-me')
ALGORITHM         = 'HS256'
TOKEN_EXPIRE_DAYS = 30
DATABASE_URL      = os.environ.get('DATABASE_URL', '')

# ── Gemini config ──────────────────────────────────────────────────────────
GEMINI_API_KEY  = os.environ.get('GEMINI_API_KEY', '')
GEMINI_ENDPOINT = (
    'https://generativelanguage.googleapis.com/v1beta/models/'
    'gemini-2.5-flash-lite:generateContent'
)

pwd_ctx = CryptContext(schemes=['bcrypt'], deprecated='auto')

# ── Global resources ───────────────────────────────────────────────────────
resources: dict = {}


# ── Database helpers ───────────────────────────────────────────────────────

def _db() -> psycopg2.extensions.connection:
    # Neon (and most cloud PG) require SSL; add sslmode if not already in URL
    url = DATABASE_URL
    if url and 'sslmode' not in url:
        sep = '&' if '?' in url else '?'
        url = url + sep + 'sslmode=require'
    return psycopg2.connect(url, cursor_factory=psycopg2.extras.RealDictCursor)


def _init_db() -> None:
    if not DATABASE_URL:
        print("WARNING: DATABASE_URL not set – user management disabled.", file=sys.stderr)
        return
    conn = _db()
    with conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id            SERIAL PRIMARY KEY,
                    username      VARCHAR(255) UNIQUE NOT NULL,
                    password_hash TEXT         NOT NULL,
                    name          VARCHAR(100) DEFAULT '',
                    age           INTEGER,
                    gender        VARCHAR(10),
                    weight_kg     REAL,
                    height_cm     REAL,
                    goals_protein REAL DEFAULT 150,
                    goals_carbs   REAL DEFAULT 200,
                    goals_fat     REAL DEFAULT 65,
                    created_at    TIMESTAMPTZ  DEFAULT NOW(),
                    email                TEXT,
                    diet_preference      VARCHAR(20),
                    fitness_goal         VARCHAR(20),
                    activity             VARCHAR(20),
                    onboarding_completed BOOLEAN DEFAULT FALSE
                )
            """)
            # Migration: add new columns to existing tables if they don't exist
            for col_sql in [
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS email TEXT",
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS diet_preference VARCHAR(20)",
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS fitness_goal VARCHAR(20)",
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS activity VARCHAR(20)",
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS onboarding_completed BOOLEAN DEFAULT FALSE",
            ]:
                cur.execute(col_sql)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS meals (
                    username VARCHAR(50) NOT NULL,
                    date     DATE        NOT NULL,
                    entries  JSONB       NOT NULL DEFAULT '[]',
                    PRIMARY KEY (username, date)
                )
            """)

            # ── Workout / Exercise static tables ───────────────────────────
            cur.execute("""
                CREATE TABLE IF NOT EXISTS workouts (
                    id   UUID PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS exercises (
                    id            UUID PRIMARY KEY,
                    workout_id    UUID REFERENCES workouts(id) ON DELETE CASCADE,
                    name          TEXT NOT NULL,
                    display_order INT,
                    UNIQUE(workout_id, name)
                )
            """)

            # ── Per-day workout log ────────────────────────────────────────
            cur.execute("""
                CREATE TABLE IF NOT EXISTS daily_workout_logs (
                    id                    UUID PRIMARY KEY,
                    username              TEXT NOT NULL,
                    workout_id            UUID REFERENCES workouts(id),
                    date                  DATE NOT NULL,
                    total_calories_burned INT  DEFAULT 0,
                    created_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Partial unique indexes handle NULL workout_id (activity-only) separately
            cur.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS udx_dwl_workout
                ON daily_workout_logs(username, workout_id, date)
                WHERE workout_id IS NOT NULL
            """)
            cur.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS udx_dwl_noworkout
                ON daily_workout_logs(username, date)
                WHERE workout_id IS NULL
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS exercise_logs (
                    id              UUID PRIMARY KEY,
                    daily_log_id    UUID REFERENCES daily_workout_logs(id) ON DELETE CASCADE,
                    exercise_id     UUID REFERENCES exercises(id),
                    calories_burned INT DEFAULT 0,
                    UNIQUE(daily_log_id, exercise_id)
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS set_logs (
                    id              UUID PRIMARY KEY,
                    exercise_log_id UUID REFERENCES exercise_logs(id) ON DELETE CASCADE,
                    set_number      INT,
                    weight          FLOAT,
                    reps            INT,
                    note            TEXT
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS activity_logs (
                    id               UUID PRIMARY KEY,
                    daily_log_id     UUID REFERENCES daily_workout_logs(id) ON DELETE CASCADE,
                    activity_name    TEXT,
                    duration_minutes INT,
                    intensity        TEXT,
                    calories_burned  INT
                )
            """)

            # ── AI food lookup cache ───────────────────────────────────────
            # Persists all food lookups across restarts; shared across all users.
            # Static foods seeded at startup; AI-found foods added on demand.
            # Keyed on normalised food name (lowercase, stripped).
            cur.execute("""
                CREATE TABLE IF NOT EXISTS food_cache (
                    name_key     TEXT PRIMARY KEY,          -- normalised lookup key
                    name         TEXT NOT NULL,             -- display name
                    unit         TEXT NOT NULL,
                    serving_g    REAL NOT NULL,
                    protein_100  REAL NOT NULL,
                    carbs_100    REAL NOT NULL,
                    fat_100      REAL NOT NULL,
                    kcal_100     REAL NOT NULL,
                    protein_srv  REAL NOT NULL,
                    carbs_srv    REAL NOT NULL,
                    fat_srv      REAL NOT NULL,
                    kcal_srv     REAL NOT NULL,
                    created_at   TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # ── Seed static foods into food_cache (idempotent) ─────────────
            if os.path.exists(META_PATH):
                with open(META_PATH) as _fh:
                    _static_foods = json.load(_fh)
                _seeded = 0
                for _food in _static_foods:
                    _sg   = _food.get('sg', 100)
                    _k100 = round(_food['p']*4 + _food['c']*4 + _food['f']*9, 1)
                    cur.execute("""
                        INSERT INTO food_cache
                            (name_key, name, unit, serving_g,
                             protein_100, carbs_100, fat_100, kcal_100,
                             protein_srv, carbs_srv, fat_srv, kcal_srv)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT (name_key) DO NOTHING
                    """, (
                        _food['n'].lower(), _food['n'], _food['u'], _sg,
                        _food['p'], _food['c'], _food['f'], _k100,
                        _food['up'], _food['uc'], _food['uf'], _food['uk'],
                    ))
                    if cur.rowcount:
                        _seeded += 1
                print(f"food_cache seeded: {_seeded} new static foods inserted ({len(_static_foods)} total).")

            # ── Body-metrics progress log ──────────────────────────────────
            # One row per user per date.  BMI recomputed from user height.
            cur.execute("""
                CREATE TABLE IF NOT EXISTS progress_logs (
                    id         SERIAL PRIMARY KEY,
                    username   TEXT NOT NULL,
                    date       DATE NOT NULL,
                    weight     REAL,
                    waist      REAL,
                    bmi        REAL,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(username, date)
                )
            """)

            # ── Seed static workouts and exercises ─────────────────────────
            for workout in _WORKOUT_SEED:
                w_id = _workout_uuid(workout['name'])
                cur.execute(
                    "INSERT INTO workouts (id, name) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING",
                    (w_id, workout['name'])
                )
                for order, ex_name in enumerate(workout['exercises']):
                    e_id = _exercise_uuid(workout['name'], ex_name)
                    cur.execute(
                        """INSERT INTO exercises (id, workout_id, name, display_order)
                           VALUES (%s, %s, %s, %s) ON CONFLICT (id) DO NOTHING""",
                        (e_id, w_id, ex_name, order)
                    )

    conn.close()
    print("Database ready.")


def _require_db() -> None:
    if not DATABASE_URL:
        raise HTTPException(status_code=503, detail='DATABASE_URL not configured')


# ── Auth helpers ───────────────────────────────────────────────────────────

def _hash_password(plain: str) -> str:
    return pwd_ctx.hash(plain)


def _verify_password(plain: str, hashed: str) -> bool:
    return pwd_ctx.verify(plain, hashed)


def _create_token(username: str) -> str:
    exp = datetime.now(timezone.utc) + timedelta(days=TOKEN_EXPIRE_DAYS)
    return jwt.encode({'sub': username, 'exp': exp}, SECRET_KEY, algorithm=ALGORITHM)


def _get_user_from_token(authorization: Optional[str] = Header(default=None)) -> dict:
    """FastAPI dependency – validates Bearer token and returns user row."""
    _require_db()
    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(status_code=401, detail='Missing or invalid Authorization header')
    token = authorization.split(' ', 1)[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get('sub')
        if not username:
            raise ValueError
    except (JWTError, ValueError):
        raise HTTPException(status_code=401, detail='Invalid or expired token')

    conn = _db()
    try:
        with conn.cursor() as cur:
            cur.execute('SELECT * FROM users WHERE username = %s', (username,))
            user = cur.fetchone()
    finally:
        conn.close()

    if not user:
        raise HTTPException(status_code=401, detail='User not found')
    return dict(user)


def _calc_suggested_goals(weight_kg, fitness_goal, activity) -> dict:
    """BLS framework: weight_lbs × goal×activity multiplier → calories → macros."""
    _DEFAULTS = {'protein': 150, 'carbs': 200, 'fat': 65, 'calories': 2000}
    if not weight_kg or not activity or not fitness_goal:
        return _DEFAULTS

    # Map stored enum → formula key
    GOAL_MAP = {'LEAN_GAIN': 'lean', 'FAT_LOSS': 'cut', 'MAINTENANCE': 'maintain',
                'lean': 'lean', 'cut': 'cut', 'maintain': 'maintain'}
    goal = GOAL_MAP.get(fitness_goal)
    if not goal:
        return _DEFAULTS

    MULTIPLIERS = {
        'cut':      {'sedentary': 8,    'light': 10, 'moderate': 12, 'high': 14},
        'lean':     {'sedentary': None, 'light': 16, 'moderate': 18, 'high': 20},
        'maintain': {'sedentary': 12,   'light': 14, 'moderate': 16, 'high': 18},
    }

    if goal == 'lean' and activity == 'sedentary':
        raise ValueError('Lean gaining requires at least light activity level')

    multiplier = MULTIPLIERS.get(goal, {}).get(activity)
    if not multiplier:
        return _DEFAULTS

    weight_lbs = weight_kg * 2.2046
    calories   = round(weight_lbs * multiplier)

    is_very_overweight = weight_kg > 100
    protein_per_lb = (0.7 if is_very_overweight else 1.1) if goal == 'cut' else 0.9
    protein_g = round(weight_lbs * protein_per_lb)

    fat_g   = round((calories * 0.25) / 9)
    carbs_g = max(0, round((calories - protein_g * 4 - fat_g * 9) / 4))

    return {'protein': protein_g, 'carbs': carbs_g, 'fat': fat_g, 'calories': calories}


# ── BM25 helpers ───────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return re.sub(r'[^a-z0-9 ]', ' ', text.lower()).split()


def _build_bm25(foods: list[dict]) -> BM25Okapi:
    corpus = [_tokenize(f['n']) for f in foods]
    return BM25Okapi(corpus)


def _bm25_search(query: str, k: int = 30) -> list[dict]:
    foods  = resources['foods']
    bm25   = resources['bm25']
    tokens = _tokenize(query)
    scores = bm25.get_scores(tokens)
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    max_score = max(scores[top_idx[0]], 1e-9)
    return [
        {'food': foods[i], 'score': scores[i] / max_score}
        for i in top_idx if scores[i] > 0
    ]


def _add_food_to_list(food: dict) -> None:
    """Add a new food to the in-memory BM25 index (used by /api/suggest fallback).
    Static foods are seeded into food_cache at startup; no file write needed."""
    resources['foods'].append(food)
    resources['bm25'] = _build_bm25(resources['foods'])


# ── food_cache DB helpers ──────────────────────────────────────────────────

def _food_cache_get(name_key: str) -> dict | None:
    """Return cached food row from DB, or None if not found."""
    if not DATABASE_URL:
        return None
    try:
        conn = _db()
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM food_cache WHERE name_key = %s", (name_key,))
            row = cur.fetchone()
        conn.close()
        return dict(row) if row else None
    except Exception as e:
        print(f"food_cache GET error: {e}", file=sys.stderr)
        return None


def _food_cache_put(name_key: str, food: dict) -> None:
    """Persist a Gemini-returned food dict into the DB cache."""
    if not DATABASE_URL:
        return
    try:
        sg = food.get('sg', 100)
        conn = _db()
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO food_cache
                        (name_key, name, unit, serving_g,
                         protein_100, carbs_100, fat_100, kcal_100,
                         protein_srv, carbs_srv, fat_srv, kcal_srv)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (name_key) DO NOTHING
                """, (
                    name_key,
                    food['n'], food['u'], sg,
                    food['p'], food['c'], food['f'],
                    round(food['p']*4 + food['c']*4 + food['f']*9, 1),
                    food['up'], food['uc'], food['uf'], food['uk'],
                ))
        conn.close()
        print(f"[FOOD_ITEM_STORED] name_key={name_key!r} name={food['n']!r} stored in food_cache")
    except Exception as e:
        print(f"food_cache PUT error for {name_key!r}: {e}", file=sys.stderr)


# ── Lifespan ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(META_PATH):
        print("ERROR: foods_meta.json not found.", file=sys.stderr)
        sys.exit(1)

    _init_db()

    with open(META_PATH) as f:
        foods = json.load(f)
    resources['foods'] = foods

    print(f"Building BM25 index for {len(foods)} foods …")
    resources['bm25'] = _build_bm25(foods)

    if GEMINI_API_KEY:
        resources['gemini_ready'] = True
        print("Gemini API ready.")
    else:
        resources['gemini_ready'] = False
        print("WARNING: GEMINI_API_KEY not set – AI food lookup disabled.", file=sys.stderr)
    print(f"Ready – {len(foods)} foods indexed.")
    yield
    resources.clear()


app = FastAPI(title='MacroTrack API', lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(Exception)
async def _global_exc(request: Request, exc: Exception):
    print(f"Unhandled error on {request.url}: {exc}", file=sys.stderr)
    return JSONResponse(status_code=500, content={'detail': str(exc)})


# ── Auth Schemas ───────────────────────────────────────────────────────────

DIET_PREFS       = {'VEG', 'NON_VEG', 'EGGETARIAN'}
FITNESS_GOALS    = {'LEAN_GAIN', 'FAT_LOSS', 'MAINTENANCE'}
VALID_ACTIVITIES = {'sedentary', 'light', 'moderate', 'high'}
_EMAIL_RE        = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')


class RegisterRequest(BaseModel):
    email:    str
    password: str


class LoginRequest(BaseModel):
    email:    str
    password: str


class OnboardingRequest(BaseModel):
    diet_preference: str
    fitness_goal:    str
    activity:  Optional[str]   = None
    name:      Optional[str]   = None
    age:       Optional[int]   = None
    gender:    Optional[str]   = None
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None


class ProfileUpdate(BaseModel):
    name:      Optional[str]   = None
    age:       Optional[int]   = None
    gender:    Optional[str]   = None
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None
    goals_protein:   Optional[float] = None
    goals_carbs:     Optional[float] = None
    goals_fat:       Optional[float] = None
    diet_preference: Optional[str]   = None
    fitness_goal:    Optional[str]   = None
    activity:        Optional[str]   = None


class PasswordChange(BaseModel):
    current_password: str
    new_password: str


def _user_dict(u: dict) -> dict:
    """Serialise a DB user row to the standard API shape."""
    return {
        'username':             u['username'],
        'email':                u.get('email') or u['username'],
        'name':                 u.get('name') or '',
        'age':                  u.get('age'),
        'gender':               u.get('gender'),
        'weight_kg':            u.get('weight_kg'),
        'height_cm':            u.get('height_cm'),
        'diet_preference':      u.get('diet_preference'),
        'fitness_goal':         u.get('fitness_goal'),
        'activity':             u.get('activity'),
        'onboarding_completed': bool(u.get('onboarding_completed')),
        'goals': {
            'protein': u.get('goals_protein', 150),
            'carbs':   u.get('goals_carbs',   200),
            'fat':     u.get('goals_fat',      65),
        },
    }


# ── Suggest / Lookup Schemas ───────────────────────────────────────────────

class SuggestRequest(BaseModel):
    ingredients:       str = ''
    remaining_protein: float = 0
    remaining_carbs:   float = 0
    remaining_fat:     float = 0
    remaining_kcal:    float = 0
    past_meals:        list[str] = []
    diet_preference:   Optional[str] = None   # VEG | NON_VEG | EGGETARIAN


class Dish(BaseModel):
    rank: int; food: str; unit: str
    protein: float; carbs: float; fat: float; kcal: float
    match_pct: int; reason: str


class SuggestResponse(BaseModel):
    dishes: list[Dish]
    candidates_found: int


class LookupRequest(BaseModel):
    food_name: str


class LookupResponse(BaseModel):
    name: str; unit: str
    protein_per_100: float; carbs_per_100: float; fat_per_100: float; kcal_per_100: float
    protein_per_serving: float; carbs_per_serving: float; fat_per_serving: float; kcal_per_serving: float
    source: str


# ── AI helpers ─────────────────────────────────────────────────────────────

def _call_ai(prompt: str, max_tokens: int = 512, retries: int = 3) -> str:
    """Call the Gemini generateContent API with exponential backoff retry."""
    if not GEMINI_API_KEY:
        raise RuntimeError(
            'GEMINI_API_KEY is not set. Add it in the Render dashboard under Environment Variables.'
        )

    url     = f'{GEMINI_ENDPOINT}?key={GEMINI_API_KEY}'
    payload = {
        'contents': [{'parts': [{'text': prompt}]}],
        'generationConfig': {'maxOutputTokens': max_tokens, 'temperature': 0},
    }
    headers = {'Content-Type': 'application/json'}

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=30)
            if resp.status_code == 429:
                # Rate-limited — back off and retry
                wait = 2 ** attempt
                print(f"Gemini rate-limited; retrying in {wait}s (attempt {attempt + 1})", file=sys.stderr)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            # Extract text from Gemini response structure
            candidates = data.get('candidates', [])
            if not candidates:
                raise ValueError(f'Gemini returned no candidates: {data}')
            parts = candidates[0].get('content', {}).get('parts', [])
            if not parts:
                raise ValueError(f'Gemini candidate had no parts: {candidates[0]}')
            return parts[0].get('text', '').strip()
        except requests.exceptions.Timeout as e:
            last_err = e
            wait = 2 ** attempt
            print(f"Gemini timeout on attempt {attempt + 1}; retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
        except requests.exceptions.RequestException as e:
            last_err = e
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f"Gemini request error on attempt {attempt + 1}: {e}; retrying in {wait}s", file=sys.stderr)
                time.sleep(wait)
            else:
                raise RuntimeError(f'Gemini API error after {retries} attempts: {e}') from e

    raise RuntimeError(f'Gemini API failed after {retries} attempts: {last_err}')


# ── Macro helpers ──────────────────────────────────────────────────────────

def _kcal(food: dict) -> float:
    return food['up'] * 4 + food['uc'] * 4 + food['uf'] * 9


def _macro_fits(food: dict, req: SuggestRequest, tolerance: float = 1.3) -> bool:
    for budget, value in [
        (req.remaining_protein, food['up']),
        (req.remaining_carbs,   food['uc']),
        (req.remaining_fat,     food['uf']),
        (req.remaining_kcal,    _kcal(food)),
    ]:
        if budget > 5 and value > budget * tolerance:
            return False
    return True


def _filter_by_macros(candidates, req, target=15):
    for tol in (1.3, 1.6, 9999):
        filtered = [c for c in candidates if _macro_fits(c['food'], req, tol) and _kcal(c['food']) >= 5]
        if len(filtered) >= 5:
            return filtered[:target]
    return candidates[:target]


def _rank_bm25_only(candidates):
    return [{
        'rank': i, 'food': c['food']['n'], 'unit': c['food']['u'],
        'protein': c['food']['up'], 'carbs': c['food']['uc'], 'fat': c['food']['uf'],
        'kcal': round(_kcal(c['food'])), 'match_pct': round(c['score'] * 100),
        'reason': 'Matched by ingredient similarity (AI ranking unavailable)',
    } for i, c in enumerate(candidates[:5], 1)]


def _rank_with_ai(candidates, req):
    rem = (f"Remaining: P {req.remaining_protein:.1f}g  C {req.remaining_carbs:.1f}g  "
           f"F {req.remaining_fat:.1f}g  {req.remaining_kcal:.0f} kcal")
    lines = [f"{i}. {c['food']['n']} (1 {c['food']['u']}): "
             f"{c['food']['up']}g P, {c['food']['uc']}g C, {c['food']['uf']}g F, "
             f"{round(_kcal(c['food']))} kcal [sim {c['score']:.3f}]"
             for i, c in enumerate(candidates, 1)]
    prompt = (
        f'You are a nutrition assistant for an Indian food tracking app.\n'
        f'User ingredients: "{req.ingredients}"\n{rem}\n\n'
        f'Rank the BEST 5 from these candidates:\n' + '\n'.join(lines) + '\n\n'
        f'Reply ONLY with a JSON array of 5 objects with keys: '
        f'rank, name (verbatim), reason (1 sentence), match_pct (0-100).'
    )
    text = _call_ai(prompt)
    m = re.search(r'\[.*?\]', text, re.DOTALL)
    if not m:
        raise ValueError('No JSON array in AI response')
    ranked = json.loads(m.group())
    name_map = {c['food']['n']: c['food'] for c in candidates}
    result = []
    for item in ranked[:5]:
        food = name_map.get(item['name'])
        if food is None:
            for key in name_map:
                if item['name'].lower() in key.lower() or key.lower() in item['name'].lower():
                    food = name_map[key]; break
        if food is None:
            continue
        result.append({
            'rank': item.get('rank', len(result)+1), 'food': food['n'], 'unit': food['u'],
            'protein': food['up'], 'carbs': food['uc'], 'fat': food['uf'],
            'kcal': round(_kcal(food)), 'match_pct': int(item.get('match_pct', 0)),
            'reason': item.get('reason', ''),
        })
    return result


def _lookup_nutrition(food_name: str) -> dict:
    prompt = (
        'You are a nutrition database. Estimate macronutrients for: "' + food_name + '"\n\n'
        'Reply ONLY with a JSON object (no markdown):\n'
        '{"name":str,"unit":str,"serving_g":int,"protein_100":float,"carbs_100":float,"fat_100":float}\n'
        'Example: {"name":"Masala dosa","unit":"piece","serving_g":200,"protein_100":3.2,"carbs_100":28.5,"fat_100":5.1}'
    )
    text = _call_ai(prompt)
    m = re.search(r'\{.*?\}', text, re.DOTALL)
    if not m:
        raise ValueError(f'No JSON in AI response: {text}')
    d = json.loads(m.group())
    sg   = float(d['serving_g'])
    p100 = round(float(d['protein_100']), 1)
    c100 = round(float(d['carbs_100']),   1)
    f100 = round(float(d['fat_100']),     1)
    k100 = round(p100*4 + c100*4 + f100*9, 1)
    return {
        'n': d['name'], 'u': d['unit'], 'sg': sg,
        'p': p100, 'c': c100, 'f': f100, 'k': k100,
        'up': round(p100*sg/100, 2), 'uc': round(c100*sg/100, 2),
        'uf': round(f100*sg/100, 2), 'uk': round(k100*sg/100, 1),
    }


# ── Auth Routes ────────────────────────────────────────────────────────────

@app.post('/auth/register', status_code=201)
def register(req: RegisterRequest):
    _require_db()
    email = req.email.strip().lower()
    if not _EMAIL_RE.match(email):
        raise HTTPException(400, 'Please enter a valid email address')
    if len(req.password) < 6:
        raise HTTPException(400, 'Password must be at least 6 characters')

    conn = _db()
    try:
        with conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        """INSERT INTO users (username, email, password_hash, onboarding_completed)
                           VALUES (%s, %s, %s, FALSE)""",
                        (email, email, _hash_password(req.password))
                    )
                except psycopg2.errors.UniqueViolation:
                    raise HTTPException(409, 'An account with this email already exists')
                cur.execute('SELECT * FROM users WHERE username = %s', (email,))
                user = dict(cur.fetchone())
    finally:
        conn.close()

    return {'token': _create_token(email), 'user': _user_dict(user)}


@app.post('/auth/login')
def login(req: LoginRequest):
    _require_db()
    email = req.email.strip().lower()
    conn = _db()
    try:
        with conn.cursor() as cur:
            # Match by email column first, fall back to username for existing accounts
            cur.execute(
                'SELECT * FROM users WHERE email = %s OR username = %s LIMIT 1',
                (email, email)
            )
            user = cur.fetchone()
    finally:
        conn.close()

    if not user or not _verify_password(req.password, user['password_hash']):
        raise HTTPException(401, 'Invalid email or password')

    return {'token': _create_token(user['username']), 'user': _user_dict(dict(user))}


@app.get('/auth/me')
def me(user: dict = Depends(_get_user_from_token)):
    return _user_dict(user)


@app.post('/auth/onboarding')
def complete_onboarding(req: OnboardingRequest, user: dict = Depends(_get_user_from_token)):
    if req.diet_preference not in DIET_PREFS:
        raise HTTPException(400, f'diet_preference must be one of: {", ".join(sorted(DIET_PREFS))}')
    if req.fitness_goal not in FITNESS_GOALS:
        raise HTTPException(400, f'fitness_goal must be one of: {", ".join(sorted(FITNESS_GOALS))}')
    if req.activity and req.activity not in VALID_ACTIVITIES:
        raise HTTPException(400, f'activity must be one of: {", ".join(sorted(VALID_ACTIVITIES))}')
    # Preferences are updatable; only email is immutable (used as login identifier)

    # Calculate macro goals using BLS formula (requires weight + activity + goal)
    try:
        suggested = _calc_suggested_goals(req.weight_kg, req.fitness_goal, req.activity)
    except ValueError as e:
        raise HTTPException(400, str(e))

    conn = _db()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """UPDATE users SET
                         diet_preference      = %s,
                         fitness_goal         = %s,
                         activity             = COALESCE(%s, activity),
                         onboarding_completed = TRUE,
                         name                 = COALESCE(%s, name),
                         age                  = COALESCE(%s, age),
                         gender               = COALESCE(%s, gender),
                         weight_kg            = COALESCE(%s, weight_kg),
                         height_cm            = COALESCE(%s, height_cm),
                         goals_protein        = %s,
                         goals_carbs          = %s,
                         goals_fat            = %s
                       WHERE username = %s""",
                    (
                        req.diet_preference, req.fitness_goal,
                        req.activity,
                        req.name, req.age, req.gender, req.weight_kg, req.height_cm,
                        suggested['protein'], suggested['carbs'], suggested['fat'],
                        user['username'],
                    )
                )
                cur.execute('SELECT * FROM users WHERE username = %s', (user['username'],))
                updated = dict(cur.fetchone())
    finally:
        conn.close()

    return _user_dict(updated)


@app.put('/auth/profile')
def update_profile(req: ProfileUpdate, user: dict = Depends(_get_user_from_token)):
    # Validate enum fields if provided
    if req.diet_preference is not None and req.diet_preference not in DIET_PREFS:
        raise HTTPException(400, f'diet_preference must be one of: {", ".join(sorted(DIET_PREFS))}')
    if req.fitness_goal is not None and req.fitness_goal not in FITNESS_GOALS:
        raise HTTPException(400, f'fitness_goal must be one of: {", ".join(sorted(FITNESS_GOALS))}')
    if req.activity is not None and req.activity not in VALID_ACTIVITIES:
        raise HTTPException(400, f'activity must be one of: {", ".join(sorted(VALID_ACTIVITIES))}')

    # Auto-recalculate goals when fitness-relevant stats change (weight, activity, goal)
    # Use updated values or fall back to what's stored
    new_weight   = req.weight_kg   or user.get('weight_kg')
    new_activity = req.activity    or user.get('activity')
    new_goal     = req.fitness_goal or user.get('fitness_goal')
    # Only recalculate if the user didn't explicitly send manual goal values
    manual_goals = req.goals_protein or req.goals_carbs or req.goals_fat
    if not manual_goals and (req.weight_kg or req.activity or req.fitness_goal):
        try:
            recalc = _calc_suggested_goals(new_weight, new_goal, new_activity)
            req.goals_protein = recalc['protein']
            req.goals_carbs   = recalc['carbs']
            req.goals_fat     = recalc['fat']
        except ValueError as e:
            raise HTTPException(400, str(e))

    fields, values = [], []
    for col, val in [
        ('name', req.name), ('age', req.age), ('gender', req.gender),
        ('weight_kg', req.weight_kg), ('height_cm', req.height_cm),
        ('goals_protein', req.goals_protein), ('goals_carbs', req.goals_carbs),
        ('goals_fat', req.goals_fat),
        ('diet_preference', req.diet_preference), ('fitness_goal', req.fitness_goal),
        ('activity', req.activity),
    ]:
        if val is not None:
            fields.append(f'{col} = %s')
            values.append(val)
    if not fields:
        raise HTTPException(400, 'No fields to update')
    values.append(user['username'])
    conn = _db()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(f"UPDATE users SET {', '.join(fields)} WHERE username = %s", values)
                cur.execute('SELECT * FROM users WHERE username = %s', (user['username'],))
                updated = dict(cur.fetchone())
    finally:
        conn.close()
    return _user_dict(updated)


@app.put('/auth/password')
def change_password(req: PasswordChange, user: dict = Depends(_get_user_from_token)):
    if not _verify_password(req.current_password, user['password_hash']):
        raise HTTPException(400, 'Current password is incorrect')
    if len(req.new_password) < 6:
        raise HTTPException(400, 'New password must be at least 6 characters')
    conn = _db()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute('UPDATE users SET password_hash = %s WHERE username = %s',
                            (_hash_password(req.new_password), user['username']))
    finally:
        conn.close()
    return {'detail': 'Password updated'}


# ── Meal Sync Routes ───────────────────────────────────────────────────────

@app.get('/api/meals')
def get_meals(user: dict = Depends(_get_user_from_token)):
    conn = _db()
    try:
        with conn.cursor() as cur:
            cur.execute('SELECT date, entries FROM meals WHERE username = %s', (user['username'],))
            rows = cur.fetchall()
    finally:
        conn.close()
    return {str(row['date']): row['entries'] for row in rows}


class MealDayRequest(BaseModel):
    entries: list


@app.put('/api/meals/{date}')
def put_meals(date: str, req: MealDayRequest, user: dict = Depends(_get_user_from_token)):
    conn = _db()
    try:
        with conn:
            with conn.cursor() as cur:
                if req.entries:
                    cur.execute("""
                        INSERT INTO meals (username, date, entries)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (username, date) DO UPDATE SET entries = EXCLUDED.entries
                    """, (user['username'], date, json.dumps(req.entries)))
                else:
                    cur.execute('DELETE FROM meals WHERE username = %s AND date = %s',
                                (user['username'], date))
    finally:
        conn.close()
    return {'ok': True}


# ── Food Routes ────────────────────────────────────────────────────────────

@app.get('/health')
def health():
    return {
        'status': 'ok',
        'foods_indexed': len(resources.get('foods', [])),
        'db': bool(DATABASE_URL),
    }


_DIET_RULES = {
    'VEG': (
        "🚫 STRICT DIETARY RULE — The user is VEGETARIAN. "
        "You MUST suggest ONLY vegetarian dishes. "
        "Absolutely NO meat, poultry, seafood, or eggs in any form. "
        "Dairy (milk, paneer, curd, ghee, cheese) is allowed."
    ),
    'EGGETARIAN': (
        "🚫 STRICT DIETARY RULE — The user is EGGETARIAN (vegetarian + eggs). "
        "You MUST suggest ONLY vegetarian or egg-based dishes. "
        "Absolutely NO meat, poultry, or seafood. "
        "Eggs and dairy are allowed."
    ),
    'NON_VEG': (
        "ℹ️ DIETARY PREFERENCE — The user eats non-vegetarian food. "
        "You may include meat, poultry, seafood, and egg-based dishes where they best fit the macro goals."
    ),
}


def _suggest_with_ai_new(req: SuggestRequest) -> list[dict]:
    """Ask Gemini to generate 5 meal suggestions from scratch using macro budget,
    optional ingredients, past-meal history, and diet preference."""
    rem = (f"Remaining calories: {req.remaining_kcal:.0f} kcal | "
           f"Protein: {req.remaining_protein:.1f}g | "
           f"Carbs: {req.remaining_carbs:.1f}g | "
           f"Fat: {req.remaining_fat:.1f}g")

    parts = [
        "You are a smart nutrition assistant for an Indian food tracking app.",
        f"User's remaining nutrition budget for today: {rem}",
    ]

    # ── Diet preference rule (injected early so Gemini treats it as hard constraint) ──
    diet_pref = (req.diet_preference or '').upper().strip()
    diet_rule = _DIET_RULES.get(diet_pref)
    if diet_rule:
        parts.append(diet_rule)

    if req.ingredients.strip():
        parts.append(f'Ingredients the user has available: "{req.ingredients.strip()}"')
        parts.append(
            "Use these as the primary base for your suggestions. "
            "You may add 1–2 complementary ingredients to better meet the macro goals."
        )
    else:
        parts.append(
            "No specific ingredients were provided. "
            "Suggest any suitable dishes that fit the nutrition budget."
        )

    if req.past_meals:
        meal_list = ', '.join(req.past_meals[:20])
        parts.append(
            f"User's recent meals over the past 3 days: {meal_list}. "
            "Use these to understand their food preferences and suggest dishes they would likely enjoy."
        )

    parts.append(
        "\nRules for your suggestions:"
        "\n- Each dish must be a COMPLETE meal or snack (not a condiment, pickle, sauce, chutney, or side)"
        "\n- Each dish must be 500 kcal or less per serving"
        "\n- Strictly honour the dietary rule stated above — violating it is not acceptable"
        "\n- Prioritise healthy, balanced, and genuinely tasty options — think dal, sabzi, grilled protein, rice bowls, wraps, salads, etc."
        "\n- Avoid deep-fried heavy items; prefer grilled, baked, steamed, or lightly sautéed"
        "\nSuggest the TOP 5 dishes that best fit within this nutrition budget."
        "\nEstimate realistic macronutrients for a typical single serving."
        "\nReply ONLY with a valid JSON array of exactly 5 objects — no markdown, no extra text:"
        '\n[{"rank":1,"name":"Dish Name","unit":"serving unit","protein":25.5,"carbs":45.0,"fat":8.0,"kcal":355,"match_pct":85,"reason":"One sentence why this fits"}]'
    )

    prompt = '\n'.join(parts)

    # ── Log the full prompt so it's visible in Render logs ───────────────────
    print(
        f"\n[SUGGEST_PROMPT] diet={diet_pref or 'not_set'} "
        f"kcal_rem={req.remaining_kcal:.0f} "
        f"ingredients={req.ingredients.strip()!r}\n"
        f"{'─'*60}\n{prompt}\n{'─'*60}"
    )

    text = _call_ai(prompt, max_tokens=1024)
    m = re.search(r'\[.*?\]', text, re.DOTALL)
    if not m:
        raise ValueError(f'No JSON array in AI response: {text[:200]}')
    items = json.loads(m.group())
    result = []
    for item in items[:5]:
        p = float(item.get('protein', 0))
        c = float(item.get('carbs', 0))
        f = float(item.get('fat', 0))
        k = item.get('kcal') or round(p * 4 + c * 4 + f * 9)
        result.append({
            'rank':      int(item.get('rank', len(result) + 1)),
            'food':      str(item.get('name', '')),
            'unit':      str(item.get('unit', 'serving')),
            'protein':   round(p, 1),
            'carbs':     round(c, 1),
            'fat':       round(f, 1),
            'kcal':      int(k),
            'match_pct': min(100, max(0, int(item.get('match_pct', 0)))),
            'reason':    str(item.get('reason', '')),
        })
    return result


@app.post('/api/suggest', response_model=SuggestResponse)
def suggest(req: SuggestRequest):
    # Primary path: ask Gemini to generate dishes directly
    try:
        dishes = _suggest_with_ai_new(req)
        if not dishes:
            raise ValueError('Empty dish list returned')
        return SuggestResponse(dishes=dishes, candidates_found=len(dishes))
    except Exception as primary_err:
        pass  # fall through to BM25 fallback

    # Fallback: BM25 + AI ranking from food DB (requires ingredients)
    if not req.ingredients.strip():
        raise HTTPException(503, 'AI suggestion unavailable and no ingredients provided for fallback.')
    raw        = _bm25_search(req.ingredients, k=30)
    candidates = _filter_by_macros(raw, req, target=15)
    if not candidates:
        raise HTTPException(404, 'No matching foods found for given ingredients.')
    try:
        dishes = _rank_with_ai(candidates, req)
    except Exception:
        dishes = _rank_bm25_only(candidates)
    return SuggestResponse(dishes=dishes, candidates_found=len(candidates))


# ── Exercise calorie estimation ────────────────────────────────────────────

class ExerciseCaloriesRequest(BaseModel):
    exercise_name: str
    sets: list          # [{weight, reps}]
    user_context: dict  # {goal, weight_kg}

class ActivityCaloriesRequest(BaseModel):
    activity: str
    duration: int       # minutes
    intensity: str      # light / moderate / intense

@app.post('/api/exercise/calories')
def estimate_exercise_calories(req: ExerciseCaloriesRequest, user: dict = Depends(_get_user_from_token)):
    """Use Gemini to estimate kcal burned from a resistance exercise."""
    sets_desc = ', '.join(
        f"Set {i+1}: {s.get('weight','?')}kg × {s.get('reps','?')} reps"
        for i, s in enumerate(req.sets)
        if s.get('weight') or s.get('reps')
    )
    if not sets_desc:
        return {'calories_burned': 0}

    goal = req.user_context.get('goal', 'MAINTENANCE')
    weight_kg = req.user_context.get('weight_kg') or user.get('weight_kg') or 75

    prompt = (
        f"Estimate the net calories burned (above resting) for this resistance exercise session.\n"
        f"Exercise: {req.exercise_name}\n"
        f"Sets: {sets_desc}\n"
        f"User body weight: {weight_kg} kg, fitness goal: {goal}\n"
        f"Reply with ONLY a JSON object: {{\"calories_burned\": <integer>}}\n"
        f"No explanation, no markdown, just the JSON."
    )
    try:
        raw = _call_ai(prompt, max_tokens=64)
        # Extract JSON from the response
        import re as _re
        match = _re.search(r'\{[^}]+\}', raw)
        if not match:
            raise ValueError('No JSON in response')
        result = json.loads(match.group())
        burned = max(0, int(result.get('calories_burned', 0)))
        return {'calories_burned': burned}
    except Exception:
        # Rough fallback: ~5 kcal per rep × total volume
        total_reps = sum(int(s.get('reps') or 0) for s in req.sets)
        return {'calories_burned': max(0, total_reps * 5)}


@app.post('/api/activity/calories')
def estimate_activity_calories(req: ActivityCaloriesRequest, user: dict = Depends(_get_user_from_token)):
    """Use Gemini to estimate kcal burned from a free-form activity."""
    weight_kg = user.get('weight_kg') or 75
    prompt = (
        f"Estimate the net calories burned (above resting) for this activity.\n"
        f"Activity: {req.activity}\n"
        f"Duration: {req.duration} minutes\n"
        f"Intensity: {req.intensity}\n"
        f"User body weight: {weight_kg} kg\n"
        f"Reply with ONLY a JSON object: {{\"calories_burned\": <integer>}}\n"
        f"No explanation, no markdown, just the JSON."
    )
    try:
        raw = _call_ai(prompt, max_tokens=64)
        import re as _re
        match = _re.search(r'\{[^}]+\}', raw)
        if not match:
            raise ValueError('No JSON in response')
        result = json.loads(match.group())
        burned = max(0, int(result.get('calories_burned', 0)))
        return {'calories_burned': burned}
    except Exception:
        # MET fallback: moderate running ~8 kcal/min
        return {'calories_burned': max(0, req.duration * 6)}


# ── Workout persistence ────────────────────────────────────────────────────

@app.get('/api/workouts')
def get_workouts():
    """Return static workout + exercise list with stable UUIDs (no auth needed)."""
    conn = _db()
    try:
        with conn.cursor() as cur:
            cur.execute('SELECT id, name FROM workouts ORDER BY name')
            workouts = [dict(r) for r in cur.fetchall()]
            for w in workouts:
                cur.execute(
                    'SELECT id, name, display_order FROM exercises WHERE workout_id = %s ORDER BY display_order',
                    (str(w['id']),)
                )
                w['exercises'] = [
                    {'id': str(e['id']), 'name': e['name'], 'display_order': e['display_order']}
                    for e in cur.fetchall()
                ]
                w['id'] = str(w['id'])
    finally:
        conn.close()
    return workouts


class SetLogItem(BaseModel):
    set: int
    weight: float = 0.0
    reps:   int   = 0
    note:   str   = ''


class SaveExerciseRequest(BaseModel):
    workout_id:      str
    exercise_id:     str
    date:            str
    sets:            list[SetLogItem]
    calories_burned: int


class LogActivityRequest(BaseModel):
    workout_id:       Optional[str] = None
    date:             str
    activity_name:    str
    duration_minutes: int
    intensity:        str
    calories_burned:  int


def _find_or_create_daily_log(cur, username: str, workout_id, date: str) -> str:
    """Return daily_workout_log id, creating it if absent. workout_id may be None."""
    new_id = str(_uuid_mod.uuid4())
    if workout_id is not None:
        cur.execute(
            """INSERT INTO daily_workout_logs (id, username, workout_id, date, total_calories_burned)
               VALUES (%s, %s, %s, %s, 0)
               ON CONFLICT (username, workout_id, date) WHERE workout_id IS NOT NULL DO NOTHING""",
            (new_id, username, workout_id, date)
        )
        cur.execute(
            'SELECT id FROM daily_workout_logs WHERE username=%s AND workout_id=%s AND date=%s',
            (username, workout_id, date)
        )
    else:
        cur.execute(
            """INSERT INTO daily_workout_logs (id, username, workout_id, date, total_calories_burned)
               VALUES (%s, %s, NULL, %s, 0)
               ON CONFLICT (username, date) WHERE workout_id IS NULL DO NOTHING""",
            (new_id, username, date)
        )
        cur.execute(
            'SELECT id FROM daily_workout_logs WHERE username=%s AND date=%s AND workout_id IS NULL',
            (username, date)
        )
    row = cur.fetchone()
    return str(row['id'])


def _recalc_total(cur, daily_log_id: str) -> None:
    """Sum exercise + activity calories and write to daily_workout_logs.total_calories_burned."""
    cur.execute("""
        UPDATE daily_workout_logs SET total_calories_burned = (
            SELECT COALESCE(SUM(el.calories_burned), 0)
            FROM exercise_logs el WHERE el.daily_log_id = %s
        ) + (
            SELECT COALESCE(SUM(al.calories_burned), 0)
            FROM activity_logs al WHERE al.daily_log_id = %s
        )
        WHERE id = %s
    """, (daily_log_id, daily_log_id, daily_log_id))


@app.post('/api/exercise/save')
def save_exercise(req: SaveExerciseRequest, user: dict = Depends(_get_user_from_token)):
    """Persist exercise sets + calories. Upserts on (username, workout_id, date)."""
    conn = _db()
    try:
        with conn:
            with conn.cursor() as cur:
                daily_log_id = _find_or_create_daily_log(
                    cur, user['username'], req.workout_id, req.date
                )

                # Upsert exercise_log
                ex_log_id = str(_uuid_mod.uuid4())
                cur.execute("""
                    INSERT INTO exercise_logs (id, daily_log_id, exercise_id, calories_burned)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (daily_log_id, exercise_id)
                    DO UPDATE SET calories_burned = EXCLUDED.calories_burned
                    RETURNING id
                """, (ex_log_id, daily_log_id, req.exercise_id, req.calories_burned))
                ex_log_id = str(cur.fetchone()['id'])

                # Replace set_logs
                cur.execute('DELETE FROM set_logs WHERE exercise_log_id = %s', (ex_log_id,))
                for s in req.sets:
                    cur.execute("""
                        INSERT INTO set_logs (id, exercise_log_id, set_number, weight, reps, note)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (str(_uuid_mod.uuid4()), ex_log_id, s.set, s.weight, s.reps, s.note))

                _recalc_total(cur, daily_log_id)
    finally:
        conn.close()
    return {'ok': True}


@app.post('/api/activity/log')
def log_activity(req: LogActivityRequest, user: dict = Depends(_get_user_from_token)):
    """Append a custom activity and update total calories for the day."""
    conn = _db()
    try:
        with conn:
            with conn.cursor() as cur:
                daily_log_id = _find_or_create_daily_log(
                    cur, user['username'], req.workout_id or None, req.date
                )

                cur.execute("""
                    INSERT INTO activity_logs
                        (id, daily_log_id, activity_name, duration_minutes, intensity, calories_burned)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (str(_uuid_mod.uuid4()), daily_log_id,
                      req.activity_name, req.duration_minutes,
                      req.intensity, req.calories_burned))

                _recalc_total(cur, daily_log_id)
    finally:
        conn.close()
    return {'ok': True}


@app.get('/api/workout/today')
def get_workout_today(date: str, user: dict = Depends(_get_user_from_token)):
    """Return all logged workout + activity data for the user on a given date."""
    conn = _db()
    try:
        with conn.cursor() as cur:
            # All workout-based daily logs for this user+date
            cur.execute("""
                SELECT dwl.id, dwl.workout_id, dwl.total_calories_burned, w.name AS workout_name
                FROM daily_workout_logs dwl
                JOIN workouts w ON w.id = dwl.workout_id
                WHERE dwl.username = %s AND dwl.date = %s AND dwl.workout_id IS NOT NULL
            """, (user['username'], date))
            workout_log_rows = [dict(r) for r in cur.fetchall()]

            result = {
                'workout_logs':          [],
                'activity_logs':         [],
                'total_calories_burned': 0,
            }

            for wl in workout_log_rows:
                daily_log_id = str(wl['id'])

                # Exercise logs
                cur.execute("""
                    SELECT el.id, el.exercise_id, el.calories_burned, e.name AS exercise_name
                    FROM exercise_logs el
                    JOIN exercises e ON e.id = el.exercise_id
                    WHERE el.daily_log_id = %s
                """, (daily_log_id,))
                exercise_logs = []
                for el in cur.fetchall():
                    el = dict(el)
                    ex_log_id = str(el['id'])
                    cur.execute(
                        'SELECT set_number, weight, reps, note FROM set_logs WHERE exercise_log_id = %s ORDER BY set_number',
                        (ex_log_id,)
                    )
                    el['sets']        = [dict(s) for s in cur.fetchall()]
                    el['id']          = ex_log_id
                    el['exercise_id'] = str(el['exercise_id'])
                    exercise_logs.append(el)

                # Activity logs tied to this workout daily log
                cur.execute("""
                    SELECT activity_name, duration_minutes, intensity, calories_burned
                    FROM activity_logs WHERE daily_log_id = %s
                """, (daily_log_id,))
                act_logs = [dict(r) for r in cur.fetchall()]

                result['workout_logs'].append({
                    'daily_log_id':          daily_log_id,
                    'workout_id':            str(wl['workout_id']),
                    'workout_name':          wl['workout_name'],
                    'total_calories_burned': wl['total_calories_burned'],
                    'exercise_logs':         exercise_logs,
                })
                result['activity_logs'].extend(act_logs)
                result['total_calories_burned'] += (wl['total_calories_burned'] or 0)

            # Activity-only daily log (workout_id IS NULL)
            cur.execute("""
                SELECT id, total_calories_burned
                FROM daily_workout_logs
                WHERE username = %s AND date = %s AND workout_id IS NULL
            """, (user['username'], date))
            ao_row = cur.fetchone()
            if ao_row:
                ao_log_id = str(ao_row['id'])
                cur.execute("""
                    SELECT activity_name, duration_minutes, intensity, calories_burned
                    FROM activity_logs WHERE daily_log_id = %s
                """, (ao_log_id,))
                ao_acts = [dict(r) for r in cur.fetchall()]
                result['activity_logs'].extend(ao_acts)
                result['total_calories_burned'] += (ao_row['total_calories_burned'] or 0)

    finally:
        conn.close()
    return result


# ── Progress Endpoints ─────────────────────────────────────────────────────

class ProgressLogRequest(BaseModel):
    date:  str
    value: float


@app.get('/api/progress')
def get_progress(range: str = 'weekly', user: dict = Depends(_get_user_from_token)):
    """Return body-metric log entries for the requested range (weekly=7d, monthly=30d)."""
    days = 7 if range == 'weekly' else 30
    conn = _db()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT date, weight, waist, bmi
                FROM progress_logs
                WHERE username = %s
                  AND date >= (CURRENT_DATE - INTERVAL '1 day' * %s)
                ORDER BY date ASC
            """, (user['username'], days))
            rows = [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()
    return [
        {
            'date':   str(r['date']),
            'weight': r['weight'],
            'waist':  r['waist'],
            'bmi':    r['bmi'],
        }
        for r in rows
    ]


@app.post('/api/progress/weight')
def log_weight(req: ProgressLogRequest, user: dict = Depends(_get_user_from_token)):
    """Upsert a weight entry; auto-computes BMI from stored height."""
    height_cm = user.get('height_cm') or 0
    bmi = None
    if height_cm > 0:
        h = height_cm / 100.0
        bmi = round(req.value / (h * h), 1)

    conn = _db()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO progress_logs (username, date, weight, bmi)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (username, date)
                    DO UPDATE SET weight = EXCLUDED.weight, bmi = EXCLUDED.bmi
                """, (user['username'], req.date, req.value, bmi))
    finally:
        conn.close()
    return {'ok': True, 'bmi': bmi}


@app.post('/api/progress/waist')
def log_waist(req: ProgressLogRequest, user: dict = Depends(_get_user_from_token)):
    """Upsert a waist measurement entry."""
    conn = _db()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO progress_logs (username, date, waist)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (username, date)
                    DO UPDATE SET waist = EXCLUDED.waist
                """, (user['username'], req.date, req.value))
    finally:
        conn.close()
    return {'ok': True}


@app.post('/api/lookup', response_model=LookupResponse)
def lookup(req: LookupRequest):
    if not req.food_name.strip():
        raise HTTPException(400, 'food_name must not be empty')

    q = req.food_name.strip().lower()

    # ── Tier 1: PostgreSQL food_cache (static + AI-found foods, persisted) ────
    # Static foods (1 023) are seeded here at startup; AI results added on demand.
    row = _food_cache_get(q)
    if row:
        print(f"[LLM_CALL_SKIPPED_DB_HIT] source=food_cache name_key={q!r}")
        return LookupResponse(
            name=row['name'], unit=row['unit'],
            protein_per_100=row['protein_100'], carbs_per_100=row['carbs_100'],
            fat_per_100=row['fat_100'], kcal_per_100=row['kcal_100'],
            protein_per_serving=row['protein_srv'], carbs_per_serving=row['carbs_srv'],
            fat_per_serving=row['fat_srv'], kcal_per_serving=row['kcal_srv'],
            source='db_cache',
        )

    # ── Tier 2: Gemini API — only reached on genuine cache miss ───────────────
    print(f"[LLM_CALL_TRIGGERED] provider=gemini endpoint=gemini-2.5-flash-lite name_key={q!r}")
    try:
        food = _lookup_nutrition(req.food_name)
    except Exception as e:
        raise HTTPException(503, f'AI lookup failed: {e}')

    # Persist to food_cache (survives restarts) + update BM25 index for suggest
    _food_cache_put(q, food)
    _add_food_to_list(food)

    p100, c100, f100 = food['p'], food['c'], food['f']
    k100 = round(p100*4 + c100*4 + f100*9, 1)
    return LookupResponse(
        name=food['n'], unit=food['u'],
        protein_per_100=p100, carbs_per_100=c100, fat_per_100=f100, kcal_per_100=k100,
        protein_per_serving=food['up'], carbs_per_serving=food['uc'],
        fat_per_serving=food['uf'], kcal_per_serving=food['uk'], source='ai',
    )
