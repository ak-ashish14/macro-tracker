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
POST /api/suggest           ingredient → BM25 → macro filter → Claude → top 5
POST /api/lookup            food name → Claude macro estimate → add to index
"""

import json
import os
import re
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional

from groq import Groq
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
META_PATH = os.path.join(BASE, 'foods_meta.json')

# ── Auth config ────────────────────────────────────────────────────────────
SECRET_KEY        = os.environ.get('SECRET_KEY', 'macrotrack-dev-secret-change-me')
ALGORITHM         = 'HS256'
TOKEN_EXPIRE_DAYS = 30
DATABASE_URL      = os.environ.get('DATABASE_URL', '')

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
    resources['foods'].append(food)
    resources['bm25'] = _build_bm25(resources['foods'])
    with open(META_PATH, 'w') as fh:
        json.dump(resources['foods'], fh, separators=(',', ':'))


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

    groq_key = os.environ.get('GROQ_API_KEY')
    if groq_key:
        resources['groq'] = Groq(api_key=groq_key)
        print("Groq API ready.")
    else:
        resources['groq'] = None
        print("WARNING: GROQ_API_KEY not set – AI food lookup disabled.", file=sys.stderr)
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

def _call_ai(prompt: str, max_tokens: int = 512) -> str:
    client = resources.get('groq')
    if client is None:
        raise RuntimeError('GROQ_API_KEY is not set. Add it in the Render dashboard under Environment Variables.')
    resp = client.chat.completions.create(
        model='llama-3.1-8b-instant',
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


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
        'n': d['name'], 'u': d['unit'],
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


def _suggest_with_ai_new(req: SuggestRequest) -> list[dict]:
    """Ask Groq to generate 5 meal suggestions from scratch using macro budget,
    optional ingredients, and past-meal history for taste alignment."""
    rem = (f"Remaining calories: {req.remaining_kcal:.0f} kcal | "
           f"Protein: {req.remaining_protein:.1f}g | "
           f"Carbs: {req.remaining_carbs:.1f}g | "
           f"Fat: {req.remaining_fat:.1f}g")

    parts = [
        "You are a smart nutrition assistant for an Indian food tracking app.",
        f"User's remaining nutrition budget for today: {rem}",
    ]

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
        "\n- Prioritise healthy, balanced, and genuinely tasty options — think dal, sabzi, grilled protein, rice bowls, wraps, salads, etc."
        "\n- Avoid deep-fried heavy items; prefer grilled, baked, steamed, or lightly sautéed"
        "\nSuggest the TOP 5 dishes that best fit within this nutrition budget."
        "\nEstimate realistic macronutrients for a typical single serving."
        "\nReply ONLY with a valid JSON array of exactly 5 objects — no markdown, no extra text:"
        '\n[{"rank":1,"name":"Dish Name","unit":"serving unit","protein":25.5,"carbs":45.0,"fat":8.0,"kcal":355,"match_pct":85,"reason":"One sentence why this fits"}]'
    )

    prompt = '\n'.join(parts)
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
    # Primary path: ask Groq to generate dishes directly
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
    """Use Groq to estimate kcal burned from a resistance exercise."""
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
    """Use Groq to estimate kcal burned from a free-form activity."""
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


@app.post('/api/lookup', response_model=LookupResponse)
def lookup(req: LookupRequest):
    if not req.food_name.strip():
        raise HTTPException(400, 'food_name must not be empty')
    q = req.food_name.strip().lower()
    for food in resources['foods']:
        if food['n'].lower() == q:
            p100, c100, f100 = food['p'], food['c'], food['f']
            k100 = round(p100*4 + c100*4 + f100*9, 1)
            return LookupResponse(
                name=food['n'], unit=food['u'],
                protein_per_100=p100, carbs_per_100=c100, fat_per_100=f100, kcal_per_100=k100,
                protein_per_serving=food['up'], carbs_per_serving=food['uc'],
                fat_per_serving=food['uf'], kcal_per_serving=food['uk'], source='cache',
            )
    try:
        food = _lookup_nutrition(req.food_name)
    except Exception as e:
        raise HTTPException(503, f'AI lookup failed: {e}')
    _add_food_to_list(food)
    p100, c100, f100 = food['p'], food['c'], food['f']
    k100 = round(p100*4 + c100*4 + f100*9, 1)
    return LookupResponse(
        name=food['n'], unit=food['u'],
        protein_per_100=p100, carbs_per_100=c100, fat_per_100=f100, kcal_per_100=k100,
        protein_per_serving=food['up'], carbs_per_serving=food['uc'],
        fat_per_serving=food['uf'], kcal_per_serving=food['uk'], source='ai',
    )
