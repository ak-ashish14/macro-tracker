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
POST /api/suggest           ingredient → FAISS → macro filter → Claude → top 5
POST /api/lookup            food name → Claude macro estimate → add to FAISS index
"""

import json
import os
import re
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional

import anthropic
import faiss
import numpy as np
import psycopg2
import psycopg2.extras
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ── Paths ─────────────────────────────────────────────────────────────────
BASE       = os.path.dirname(__file__)
INDEX_PATH = os.path.join(BASE, 'foods_index.faiss')
META_PATH  = os.path.join(BASE, 'foods_meta.json')
MODEL_NAME = 'all-MiniLM-L6-v2'

# ── Auth config ────────────────────────────────────────────────────────────
SECRET_KEY     = os.environ.get('SECRET_KEY', 'macrotrack-dev-secret-change-me')
ALGORITHM      = 'HS256'
TOKEN_EXPIRE_DAYS = 30
DATABASE_URL   = os.environ.get('DATABASE_URL', '')

pwd_ctx = CryptContext(schemes=['bcrypt'], deprecated='auto')

# ── Global resources ───────────────────────────────────────────────────────
resources: dict = {}


# ── Database helpers ───────────────────────────────────────────────────────

def _db() -> psycopg2.extensions.connection:
    return psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)


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
                    username      VARCHAR(50)  UNIQUE NOT NULL,
                    password_hash TEXT         NOT NULL,
                    name          VARCHAR(100) DEFAULT '',
                    age           INTEGER,
                    gender        VARCHAR(10),
                    weight_kg     REAL,
                    height_cm     REAL,
                    goals_protein REAL DEFAULT 150,
                    goals_carbs   REAL DEFAULT 200,
                    goals_fat     REAL DEFAULT 65,
                    created_at    TIMESTAMPTZ  DEFAULT NOW()
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


def _calc_suggested_goals(weight_kg, height_cm, age, gender) -> dict:
    """Mifflin-St Jeor BMR × 1.55 activity → macro split."""
    if not all([weight_kg, height_cm, age]):
        return {'protein': 150, 'carbs': 200, 'fat': 65}
    if gender == 'female':
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
    else:
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    tdee = bmr * 1.55
    protein = round(weight_kg * 1.6)   # 1.6 g/kg
    fat     = round(tdee * 0.25 / 9)   # 25% of calories from fat
    carbs   = round((tdee - protein * 4 - fat * 9) / 4)
    return {'protein': max(protein, 50), 'carbs': max(carbs, 50), 'fat': max(fat, 30)}


# ── Lifespan ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        print("ERROR: FAISS index not found. Run  python build_index.py  first.", file=sys.stderr)
        sys.exit(1)

    _init_db()

    print("Loading sentence-transformer model …")
    resources['model'] = SentenceTransformer(MODEL_NAME)

    print("Loading FAISS index …")
    resources['index'] = faiss.read_index(INDEX_PATH)

    with open(META_PATH) as f:
        resources['foods'] = json.load(f)

    if os.environ.get('ANTHROPIC_API_KEY'):
        resources['anthropic'] = anthropic.Anthropic()
    else:
        resources['anthropic'] = None
    resources['claude_cli'] = os.environ.get('CLAUDE_CODE_EXECPATH', 'claude')
    print(f"Ready – {resources['index'].ntotal} food vectors indexed.")
    yield
    resources.clear()


app = FastAPI(title='MacroTrack API', lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)


# ── Auth Schemas ───────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str
    password: str
    name:     str = ''
    age:      Optional[int] = None
    gender:   Optional[str] = None
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None


class LoginRequest(BaseModel):
    username: str
    password: str


class ProfileUpdate(BaseModel):
    name:      Optional[str]   = None
    age:       Optional[int]   = None
    gender:    Optional[str]   = None
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None
    goals_protein: Optional[float] = None
    goals_carbs:   Optional[float] = None
    goals_fat:     Optional[float] = None


class PasswordChange(BaseModel):
    current_password: str
    new_password: str


# ── Suggest / Lookup Schemas ───────────────────────────────────────────────

class SuggestRequest(BaseModel):
    ingredients: str
    remaining_protein: float = 0
    remaining_carbs:   float = 0
    remaining_fat:     float = 0
    remaining_kcal:    float = 0


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


# ── Claude helpers ─────────────────────────────────────────────────────────

def _call_claude(prompt: str, max_tokens: int = 1024) -> str:
    client = resources.get('anthropic')
    if client is not None:
        msg = client.messages.create(
            model='claude-sonnet-4-6', max_tokens=max_tokens,
            messages=[{'role': 'user', 'content': prompt}],
        )
        return msg.content[0].text.strip()
    cli = resources.get('claude_cli', 'claude')
    result = subprocess.run(
        [cli, '--print', '--output-format', 'text'],
        input=prompt, capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f'Claude CLI error: {result.stderr.strip()}')
    return result.stdout.strip()


# ── FAISS helpers ──────────────────────────────────────────────────────────

def _kcal(food: dict) -> float:
    return food['up'] * 4 + food['uc'] * 4 + food['uf'] * 9


def _embed_query(text: str) -> np.ndarray:
    vec = resources['model'].encode([text], convert_to_numpy=True).astype('float32')
    faiss.normalize_L2(vec)
    return vec


def _faiss_search(query_vec: np.ndarray, k: int = 30):
    D, I = resources['index'].search(query_vec, k)
    return [
        {'food': resources['foods'][idx], 'score': float(score)}
        for score, idx in zip(D[0], I[0]) if idx >= 0
    ]


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


def _rank_faiss_only(candidates):
    return [{
        'rank': i, 'food': c['food']['n'], 'unit': c['food']['u'],
        'protein': c['food']['up'], 'carbs': c['food']['uc'], 'fat': c['food']['uf'],
        'kcal': round(_kcal(c['food'])), 'match_pct': round(c['score'] * 100),
        'reason': 'Matched by ingredient similarity (AI ranking unavailable)',
    } for i, c in enumerate(candidates[:5], 1)]


def _rank_with_claude(candidates, req):
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
    text = _call_claude(prompt, max_tokens=1024)
    m = re.search(r'\[.*?\]', text, re.DOTALL)
    if not m:
        raise ValueError('No JSON array in Claude response')
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
    text = _call_claude(prompt, max_tokens=256)
    m = re.search(r'\{.*?\}', text, re.DOTALL)
    if not m:
        raise ValueError(f'No JSON in Claude response: {text}')
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
        '_serving_g': sg,
    }


def _add_food_to_index(food: dict) -> None:
    kcal = round(food['up']*4 + food['uc']*4 + food['uf']*9, 1)
    text = (f"{food['n']}. Served as 1 {food['u']}. Per serving: "
            f"{food['up']}g protein, {food['uc']}g carbs, {food['uf']}g fat, {kcal} kcal.")
    vec = resources['model'].encode([text], convert_to_numpy=True).astype('float32')
    faiss.normalize_L2(vec)
    resources['index'].add(vec)
    resources['foods'].append(food)
    faiss.write_index(resources['index'], INDEX_PATH)
    with open(META_PATH, 'w') as fh:
        json.dump(resources['foods'], fh, separators=(',', ':'))


# ── Auth Routes ────────────────────────────────────────────────────────────

@app.post('/auth/register', status_code=201)
def register(req: RegisterRequest):
    _require_db()
    if len(req.username) < 3:
        raise HTTPException(400, 'Username must be at least 3 characters')
    if len(req.password) < 6:
        raise HTTPException(400, 'Password must be at least 6 characters')

    goals = _calc_suggested_goals(req.weight_kg, req.height_cm, req.age, req.gender)
    conn = _db()
    try:
        with conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        """INSERT INTO users
                           (username, password_hash, name, age, gender, weight_kg, height_cm,
                            goals_protein, goals_carbs, goals_fat)
                           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                        (req.username, _hash_password(req.password),
                         req.name, req.age, req.gender, req.weight_kg, req.height_cm,
                         goals['protein'], goals['carbs'], goals['fat'])
                    )
                except psycopg2.errors.UniqueViolation:
                    raise HTTPException(409, 'Username already taken')
    finally:
        conn.close()

    token = _create_token(req.username)
    return {
        'token': token,
        'user': {
            'username': req.username, 'name': req.name,
            'age': req.age, 'gender': req.gender,
            'weight_kg': req.weight_kg, 'height_cm': req.height_cm,
            'goals': goals,
        }
    }


@app.post('/auth/login')
def login(req: LoginRequest):
    _require_db()
    conn = _db()
    try:
        with conn.cursor() as cur:
            cur.execute('SELECT * FROM users WHERE username = %s', (req.username,))
            user = cur.fetchone()
    finally:
        conn.close()

    if not user or not _verify_password(req.password, user['password_hash']):
        raise HTTPException(401, 'Invalid username or password')

    return {
        'token': _create_token(req.username),
        'user': {
            'username': user['username'], 'name': user['name'],
            'age': user['age'], 'gender': user['gender'],
            'weight_kg': user['weight_kg'], 'height_cm': user['height_cm'],
            'goals': {
                'protein': user['goals_protein'],
                'carbs':   user['goals_carbs'],
                'fat':     user['goals_fat'],
            },
        }
    }


@app.get('/auth/me')
def me(user: dict = Depends(_get_user_from_token)):
    return {
        'username': user['username'], 'name': user['name'],
        'age': user['age'], 'gender': user['gender'],
        'weight_kg': user['weight_kg'], 'height_cm': user['height_cm'],
        'goals': {
            'protein': user['goals_protein'],
            'carbs':   user['goals_carbs'],
            'fat':     user['goals_fat'],
        },
    }


@app.put('/auth/profile')
def update_profile(req: ProfileUpdate, user: dict = Depends(_get_user_from_token)):
    fields, values = [], []
    for col, val in [
        ('name', req.name), ('age', req.age), ('gender', req.gender),
        ('weight_kg', req.weight_kg), ('height_cm', req.height_cm),
        ('goals_protein', req.goals_protein), ('goals_carbs', req.goals_carbs),
        ('goals_fat', req.goals_fat),
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
    return {
        'username': updated['username'], 'name': updated['name'],
        'age': updated['age'], 'gender': updated['gender'],
        'weight_kg': updated['weight_kg'], 'height_cm': updated['height_cm'],
        'goals': {
            'protein': updated['goals_protein'],
            'carbs':   updated['goals_carbs'],
            'fat':     updated['goals_fat'],
        },
    }


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


# ── Food Routes ────────────────────────────────────────────────────────────

@app.get('/health')
def health():
    return {
        'status': 'ok',
        'foods_indexed': resources.get('index') and resources['index'].ntotal,
        'db': bool(DATABASE_URL),
    }


@app.post('/api/suggest', response_model=SuggestResponse)
def suggest(req: SuggestRequest):
    if not req.ingredients.strip():
        raise HTTPException(400, 'ingredients must not be empty')
    query_vec    = _embed_query(req.ingredients)
    raw          = _faiss_search(query_vec, k=30)
    candidates   = _filter_by_macros(raw, req, target=15)
    if not candidates:
        raise HTTPException(404, 'No matching foods found')
    try:
        dishes = _rank_with_claude(candidates, req)
    except Exception:
        dishes = _rank_faiss_only(candidates)
    return SuggestResponse(dishes=dishes, candidates_found=len(candidates))


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
    _add_food_to_index(food)
    p100, c100, f100 = food['p'], food['c'], food['f']
    k100 = round(p100*4 + c100*4 + f100*9, 1)
    return LookupResponse(
        name=food['n'], unit=food['u'],
        protein_per_100=p100, carbs_per_100=c100, fat_per_100=f100, kcal_per_100=k100,
        protein_per_serving=food['up'], carbs_per_serving=food['uc'],
        fat_per_serving=food['uf'], kcal_per_serving=food['uk'], source='claude',
    )
