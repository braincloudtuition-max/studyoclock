#app.py
from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
from pathlib import Path



APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
DATA_HYBRID = DATA_DIR / "all_hybrid_recommendations.csv"
DATA_SVD = DATA_DIR / "svd_recommendations.csv"

app = FastAPI(title="Study O'Clock – Recommendations")
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

#CSV helpers
def read_csv_robust(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, sep=None, engine="python", dtype=str)
    except Exception:
        df = pd.read_csv(path, sep=",", dtype=str)

    
    if df.shape[1] == 1:
        col = df.columns[0]
        df = df[col].str.split(",", expand=True)
        
        first_row = df.iloc[0].tolist()
        looks_like_header = any("user" in str(x).lower() or "topic" in str(x).lower() for x in first_row)
        if looks_like_header:
            df.columns = [str(c).strip() for c in first_row]
            df = df.iloc[1:].reset_index(drop=True)

    
    obj_cols = df.select_dtypes(include="object").columns
    for c in obj_cols:
        df[c] = df[c].str.strip()
    return df

def normalise_hybrid_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    rename_map = {}
    for cand in ["input_user","user_id","user","uid"]:
        if cand in df.columns: rename_map[cand] = "user_id"; break
    for cand in ["input_topic","seed_topic"]:
        if cand in df.columns: rename_map[cand] = "input_topic"; break
    for cand in ["topic_name","item_name","item","recommended_topic"]:
        if cand in df.columns: rename_map[cand] = "topic_name"; break
    for cand in ["content_score","score_content","cbf_score","cosine_score"]:
        if cand in df.columns: rename_map[cand] = "content_score"; break
    for cand in ["cf_score","collab_score","svd_score","predicted_score"]:
        if cand in df.columns: rename_map[cand] = "cf_score"; break
    for cand in ["hybrid_score","score_hybrid","hybrid"]:
        if cand in df.columns: rename_map[cand] = "hybrid_score"; break
    for cand in ["difficulty","level"]:
        if cand in df.columns: rename_map[cand] = "difficulty"; break
    for cand in ["subject","course"]:
        if cand in df.columns: rename_map[cand] = "subject"; break

    df = df.rename(columns=rename_map)
    # numeric scores
    for c in ["content_score","cf_score","hybrid_score"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def normalise_svd_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    rename_map = {}
    for cand in ["user_id","input_user","user","uid"]:
        if cand in df.columns: rename_map[cand] = "user_id"; break
    for cand in ["topic_name","item_name","item"]:
        if cand in df.columns: rename_map[cand] = "topic_name"; break
    for cand in ["predicted_score","score","svd_score","cf_score"]:
        if cand in df.columns: rename_map[cand] = "predicted_score"; break
    df = df.rename(columns=rename_map)
    if "predicted_score" in df.columns:
        df["predicted_score"] = pd.to_numeric(df["predicted_score"], errors="coerce")
    return df

HYBRID = normalise_hybrid_cols(read_csv_robust(DATA_HYBRID))
SVD    = normalise_svd_cols(read_csv_robust(DATA_SVD))

#Dropdowns
USERS = sorted(set(HYBRID.get("user_id", pd.Series(dtype=str))) | set(SVD.get("user_id", pd.Series(dtype=str))))
MODELS = []
if not HYBRID.empty: MODELS.append("hybrid")
if not SVD.empty: MODELS.append("svd")
if not MODELS: MODELS = ["hybrid"]

def topics_for_user(user_id: str) -> list[str]:
    """Return a sorted list of seed input_topic values available for this user in the hybrid CSV."""
    if HYBRID.empty or "input_topic" not in HYBRID.columns:
        return []
    sub = HYBRID[HYBRID["user_id"].astype(str) == str(user_id)]
    return sorted(sub["input_topic"].dropna().unique().tolist())

#Pages
@app.get("/", response_class=HTMLResponse)
def index(
    request: Request,
    user_id: str | None = Query(default=None),
    model: str = Query("hybrid"),
):
    model = model.lower()
    seed_topics = topics_for_user(user_id) if (model == "hybrid" and user_id) else []
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "users": USERS, "models": MODELS,
         "selected_user": user_id, "selected_model": model,
         "seed_topics": seed_topics}
    )

@app.get("/recs", response_class=HTMLResponse)
def show_recs(
    request: Request,
    user_id: str = Query(...),
    model: str = Query("hybrid"),
    k: int = Query(5, ge=1, le=50),
    input_topic: str | None = Query(default=None),
):
    model = model.lower()

    #HYBRID: require a seed (input_topic)
    if model == "hybrid":
        df = HYBRID.copy()
        df = df[df["user_id"].astype(str) == str(user_id)]
        # if available, filter by selected seed topic
        if input_topic:
            df = df[df["input_topic"].astype(str) == str(input_topic)]
        # sort and keep columns
        cols = [c for c in ["subject","topic_name","input_topic","difficulty",
                            "content_score","cf_score","hybrid_score"] if c in df.columns]
        df = df.sort_values(by="hybrid_score", ascending=False)[cols].head(k)
        rows = df.to_dict(orient="records")
        return templates.TemplateResponse(
            "recs.html",
            {"request": request, "user_id": user_id, "model": "hybrid",
             "k": k, "rows": rows, "input_topic": input_topic}
        )

    #SVD
    df = SVD.copy()
    df = df[df["user_id"].astype(str) == str(user_id)]
    df = df.sort_values(by="predicted_score", ascending=False)[["topic_name","predicted_score"]].head(k)
    rows = df.to_dict(orient="records")
    return templates.TemplateResponse(
        "recs.html",
        {"request": request, "user_id": user_id, "model": "svd", "k": k, "rows": rows}
    )

#JSON endpoints
@app.get("/api/hybrid/topics", response_class=JSONResponse)
def api_hybrid_topics(user_id: str):
    return topics_for_user(user_id)

@app.get("/api/recs", response_class=JSONResponse)
def api_recs(user_id: str, model: str = "hybrid", k: int = 5, input_topic: str | None = None):
    model = model.lower()
    if model == "hybrid":
        df = HYBRID.copy()
        df = df[df["user_id"].astype(str) == str(user_id)]
        if input_topic:
            df = df[df["input_topic"].astype(str) == str(input_topic)]
        cols = [c for c in ["topic_name","input_topic","subject","difficulty",
                            "content_score","cf_score","hybrid_score"] if c in df.columns]
        out = df.sort_values(by="hybrid_score", ascending=False)[cols].head(k).to_dict(orient="records")
        return out
    df = SVD.copy()
    df = df[df["user_id"].astype(str) == str(user_id)]
    out = df.sort_values(by="predicted_score", ascending=False)[["topic_name","predicted_score"]].head(k).to_dict(orient="records")
    return out
