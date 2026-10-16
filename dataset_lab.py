"""Dataset Lab: upload any table, explore it, run several ML algorithms on it and compare them."""
from __future__ import annotations

import datetime as dt
import hashlib
import io
import time

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn import datasets as skd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score, roc_curve, silhouette_score)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from theme import ACC, ACC2, AMB, BLUE, DIV, GRN, INK, MUT, PALETTE, RED, SEQ, kpi, panel, show, sty, topbar

try:
    import xgboost as xgb
except ImportError:
    xgb = None

SEED = 42
MISSING_TOKENS = {'': np.nan, '?': np.nan, 'NA': np.nan, 'N/A': np.nan, 'na': np.nan, 'null': np.nan, 'NULL': np.nan,
                  'None': np.nan, 'nan': np.nan, 'NaN': np.nan, '-': np.nan}
SVM_MAX_ROWS = 6000


# ================================================================== loading + automatic cleaning
@st.cache_data(show_spinner=False)
def read_bytes(name: str, raw: bytes) -> pd.DataFrame:
    n = name.lower()
    if n.endswith(('.xlsx', '.xls')):
        return pd.read_excel(io.BytesIO(raw))
    if n.endswith('.json'):
        return pd.read_json(io.BytesIO(raw))
    for enc in ('utf-8-sig', 'latin-1'):
        try:
            return pd.read_csv(io.BytesIO(raw), sep=None, engine='python', encoding=enc)  # sniffs , ; tab |
        except UnicodeDecodeError:
            continue
    raise ValueError('Could not decode the file as text.')


@st.cache_data(show_spinner=False)
def demo_dataset(name: str, root: str) -> pd.DataFrame:
    if name.startswith('Telco churn'):
        return pd.read_csv(f'{root}/sample_customers.csv')
    loader = {'Iris flowers (classification, 3 classes)': skd.load_iris,
              'Wine cultivars (classification, 3 classes)': skd.load_wine,
              'Breast cancer (classification, 2 classes)': skd.load_breast_cancer,
              'Diabetes progression (regression)': skd.load_diabetes}[name]
    b = loader(as_frame=True)
    df = b.frame.copy()
    if hasattr(b, 'target_names') and name != 'Diabetes progression (regression)':
        df['target'] = df['target'].map(dict(enumerate(b.target_names)))
    return df


@st.cache_data(show_spinner=False)
def auto_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Light, safe cleaning every dataset gets. Returns the cleaned frame and a log of what changed."""
    d = df.copy()
    log: list[str] = []
    d.columns = [str(c).strip() or f'col_{i}' for i, c in enumerate(d.columns)]
    if d.columns.duplicated().any():
        d.columns = [f'{c}_{i}' if d.columns.duplicated()[i] else c for i, c in enumerate(d.columns)]
        log.append('Renamed duplicate column names.')
    unnamed = [c for c in d.columns if c.lower().startswith('unnamed') and d[c].is_monotonic_increasing]
    if unnamed:
        d = d.drop(columns=unnamed)
        log.append(f'Dropped exported index column(s): {", ".join(unnamed)}.')
    for c in d.columns:
        s = d[c]
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            s = s.map(lambda v: v.strip() if isinstance(v, str) else v).replace(MISSING_TOKENS)
            nn = s.notna().sum()
            conv = pd.to_numeric(s.astype(str).str.replace(',', '', regex=False), errors='coerce').where(s.notna())
            if nn and conv.notna().sum() >= .9 * nn:
                bad = int(nn - conv.notna().sum())
                s = conv
                log.append(f'`{c}` converted from text to numbers' + (f' ({bad} unreadable value(s) set to missing).' if bad else '.'))
            elif any(k in c.lower() for k in ('date', 'time', 'dt_', '_dt')):
                parsed = pd.to_datetime(s, errors='coerce', format='mixed')
                if nn and parsed.notna().sum() >= .8 * nn:
                    d[f'{c}_year'] = parsed.dt.year
                    d[f'{c}_month'] = parsed.dt.month
                    d[f'{c}_weekday'] = parsed.dt.dayofweek
                    log.append(f'`{c}` parsed as a date and split into year, month and weekday.')
                    d = d.drop(columns=c)
                    continue
        d[c] = s
    empty = [c for c in d.columns if d[c].isna().all()]
    if empty:
        d = d.drop(columns=empty)
        log.append(f'Dropped completely empty column(s): {", ".join(empty)}.')
    const = [c for c in d.columns if d[c].nunique(dropna=True) <= 1]
    if const and len(const) < d.shape[1]:
        d = d.drop(columns=const)
        log.append(f'Dropped constant column(s) that carry no information: {", ".join(const)}.')
    return d, log


def col_kinds(df: pd.DataFrame) -> dict[str, list[str]]:
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    cat = [c for c in df.columns if c not in num]
    n = len(df)
    idlike = []
    for c in df.columns:
        u = df[c].nunique(dropna=True)
        named = c.lower() in ('id', 'index') or c.lower().endswith(('id', '_id', ' id', 'number', 'no'))
        if n > 20 and u >= .95 * n and (c in cat or named or df[c].is_monotonic_increasing):
            idlike.append(c)
    text = [c for c in cat if c not in idlike and df[c].nunique(dropna=True) > max(50, .5 * n)]
    return dict(num=num, cat=cat, id=idlike, text=text)


def detect_task(y: pd.Series) -> str:
    y = y.dropna()
    if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
        return 'classification'
    if y.nunique() <= 15 and np.allclose(y, np.round(y)):
        return 'classification'
    return 'regression'


def sig_of(df: pd.DataFrame) -> str:
    return hashlib.md5(pd.util.hash_pandas_object(df, index=False).values.tobytes() + str(list(df.columns)).encode()).hexdigest()[:12]


# ================================================================== modelling
def as_features(X: pd.DataFrame, cat: list[str]) -> pd.DataFrame:
    X = X.copy()
    for c in cat:
        X[c] = X[c].map(lambda v: v if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v)).astype(object)
        X[c] = X[c].where(X[c].notna(), np.nan)
    return X


def make_pre(num: list[str], cat: list[str]) -> ColumnTransformer:
    parts = []
    if num:
        parts.append(('num', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), num))
    if cat:
        parts.append(('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                                       ('onehot', OneHotEncoder(handle_unknown='infrequent_if_exist', max_categories=30,
                                                                sparse_output=False))]), cat))
    return ColumnTransformer(parts, remainder='drop', verbose_feature_names_out=False)


def zoo(task: str, n_classes: int = 2) -> dict:
    if task == 'classification':
        m = {'Logistic Regression': LogisticRegression(max_iter=3000),
             'Decision Tree': DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, random_state=SEED),
             'Random Forest': RandomForestClassifier(n_estimators=250, min_samples_leaf=2, n_jobs=-1, random_state=SEED),
             'Gradient Boosting': HistGradientBoostingClassifier(random_state=SEED),
             'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=7),
             'Naive Bayes': GaussianNB(),
             'Support Vector Machine': SVC(probability=True, random_state=SEED)}
        if xgb is not None:
            m['XGBoost'] = xgb.XGBClassifier(n_estimators=300, learning_rate=.08, max_depth=5, subsample=.9, colsample_bytree=.9,
                                             random_state=SEED, n_jobs=-1, verbosity=0)
    else:
        m = {'Linear Regression': LinearRegression(),
             'Ridge Regression': Ridge(alpha=1.0),
             'Decision Tree': DecisionTreeRegressor(max_depth=8, min_samples_leaf=5, random_state=SEED),
             'Random Forest': RandomForestRegressor(n_estimators=250, min_samples_leaf=2, n_jobs=-1, random_state=SEED),
             'Gradient Boosting': HistGradientBoostingRegressor(random_state=SEED),
             'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=7),
             'Support Vector Machine': SVR()}
        if xgb is not None:
            m['XGBoost'] = xgb.XGBRegressor(n_estimators=300, learning_rate=.08, max_depth=5, subsample=.9, colsample_bytree=.9,
                                            random_state=SEED, n_jobs=-1, verbosity=0)
    return m


def train_all(df: pd.DataFrame, target: str, features: list[str], task: str, algos: list[str], test_size: float,
              folds: int, max_rows: int, progress) -> dict:
    d = df[features + [target]].dropna(subset=[target])
    notes = []
    if len(d) < len(df):
        notes.append(f'{len(df) - len(d)} row(s) with a missing target were removed.')
    if len(d) > max_rows:
        d = d.sample(max_rows, random_state=SEED)
        notes.append(f'Trained on a random sample of {max_rows:,} rows to keep it fast.')
    kinds = col_kinds(d[features])
    num, cat = kinds['num'], kinds['cat']
    X = as_features(d[features], cat)
    classes = None
    if task == 'classification':
        y_raw = d[target].astype(str)
        vc = y_raw.value_counts()
        rare = vc[vc < 2].index.tolist()
        if rare:
            keep = ~y_raw.isin(rare)
            X, y_raw = X[keep], y_raw[keep]
            notes.append(f'Removed {len(rare)} class(es) with a single example: {", ".join(map(str, rare[:5]))}.')
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        classes = list(le.classes_)
        strat = y
        cv = StratifiedKFold(folds, shuffle=True, random_state=SEED)
        scoring = 'f1_weighted'
    else:
        y = pd.to_numeric(d[target], errors='coerce').to_numpy()
        ok = ~np.isnan(y)
        X, y = X[ok], y[ok]
        strat = None
        cv = KFold(folds, shuffle=True, random_state=SEED)
        scoring = 'r2'
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=SEED, stratify=strat)
    models = zoo(task, len(classes or []))
    rows, fitted, preds, probas, cvs, errors = [], {}, {}, {}, {}, {}
    for i, name in enumerate(algos):
        progress(i / len(algos), f'Training {name} ({i + 1}/{len(algos)})')
        if name == 'Support Vector Machine' and len(Xtr) > SVM_MAX_ROWS:
            errors[name] = f'skipped: SVM is too slow above {SVM_MAX_ROWS:,} training rows'
            continue
        pipe = Pipeline([('prep', make_pre(num, cat)), ('model', models[name])])
        t0 = time.time()
        try:
            cvs[name] = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring=scoring)
            pipe.fit(Xtr, ytr)
            pr = pipe.predict(Xte)
        except Exception as exc:  # one bad algorithm must not stop the others
            errors[name] = f'{type(exc).__name__}: {exc}'
            continue
        secs = time.time() - t0
        fitted[name], preds[name] = pipe, pr
        r = {'Model': name}
        if task == 'classification':
            pb = pipe.predict_proba(Xte) if hasattr(pipe, 'predict_proba') else None
            probas[name] = pb
            r.update({'CV F1': cvs[name].mean(), 'CV ±': cvs[name].std(), 'Accuracy': accuracy_score(yte, pr),
                      'Precision': precision_score(yte, pr, average='weighted', zero_division=0),
                      'Recall': recall_score(yte, pr, average='weighted', zero_division=0),
                      'F1': f1_score(yte, pr, average='weighted', zero_division=0)})
            try:
                r['ROC AUC'] = (roc_auc_score(yte, pb[:, 1]) if len(classes) == 2 else
                                roc_auc_score(yte, pb, multi_class='ovr', average='weighted', labels=list(range(len(classes)))))
            except Exception:
                r['ROC AUC'] = np.nan
        else:
            r.update({'CV R²': cvs[name].mean(), 'CV ±': cvs[name].std(), 'R²': r2_score(yte, pr),
                      'MAE': mean_absolute_error(yte, pr), 'RMSE': float(np.sqrt(mean_squared_error(yte, pr)))})
        r['Seconds'] = secs
        rows.append(r)
    progress(1.0, 'Done')
    board = pd.DataFrame(rows)
    key = 'CV F1' if task == 'classification' else 'CV R²'
    if not board.empty:
        board = board.sort_values(key, ascending=False).reset_index(drop=True)
    if task == 'classification':
        base = float(pd.Series(yte).value_counts(normalize=True).max())
        baseline = f'always predicting the most common class gives accuracy {base:.1%}'
    else:
        baseline = f'always predicting the training mean gives R² {r2_score(yte, np.full(len(yte), np.mean(ytr))):.3f}'
    return dict(target=target, features=features, task=task, classes=classes, num=num, cat=cat, board=board, fitted=fitted,
                preds=preds, probas=probas, cvs=cvs, errors=errors, Xte=Xte, yte=yte, n_train=len(Xtr), notes=notes,
                baseline=baseline, key=key, importance={})


def perm_importance(res: dict, name: str) -> pd.Series:
    if name not in res['importance']:
        Xte, yte = res['Xte'], res['yte']
        if len(Xte) > 1200:
            idx = np.random.RandomState(SEED).choice(len(Xte), 1200, replace=False)
            Xte, yte = Xte.iloc[idx], yte[idx]
        pi = permutation_importance(res['fitted'][name], Xte, yte, n_repeats=5, random_state=SEED,
                                    scoring='f1_weighted' if res['task'] == 'classification' else 'r2')
        res['importance'][name] = pd.Series(pi.importances_mean, index=res['features']).sort_values(ascending=False)
    return res['importance'][name]


# ================================================================== page sections
def section_overview(df: pd.DataFrame, raw: pd.DataFrame, log: list[str], kinds: dict) -> None:
    miss = df.isna().sum()
    k = st.columns(6)
    with k[0]: kpi('Rows', f'{len(df):,}', f'{len(raw):,} uploaded', ACC)
    with k[1]: kpi('Columns', f'{df.shape[1]}', f"{len(kinds['num'])} numeric · {len(kinds['cat'])} categorical", ACC2)
    with k[2]: kpi('Missing cells', f'{miss.sum() / max(df.size, 1):.1%}', f'{int(miss.sum()):,} empty values', AMB if miss.sum() else GRN)
    with k[3]: kpi('Duplicate rows', f'{int(df.duplicated().sum()):,}', 'exact repeats', AMB if df.duplicated().any() else GRN)
    with k[4]: kpi('ID-like columns', f"{len(kinds['id'])}", ', '.join(kinds['id'][:2]) or 'none found', BLUE)
    with k[5]: kpi('Memory', f'{df.memory_usage(deep=True).sum() / 1e6:.1f} MB', 'in this session', MUT)

    ins = []
    if log:
        ins += log
    worst = miss[miss > 0].sort_values(ascending=False)
    if len(worst):
        ins.append(f"Most missing: <b>{worst.index[0]}</b> ({worst.iloc[0] / len(df):.0%} empty). Models fill gaps with the median or most common value.")
    if kinds['id']:
        ins.append(f"<b>{', '.join(kinds['id'])}</b> look like identifiers and are left out of modelling by default.")
    if kinds['text']:
        ins.append(f"<b>{', '.join(kinds['text'])}</b> look like free text (too many unique values) and are left out by default.")
    if kinds['num']:
        sk = df[kinds['num']].skew(numeric_only=True).abs().sort_values(ascending=False)
        if len(sk) and sk.iloc[0] > 2:
            ins.append(f"<b>{sk.index[0]}</b> is strongly skewed (skew {sk.iloc[0]:.1f}); tree models handle that better than linear ones.")
    if not ins:
        ins.append('The file looks clean: no missing values, duplicates or identifier columns found.')
    st.markdown("<div class='insight'><ul>" + ''.join(f'<li>{x}</li>' for x in ins) + "</ul></div>", unsafe_allow_html=True)

    with st.container(border=True):
        panel('Data preview')
        st.dataframe(df.head(200), use_container_width=True, height=300)
    a, b = st.columns([1.3, 1])
    with a, st.container(border=True):
        panel('Column profile')
        prof = pd.DataFrame({'Column': df.columns, 'Type': [('numeric' if c in kinds['num'] else 'categorical') + (' · id' if c in kinds['id'] else '')
                                                            + (' · text' if c in kinds['text'] else '') for c in df.columns],
                             'Missing %': (miss / max(len(df), 1) * 100).round(1).values, 'Unique': df.nunique().values,
                             'Example': [str(df[c].dropna().iloc[0])[:30] if df[c].notna().any() else '' for c in df.columns]})
        st.dataframe(prof, hide_index=True, use_container_width=True, height=360,
                     column_config={'Missing %': st.column_config.ProgressColumn('Missing %', min_value=0, max_value=100, format='%.1f%%')})
    with b, st.container(border=True):
        panel('Summary statistics (numeric)')
        if kinds['num']:
            st.dataframe(df[kinds['num']].describe().T.round(3), use_container_width=True, height=360)
        else:
            st.info('No numeric columns.')


def section_explore(df: pd.DataFrame, kinds: dict) -> None:
    num, cat = kinds['num'], [c for c in kinds['cat'] if c not in kinds['text']]
    view = df.sample(5000, random_state=SEED) if len(df) > 5000 else df
    a, b = st.columns(2)
    with a, st.container(border=True):
        panel('Distribution of one column')
        col = st.selectbox('Column', list(df.columns), key='lab_dist_col')
        if col in num:
            fig = px.histogram(view, x=col, nbins=40, marginal='box', color_discrete_sequence=[ACC])
            fig.update_traces(marker_line_color='#0b1020', marker_line_width=1)
        else:
            vc = df[col].astype(str).value_counts().head(20)
            fig = go.Figure(go.Bar(x=vc.values, y=vc.index, orientation='h', marker=dict(color=vc.values, colorscale=SEQ),
                                   text=vc.values, textposition='outside', cliponaxis=False))
            fig.update_yaxes(autorange='reversed')
        show(sty(fig, 360), 'lab_dist')
    with b, st.container(border=True):
        panel('Missing values per column')
        m = (df.isna().mean() * 100).sort_values(ascending=False)
        m = m[m > 0].head(25)
        if m.empty:
            st.success('No missing values in any column.')
        else:
            fig = go.Figure(go.Bar(x=m.values, y=m.index, orientation='h', marker_color=AMB, text=[f'{v:.1f}%' for v in m.values],
                                   textposition='outside', cliponaxis=False))
            fig.update_yaxes(autorange='reversed')
            show(sty(fig, 360, xaxis_title='% missing'), 'lab_missing')

    if len(num) >= 2:
        c1, c2 = st.columns([1.3, 1])
        corr = df[num[:30]].corr()
        with c1, st.container(border=True):
            panel('Correlation heatmap (numeric columns)')
            fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1, colorscale=DIV,
                                       text=np.round(corr.values, 2), texttemplate='%{text}' if len(corr) <= 12 else None))
            show(sty(fig, 460), 'lab_corr')
        with c2, st.container(border=True):
            panel('Strongest relationships')
            pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), 1)).stack().rename('r').reset_index()
            pairs.columns = ['Column A', 'Column B', 'r']
            pairs = pairs.reindex(pairs.r.abs().sort_values(ascending=False).index).head(12)
            st.dataframe(pairs.round(3), hide_index=True, use_container_width=True, height=300,
                         column_config={'r': st.column_config.ProgressColumn('Correlation r', min_value=-1, max_value=1, format='%.2f')})
            out = {c: int(((df[c] < df[c].quantile(.25) - 1.5 * (df[c].quantile(.75) - df[c].quantile(.25))) |
                           (df[c] > df[c].quantile(.75) + 1.5 * (df[c].quantile(.75) - df[c].quantile(.25)))).sum()) for c in num}
            out = {k: v for k, v in sorted(out.items(), key=lambda kv: -kv[1]) if v}
            st.markdown("<div class='note'>Outliers (1.5 × IQR rule): " +
                        (', '.join(f'<b>{k}</b> {v}' for k, v in list(out.items())[:6]) if out else 'none found') + '</div>', unsafe_allow_html=True)

    with st.container(border=True):
        panel('Relationship explorer')
        cols = list(df.columns)
        x1, x2, x3 = st.columns(3)
        x = x1.selectbox('X axis', cols, index=0, key='lab_rx')
        y = x2.selectbox('Y axis', cols, index=min(1, len(cols) - 1), key='lab_ry')
        color = x3.selectbox('Colour by', ['(none)'] + cat + num, key='lab_rc')
        cc = None if color == '(none)' else color
        if x in num and y in num:
            fig = px.scatter(view, x=x, y=y, color=cc, opacity=.75, color_discrete_sequence=PALETTE,
                             color_continuous_scale=SEQ if cc in num else None)
        elif x in num or y in num:
            nc, cc_ = (x, y) if x in num else (y, x)
            fig = px.box(view.assign(**{cc_: view[cc_].astype(str)}), x=cc_, y=nc, color=cc if cc in cat else None, color_discrete_sequence=PALETTE)
        else:
            ct = pd.crosstab(df[y].astype(str), df[x].astype(str))
            fig = go.Figure(go.Heatmap(z=ct.values, x=ct.columns, y=ct.index, colorscale=SEQ, text=ct.values, texttemplate='%{text}'))
        show(sty(fig, 420, xaxis_title=x, yaxis_title=y), 'lab_rel')


def section_train(df: pd.DataFrame, kinds: dict, sig: str) -> None:
    cols = list(df.columns)
    guess = next((c for c in cols if c.lower() in ('target', 'label', 'class', 'churn', 'y', 'outcome', 'price', 'species')), cols[-1])
    t1, t2 = st.columns([1.4, 1])
    target = t1.selectbox('🎯 Target column (what to predict)', cols, index=cols.index(guess), key=f'lab_target_{sig}')
    auto = detect_task(df[target])
    task = t2.radio('Problem type', ['classification', 'regression'], index=0 if auto == 'classification' else 1, horizontal=True,
                    key=f'lab_task_{sig}_{target}', help=f'Auto-detected: {auto}')
    if task == 'regression' and not pd.api.types.is_numeric_dtype(df[target]):
        st.error('Regression needs a numeric target. Pick another column or switch to classification.')
        return
    if task == 'classification' and df[target].nunique() > 50:
        st.warning(f'`{target}` has {df[target].nunique()} different values. That is a lot of classes; regression may fit better.')

    default_feats = [c for c in cols if c != target and c not in kinds['id'] and c not in kinds['text']]
    all_algos = list(zoo(task))
    with st.form(f'lab_form_{sig}'):
        feats = st.multiselect('Input features', [c for c in cols if c != target], default=default_feats)
        f1, f2, f3, f4 = st.columns(4)
        algos = f1.multiselect('Algorithms', all_algos, default=all_algos)
        test_size = f2.slider('Test set size', .1, .4, .2, .05)
        folds = f3.slider('Cross-validation folds', 3, 10, 5)
        max_rows = f4.number_input('Max rows to train on', 500, 500_000, 30_000, 500)
        go_btn = st.form_submit_button('🚀 Train & compare models', type='primary', use_container_width=True)
    if go_btn:
        if not feats or not algos:
            st.error('Pick at least one feature and one algorithm.')
            return
        bar = st.progress(0.0, 'Preparing data')
        try:
            res = train_all(df, target, feats, task, algos, test_size, folds, int(max_rows), lambda v, t: bar.progress(v, t))
        except Exception as exc:
            bar.empty()
            st.error(f'Training failed: {type(exc).__name__}: {exc}')
            return
        bar.empty()
        res['sig'] = sig
        if not res['board'].empty:
            perm_importance(res, res['board'].iloc[0]['Model'])
        st.session_state['lab_res'] = res

    res = st.session_state.get('lab_res')
    if not res or res.get('sig') != sig or res['target'] != target:
        st.info('Choose a target and press **Train & compare models**. Every algorithm is cross-validated on the training split '
                'and then scored once on the held-out test split.')
        return
    render_results(res)


def render_results(res: dict) -> None:
    board, task, key = res['board'], res['task'], res['key']
    for m in res['notes']:
        st.caption('ℹ️ ' + m)
    for n, e in res['errors'].items():
        st.warning(f'**{n}** {e}')
    if board.empty:
        st.error('No model finished training.')
        return
    best = board.iloc[0]
    main = 'F1' if task == 'classification' else 'R²'
    k = st.columns(4)
    with k[0]: kpi('Best algorithm', best['Model'], f'ranked by {key}', ACC)
    with k[1]: kpi(key, f"{best[key]:.3f}", f"± {best['CV ±']:.3f} across folds", ACC2)
    with k[2]: kpi(f'Test {main}', f"{best[main]:.3f}", 'held-out data, used once', GRN)
    if task == 'classification':
        with k[3]: kpi('Test accuracy', f"{best['Accuracy']:.1%}", f"ROC AUC {best['ROC AUC']:.3f}" if best['ROC AUC'] == best['ROC AUC'] else '', AMB)
    else:
        with k[3]: kpi('Test RMSE', f"{best['RMSE']:,.3f}", f"MAE {best['MAE']:,.3f}", AMB)

    imp = res['importance'].get(best['Model'])
    lines = [f"<b>{best['Model']}</b> is the strongest of {len(board)} algorithms on <b>{res['target']}</b> "
             f"({key} {best[key]:.3f}, test {main} {best[main]:.3f}).",
             f"For comparison, {res['baseline']}."]
    if len(board) > 1:
        w = board.iloc[-1]
        lines.append(f"Weakest: <b>{w['Model']}</b> ({key} {w[key]:.3f}). The gap between best and worst is {best[key] - w[key]:.3f}.")
    gap = best[key] - best[main]
    if abs(gap) > .08:
        lines.append(f"Cross-validation and test scores differ by {gap:+.3f}: the test split may be small or unusual, so treat the exact number with care.")
    if imp is not None and len(imp):
        lines.append('Most influential inputs for the best model: ' + ', '.join(f'<b>{c}</b>' for c in imp.index[:3]) + '.')
    st.markdown("<div class='insight'><ul>" + ''.join(f'<li>{x}</li>' for x in lines) + "</ul></div>", unsafe_allow_html=True)

    a, b = st.columns([1.25, 1])
    with a, st.container(border=True):
        panel('Leaderboard')
        fmt = {c: st.column_config.NumberColumn(c, format='%.3f') for c in board.columns if c not in ('Model',)}
        fmt['Seconds'] = st.column_config.NumberColumn('Seconds', format='%.2f')
        st.dataframe(board, hide_index=True, use_container_width=True, column_config=fmt)
    with b, st.container(border=True):
        panel('Score spread across CV folds')
        fig = go.Figure()
        for i, n in enumerate(board['Model']):
            fig.add_trace(go.Box(y=res['cvs'][n], name=n, marker_color=PALETTE[i % len(PALETTE)], boxpoints='all', jitter=.4))
        show(sty(fig, 330, showlegend=False, yaxis_title=key), 'lab_cvbox')

    with st.container(border=True):
        panel('Test metrics by algorithm')
        mets = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC'] if task == 'classification' else ['R²']
        long = board.melt(id_vars='Model', value_vars=[m for m in mets if m in board], var_name='Metric', value_name='Score')
        fig = px.bar(long, x='Model', y='Score', color='Metric', barmode='group', color_discrete_sequence=PALETTE, text_auto='.2f')
        show(sty(fig, 360, yaxis_range=[min(0, long.Score.min()), 1.08]), 'lab_metrics')

    pick = st.selectbox('Inspect algorithm', list(board['Model']), key='lab_pick')
    yte, pr = res['yte'], res['preds'][pick]
    c1, c2 = st.columns(2)
    if task == 'classification':
        cl = res['classes']
        with c1, st.container(border=True):
            panel(f'Confusion matrix · {pick}')
            cm = confusion_matrix(yte, pr, labels=list(range(len(cl))))
            pct = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
            fig = go.Figure(go.Heatmap(z=pct, x=[f'pred {c}' for c in cl], y=[f'true {c}' for c in cl], colorscale=SEQ, zmin=0, zmax=1,
                                       text=[[f'{v}<br>{p:.0%}' for v, p in zip(r1, r2)] for r1, r2 in zip(cm, pct)], texttemplate='%{text}',
                                       showscale=False))
            fig.update_yaxes(autorange='reversed')
            show(sty(fig, 380), 'lab_cm')
        with c2, st.container(border=True):
            panel('ROC curves' + (' · all algorithms' if len(cl) == 2 else f' · {pick}, one class vs rest'))
            fig = go.Figure()
            fig.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, line=dict(color=MUT, dash='dot'))
            if len(cl) == 2:
                for i, n in enumerate(board['Model']):
                    pb = res['probas'].get(n)
                    if pb is not None:
                        fpr, tpr, _ = roc_curve(yte, pb[:, 1])
                        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{n} ({roc_auc_score(yte, pb[:, 1]):.3f})", mode='lines',
                                                 line=dict(width=3 if n == pick else 1.6, color=PALETTE[i % len(PALETTE)])))
            elif res['probas'].get(pick) is not None:
                pb = res['probas'][pick]
                for i, c in enumerate(cl[:10]):
                    if (yte == i).any() and (yte != i).any():
                        fpr, tpr, _ = roc_curve(yte == i, pb[:, i])
                        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=str(c), mode='lines', line=dict(width=2.4, color=PALETTE[i % len(PALETTE)])))
            show(sty(fig, 380, xaxis_title='False positive rate', yaxis_title='True positive rate', legend=dict(orientation='v', x=.55, y=.05)), 'lab_roc')
    else:
        with c1, st.container(border=True):
            panel(f'Actual vs predicted · {pick}')
            lo, hi = float(min(yte.min(), pr.min())), float(max(yte.max(), pr.max()))
            fig = go.Figure(go.Scatter(x=yte, y=pr, mode='markers', marker=dict(color=ACC2, size=7, opacity=.7, line=dict(width=.5, color='#0b1020'))))
            fig.add_shape(type='line', x0=lo, y0=lo, x1=hi, y1=hi, line=dict(color=RED, dash='dash'))
            show(sty(fig, 380, xaxis_title='Actual', yaxis_title='Predicted', showlegend=False), 'lab_avp')
        with c2, st.container(border=True):
            panel('Residuals (actual − predicted)')
            fig = px.histogram(x=yte - pr, nbins=40, color_discrete_sequence=[ACC])
            fig.add_vline(x=0, line=dict(color=RED, dash='dash'))
            show(sty(fig, 380, xaxis_title='Residual', yaxis_title='Count'), 'lab_resid')

    with st.container(border=True):
        panel(f'Feature importance · {pick} (permutation, on the test split)')
        with st.spinner('Measuring how much each input matters...'):
            imp = perm_importance(res, pick).head(20).sort_values()
        fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation='h', marker=dict(color=imp.values, colorscale=SEQ)))
        show(sty(fig, 120 + 26 * len(imp), xaxis_title=f'Drop in {"F1" if task == "classification" else "R²"} when the column is shuffled'), 'lab_imp')
        st.caption('Works the same way for every algorithm: shuffle one input, re-score, and see how much worse the model gets. '
                   'Importance shows what the model uses, not what causes the outcome.')


def section_cluster(df: pd.DataFrame, kinds: dict, sig: str) -> None:
    num = kinds['num']
    cat = [c for c in kinds['cat'] if c not in kinds['text'] and c not in kinds['id']]
    feats = st.multiselect('Columns to group on', num + cat, default=[c for c in num if c not in kinds['id']][:12], key=f'lab_cl_feats_{sig}')
    if len(feats) < 2:
        st.info('Pick at least two columns. K-Means groups rows that look alike; no target is needed.')
        return
    d = df[feats].copy()
    d = d.sample(20000, random_state=SEED) if len(d) > 20000 else d
    nn = [c for c in feats if c in num]
    cc = [c for c in feats if c not in num]
    Z = make_pre(nn, cc).fit_transform(as_features(d, cc))
    sample = Z if len(Z) <= 3000 else Z[np.random.RandomState(SEED).choice(len(Z), 3000, replace=False)]

    @st.cache_data(show_spinner=False)
    def scan(_Z, _s, key):
        ks = list(range(2, 11))
        inert, sil = [], []
        for k in ks:
            km = KMeans(k, n_init=10, random_state=SEED).fit(_s)
            inert.append(km.inertia_)
            sil.append(silhouette_score(_s, km.labels_))
        return ks, inert, sil

    ks, inert, sil = scan(Z, sample, (sig, tuple(feats)))
    best_k = ks[int(np.argmax(sil))]
    a, b = st.columns([1, 1.6])
    with a, st.container(border=True):
        panel('How many groups? Elbow and silhouette')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ks, y=sil, name='Silhouette (higher is better)', mode='lines+markers', line=dict(color=ACC2, width=3)))
        fig.add_trace(go.Scatter(x=ks, y=np.array(inert) / max(inert), name='Inertia (scaled)', mode='lines+markers', line=dict(color=ACC, width=2, dash='dot')))
        show(sty(fig, 300, xaxis_title='k'), 'lab_elbow')
        k = st.slider('Number of clusters', 2, 10, best_k, key=f'lab_k_{sig}')
        st.caption(f'Suggested k = {best_k} (highest silhouette, {max(sil):.2f}).')
    km = KMeans(k, n_init=10, random_state=SEED).fit(Z)
    lab = km.labels_
    pca = PCA(2, random_state=SEED).fit(Z)
    P = pca.transform(Z)
    with b, st.container(border=True):
        panel(f'Clusters in 2-D (PCA keeps {pca.explained_variance_ratio_.sum():.0%} of the variation)')
        fig = px.scatter(x=P[:, 0], y=P[:, 1], color=[f'Cluster {i}' for i in lab], color_discrete_sequence=PALETTE, opacity=.8,
                         category_orders={'color': [f'Cluster {i}' for i in range(k)]})
        fig.update_traces(marker=dict(size=7, line=dict(width=.4, color='#0b1020')))
        show(sty(fig, 430, xaxis_title='PC 1', yaxis_title='PC 2'), 'lab_pca')
    out = d.assign(cluster=lab)
    c1, c2 = st.columns([1, 1.6])
    with c1, st.container(border=True):
        panel('Cluster sizes')
        sz = out.cluster.value_counts().sort_index()
        fig = go.Figure(go.Pie(labels=[f'Cluster {i}' for i in sz.index], values=sz.values, hole=.6, marker=dict(colors=PALETTE), sort=False))
        show(sty(fig, 330), 'lab_csize')
    with c2, st.container(border=True):
        panel('Cluster profile: how each group differs from the average (z-score)')
        if nn:
            prof = out.groupby('cluster')[nn].mean()
            z = (prof - d[nn].mean()) / d[nn].std().replace(0, 1)
            fig = go.Figure(go.Heatmap(z=z.values, x=nn, y=[f'Cluster {i}' for i in z.index], colorscale=DIV, zmid=0,
                                       text=prof.round(2).values, texttemplate='%{text}' if len(nn) <= 10 else None))
            show(sty(fig, 330), 'lab_cprof')
        else:
            st.info('Add numeric columns to see a profile.')
    st.download_button('Download data with cluster labels', out.to_csv(index=False), 'clustered_data.csv', 'text/csv', type='primary')


def section_predict(df: pd.DataFrame, sig: str, name: str, kinds: dict, log: list[str]) -> None:
    res = st.session_state.get('lab_res')
    a, b = st.columns(2)
    with a, st.container(border=True):
        panel('Analysis report')
        lines = [f'# Dataset analysis: {name}', f'Generated {dt.datetime.now():%Y-%m-%d %H:%M}', '',
                 f'Rows: {len(df):,}  ·  Columns: {df.shape[1]} ({len(kinds["num"])} numeric, {len(kinds["cat"])} categorical)',
                 f'Missing cells: {df.isna().sum().sum():,}  ·  Duplicate rows: {int(df.duplicated().sum()):,}', '']
        if log:
            lines += ['## Automatic cleaning'] + [f'- {x}' for x in log] + ['']
        if res and res.get('sig') == sig and not res['board'].empty:
            bd = res['board']
            lines += [f"## Models: predicting `{res['target']}` ({res['task']})", f"Trained on {res['n_train']:,} rows; {res['baseline']}.", '',
                      bd.round(4).to_markdown(index=False) if _has_tabulate() else bd.round(4).to_string(index=False), '']
            bi = res['importance'].get(bd.iloc[0]['Model'])
            if bi is not None:
                lines += ['## Top inputs (best model)'] + [f'- {c}: {v:.4f}' for c, v in bi.head(8).items()] + ['']
        else:
            lines += ['_No models trained yet. Use the Train & compare tab._', '']
        lines += ['Model scores describe associations in this file, not causes.']
        report = '\n'.join(lines)
        st.markdown(f"<div class='rep'>{report}</div>", unsafe_allow_html=True)
        st.download_button('Download report (.md)', report, 'dataset_analysis.md', 'text/markdown', type='primary')
        st.download_button('Download cleaned data (.csv)', df.to_csv(index=False), 'cleaned_data.csv', 'text/csv')
    with b, st.container(border=True):
        panel('Score new rows with a trained model')
        if not res or res.get('sig') != sig or res['board'].empty:
            st.info('Train models first, then come back to use one on new data.')
            return
        m = st.selectbox('Model', list(res['board']['Model']), key='lab_pred_model')
        pipe = res['fitted'][m]
        buf = io.BytesIO()
        joblib.dump({'pipeline': pipe, 'classes': res['classes'], 'features': res['features'], 'target': res['target'], 'task': res['task']}, buf)
        st.download_button(f'Download {m} model (.joblib)', buf.getvalue(), f"{m.lower().replace(' ', '_')}.joblib", 'application/octet-stream')
        st.caption(f"Needs these columns: {', '.join(res['features'])}")
        f = st.file_uploader('New rows (CSV or Excel)', type=['csv', 'xlsx', 'xls'], key='lab_pred_file')
        use_same = st.checkbox('Or score the current dataset', key='lab_pred_same')
        new = None
        if f is not None:
            try:
                new = auto_clean(read_bytes(f.name, f.getvalue()))[0]
            except Exception as exc:
                st.error(f'Could not read the file: {exc}')
        elif use_same:
            new = df
        if new is not None:
            miss = [c for c in res['features'] if c not in new.columns]
            if miss:
                st.error(f'Missing columns: {miss}')
                return
            X = as_features(new[res['features']], res['cat'])
            pred = pipe.predict(X)
            outp = new.copy()
            if res['task'] == 'classification':
                outp[f'predicted_{res["target"]}'] = [res['classes'][i] for i in pred]
                if hasattr(pipe, 'predict_proba'):
                    outp['confidence'] = pipe.predict_proba(X).max(axis=1).round(4)
                vc = outp[f'predicted_{res["target"]}'].value_counts()
                fig = go.Figure(go.Bar(x=vc.index.astype(str), y=vc.values, marker_color=PALETTE[:len(vc)], text=vc.values, textposition='outside'))
                show(sty(fig, 260, yaxis_title='Rows'), 'lab_pred_dist')
            else:
                outp[f'predicted_{res["target"]}'] = np.round(pred, 4)
                fig = px.histogram(x=pred, nbins=40, color_discrete_sequence=[ACC2])
                show(sty(fig, 260, xaxis_title='Prediction'), 'lab_pred_dist')
            st.dataframe(outp.head(100), use_container_width=True, height=260)
            st.download_button('Download predictions (.csv)', outp.to_csv(index=False), 'predictions.csv', 'text/csv', type='primary')


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


# ================================================================== page
def render(root: str) -> None:
    with st.sidebar:
        st.markdown('## Data source')
        src = st.radio('Where is the data?', ['Upload my file', 'Built-in demo dataset'], key='lab_src')
        raw, name = None, ''
        if src == 'Upload my file':
            f = st.file_uploader('CSV, Excel or JSON', type=['csv', 'txt', 'xlsx', 'xls', 'json'], key='lab_file')
            if f is not None:
                try:
                    raw, name = read_bytes(f.name, f.getvalue()), f.name
                except Exception as exc:
                    st.error(f'Could not read this file: {exc}')
        else:
            name = st.selectbox('Dataset', ['Telco churn sample (classification)', 'Iris flowers (classification, 3 classes)',
                                            'Wine cultivars (classification, 3 classes)', 'Breast cancer (classification, 2 classes)',
                                            'Diabetes progression (regression)'], key='lab_demo')
            raw = demo_dataset(name, root)
        if raw is not None:
            st.markdown('## Cleaning')
            drop_dupes = st.checkbox('Remove duplicate rows', value=True, key='lab_dupes')
            drop_na = st.checkbox('Drop rows with any missing value', value=False, key='lab_dropna',
                                  help='Off: models fill gaps with the median or most common value instead.')

    chips = [('ALGORITHMS', str(len(zoo('classification'))), 'ok'), ('XGBOOST', 'ON' if xgb is not None else 'OFF', 'ok' if xgb is not None else 'bad')]
    if raw is None:
        topbar('Dataset Lab', 'Upload · explore · train · compare', chips, '🧪')
        st.markdown(
            "<div class='insight'><p><b>Upload any table</b> in the sidebar (CSV, Excel or JSON) or pick a built-in demo dataset. The lab will:</p><ul>"
            "<li><b>Clean</b> it automatically: trims text, turns numbers stored as text into numbers, parses dates, drops empty and constant columns.</li>"
            "<li><b>Profile</b> every column and <b>explore</b> distributions, missing values, correlations and relationships.</li>"
            "<li><b>Train and compare</b> up to 8 algorithms (Logistic/Linear Regression, Decision Tree, Random Forest, Gradient Boosting, "
            "KNN, Naive Bayes, SVM, XGBoost) with cross-validation and a held-out test set.</li>"
            "<li><b>Explain</b> the results: leaderboard, confusion matrix, ROC curves, residuals and feature importance.</li>"
            "<li><b>Cluster</b> rows with K-Means when there is no target, and <b>score new data</b> or download the model and a report.</li></ul></div>",
            unsafe_allow_html=True)
        return

    df, log = auto_clean(raw)
    if drop_dupes and df.duplicated().any():
        n = int(df.duplicated().sum())
        df = df.drop_duplicates().reset_index(drop=True)
        log = log + [f'Removed {n} duplicate row(s).']
    if drop_na:
        n0 = len(df)
        df = df.dropna().reset_index(drop=True)
        log = log + [f'Dropped {n0 - len(df)} row(s) with missing values.']
    if df.empty or df.shape[1] < 2:
        st.error('After cleaning there is not enough data left (need at least 2 columns and 1 row).')
        return
    kinds = col_kinds(df)
    sig = sig_of(df)
    topbar('Dataset Lab', name, chips + [('ROWS', f'{len(df):,}', ''), ('COLUMNS', str(df.shape[1]), ''),
                                         ('MISSING', f'{df.isna().mean().mean():.1%}', 'ok' if not df.isna().any().any() else 'bad')], '🧪')
    t = st.tabs(['📋 Overview', '🔍 Explore', '🤖 Train & compare', '🧩 Clustering', '📦 Predict & export'])
    with t[0]:
        section_overview(df, raw, log, kinds)
    with t[1]:
        section_explore(df, kinds)
    with t[2]:
        section_train(df, kinds, sig)
    with t[3]:
        section_cluster(df, kinds, sig)
    with t[4]:
        section_predict(df, sig, name, kinds, log)
