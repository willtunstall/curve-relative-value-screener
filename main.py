import time
from typing import NamedTuple, Optional

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html, Input, Output
from scipy import stats  # installed with statsmodels, which mean_reversion.py already needs

import db
from butterfly_pca import Butterfly, Steepener
from mean_reversion import MeanRev

# Yields are written to the database each morning by fetch_yields.py. They're re-read at most
# once an hour, so new data is picked up without a restart and the database isn't hit on every click.
YIELDS_MAX_AGE = 60 * 60  # seconds
_yields = {'data': None, 'loaded_at': 0.0}


def get_yields():
    if time.time() - _yields['loaded_at'] > YIELDS_MAX_AGE:
        _yields['data'] = db.load_yields()
        _yields['loaded_at'] = time.time()
    return _yields['data']


# Keys match the column names written by fetch_yields.py
TENOR_LABELS = {
    '1MO': '1M', '3MO': '3M', '6MO': '6M', '1': '1Y', '2': '2Y', '3': '3Y',
    '5': '5Y', '7': '7Y', '10': '10Y', '20': '20Y', '30': '30Y',
}
TENOR_YEARS = {
    '1MO': 1 / 12, '3MO': 0.25, '6MO': 0.5, '1': 1, '2': 2, '3': 3,
    '5': 5, '7': 7, '10': 10, '20': 20, '30': 30,
}
ALL_TENORS = list(TENOR_LABELS)

# Fixed set of tenors the PCA is run on. Every preset leg must be in this list.
# Butterfly._par_dv01 casts tenor names to float, so only year tenors are valid here.
PCA_UNIVERSE = ['2', '3', '5', '7', '10', '30']

BUTTERFLY_PRESETS = {
    '2s5s10s': ('2', '5', '10'),
    '2s5s30s': ('2', '5', '30'),    # unconventional: wide, belly close to the short wing
    '2s7s30s': ('2', '7', '30'),    # unconventional
    '2s10s30s': ('2', '10', '30'),
    '3s10s30s': ('3', '10', '30'),
    '5s10s30s': ('5', '10', '30'),
    '5s7s10s': ('5', '7', '10'),
}

STEEPENER_PRESETS = {
    '2s5s': ('2', '5'),
    '2s10s': ('2', '10'),
    '2s30s': ('2', '30'),
    '3s10s': ('3', '10'),
    '5s30s': ('5', '30'),
    '10s30s': ('10', '30'),
}

# Everything that differs between trade types. The structure of a preset is inferred from its
# number of legs, so graph callbacks only need the preset name.
STRUCTURES = {
    'butterfly': {
        'label': 'Butterfly',
        'presets': BUTTERFLY_PRESETS,
        'default': '3s10s30s',
        'model': Butterfly,
        'construct': 'construct_butterfly',
        'leg_names': ['Short wing', 'Belly', 'Long wing'],
        'reference_leg': 'belly',
        'hedged_factors': (0, 1),  # PC1 and PC2 neutral
        'residual_model': '2-factor model',
        'axis': 'Butterfly (bp)',
    },
    'steepener': {
        'label': 'Steepener',
        'presets': STEEPENER_PRESETS,
        'default': '2s10s',
        'model': Steepener,
        'construct': 'construct_steepener',
        'leg_names': ['Short leg', 'Long leg'],
        'reference_leg': 'long leg',
        'hedged_factors': (0,),  # PC1 neutral only
        'residual_model': '1-factor model',
        'axis': 'Steepener (bp)',
    },
}
STRUCTURE_OPTIONS = [{'label': spec['label'], 'value': key} for key, spec in STRUCTURES.items()]
TRADE_PRESETS = {**BUTTERFLY_PRESETS, **STEEPENER_PRESETS}
assert len(TRADE_PRESETS) == len(BUTTERFLY_PRESETS) + len(STEEPENER_PRESETS), 'Preset names must be unique'


def structure_of(preset):
    return 'butterfly' if len(TRADE_PRESETS[preset]) == 3 else 'steepener'

# Values are trading-day windows; 0 means full history
LOOKBACK_OPTIONS = [
    {'label': '6M', 'value': 126},
    {'label': '1Y', 'value': 252},
    {'label': '2Y', 'value': 504},
    {'label': '5Y', 'value': 1260},
    {'label': 'Max', 'value': 0},
]
# Values are trading days before the latest observation
SNAPSHOT_OPTIONS = [
    {'label': 'Latest', 'value': 0},
    {'label': '1W ago', 'value': 5},
    {'label': '1M ago', 'value': 21},
    {'label': '3M ago', 'value': 63},
    {'label': '6M ago', 'value': 126},
    {'label': '1Y ago', 'value': 252},
]

# Mean reversion settings, matching us_treasury_pca.ipynb (not user-controlled)
FUTURE_N = 150           # projection / simulation horizon (notebook future_n; MeanRev.__init__ projects 150)
FPT_N_PATHS = 10_000     # hard-coded as sim_n inside MeanRev.vectorised_first_pass_sim (notebook uses 5,000); display only
FPT_BINS = 100           # notebook: ax.hist(..., bins=100)
HISTORY_START = 100      # notebook: ax.set_xlim(butterfly_price_dm.index[100], ...)

# (graph id, card title, card subtitle, x-axis title, y-axis title)
# Order must match the figures returned by update_pca_tab.
# Residuals and trade subtitles depend on the trade type and are updated by update_pca_tab.
PCA_GRAPHS = [
    ('pca-eigenvalues-graph', 'Scaled Eigenvalues', 'Share of variance explained', 'Component', 'Variance (%)'),
    ('pca-eigenvectors-graph', 'Eigenvectors', 'Loadings by tenor', 'Maturity', 'Loading'),
    ('pca-factors-graph', 'PCA Factors', 'PC1 level, PC2 slope, PC3 curvature', 'Date', 'Factor'),
    ('pca-residuals-graph', 'Factor Residuals', '2-factor model, latest observation', 'Maturity', 'Residual (bp)'),
    ('pca-trade-graph', 'PCA-Hedged Trade', 'DV01 and PC1/PC2 neutral', 'Date', 'Butterfly (bp)'),
]

# The hedge check has its own factor selector, so it is built and updated separately
FACTOR_OPTIONS = [
    {'label': 'PC1: Level', 'value': 0},
    {'label': 'PC2: Slope', 'value': 1},
    {'label': 'PC3: Curvature', 'value': 2},
]

HEDGE_TABLE_COLUMNS = ['Leg', 'Tenor', 'Hedge ratio', 'Par DV01']

OU_TABLE_COLUMNS = ['Parameter', 'Value']
PLACEHOLDER_VALUE = '—'

BRAND = '#1f3a5f'
MUTED = '#9aa5b1'
FACTOR_COLORS = ['#1f3a5f', '#e07a1f', '#2a9d8f']

# Model results
class TradeResult(NamedTuple):
    model: Butterfly          # Butterfly or Steepener (Steepener inherits from Butterfly)
    structure: str            # key into STRUCTURES
    legs: tuple
    series: pd.Series
    hedge_ratios: np.ndarray
    eigenvalues: np.ndarray


class MeanRevResult(NamedTuple):
    model: MeanRev
    hitting_times: np.ndarray
    error: Optional[str]


def lookback_slice(df, lookback):
    return df.iloc[-lookback:] if lookback else df


def get_trade(preset, lookback):
    """PCA fit and hedged trade for one (preset, lookback). Shared by the PCA and Mean Reversion tabs."""
    structure = structure_of(preset)
    spec = STRUCTURES[structure]
    legs = TRADE_PRESETS[preset]
    yields = lookback_slice(get_yields()[PCA_UNIVERSE], lookback)
    model = spec['model'](yields, legs)
    series, hedge_ratios = getattr(model, spec['construct'])()
    # The classes only keep the first three eigenvectors, so eigenvalues are recomputed here
    eigenvalues = np.linalg.eigvalsh(model.y_dm.cov())[::-1]
    return TradeResult(model, structure, legs, series, hedge_ratios, eigenvalues)


def get_mean_reversion(trade_result):
    """OU fit and Monte Carlo first passage times for a trade series."""
    model = MeanRev(trade_result.series)
    if not (np.isfinite(model.lamda) and model.lamda > 0):
        # AR(1) coefficient outside (0, 1): sigma is NaN and no simulated path would ever hit the target
        return MeanRevResult(model, np.array([]), 'Not mean reverting over this calibration window')
    hitting_times = model.vectorised_first_pass_sim(n=FUTURE_N)  # also sets model.target and model.direction
    return MeanRevResult(model, hitting_times, None)

def tenor_options(tenors):
    return [{'label': TENOR_LABELS[t], 'value': t} for t in tenors]


def option_label(value, options):
    return next((o['label'] for o in options if o['value'] == value), str(value))


def format_tenors(tenors, sep=', '):
    if not tenors:
        return 'None selected'
    return sep.join(TENOR_LABELS[t] for t in sorted(tenors, key=TENOR_YEARS.get))


def preset_options(presets):
    return [{'label': name, 'value': name} for name in presets]


def fmt(value, spec):
    return format(value, spec) if np.isfinite(value) else PLACEHOLDER_VALUE


def style_figure(fig, title='', x_title='', y_title=''):
    """Shared figure styling."""
    fig.update_layout(
        title={'text': title, 'x': 0.02, 'font': {'size': 13, 'color': '#6b7785'}},
        template='plotly_white',
        height=320,
        margin={'l': 55, 'r': 20, 't': 45, 'b': 45},
        xaxis={'title': x_title, 'showgrid': False, 'zeroline': False},
        yaxis={'title': y_title, 'gridcolor': 'rgba(0,0,0,0.08)', 'zeroline': False},
        legend={'orientation': 'h', 'x': 1, 'xanchor': 'right', 'y': 1.02, 'yanchor': 'bottom', 'font': {'size': 11}},
    )
    return fig


def placeholder_figure(title='', x_title='', y_title='', message='Loading...'):
    """Empty figure with a centred message."""
    fig = style_figure(go.Figure(), title, x_title, y_title)
    fig.add_annotation(
        text=message, xref='paper', yref='paper', x=0.5, y=0.5,
        showarrow=False, font={'size': 13, 'color': MUTED},
    )
    return fig

def control(label, component):
    return html.Div([html.Label(label, className='control-label'), component], className='control')


def button_radio(component_id, options, value):
    """Single-select toggle buttons."""
    return html.Div(
        dbc.RadioItems(
            id=component_id,
            options=options,
            value=value,
            class_name='btn-group btn-group-sm',
            input_class_name='btn-check',
            label_class_name='btn btn-outline-primary',
            label_checked_class_name='active',
        ),
        className='button-group',
    )


def button_checklist(component_id, options, value):
    """Multi-select toggle buttons."""
    return html.Div(
        dbc.Checklist(
            id=component_id,
            options=options,
            value=value,
            class_name='btn-group btn-group-sm',
            input_class_name='btn-check',
            label_class_name='btn btn-outline-primary',
            label_checked_class_name='active',
        ),
        className='button-group',
    )


def control_panel(*rows):
    """Tab-level controls that apply to every graph in the tab."""
    return dbc.Card(dbc.CardBody(list(rows)), class_name='control-panel')


def graph_card(title, graph_id, subtitle=None, x_title='', y_title='', controls=None):
    """
    Graph card. `controls` are rendered inside the card and apply to this graph only.
    The subtitle has id f'{graph_id}-subtitle' so callbacks can update it.
    """
    header = [html.H6(title, className='graph-card-title')]
    if subtitle:
        header.append(html.Small(subtitle, id=f'{graph_id}-subtitle', className='graph-card-subtitle'))
    body = [html.Div(header, className='graph-card-header')]
    if controls is not None:  # Dash components define __len__, so avoid a truthiness check
        body.append(html.Div(controls, className='graph-card-controls'))
    return dbc.Card(
        dbc.CardBody([
            *body,
            dcc.Loading(
                dcc.Graph(
                    id=graph_id,
                    figure=placeholder_figure(x_title=x_title, y_title=y_title),
                    config={'displaylogo': False, 'responsive': True},
                ),
                type='circle',
                color=BRAND,
            ),
        ]),
        class_name='graph-card h-100',
    )


def graph_grid(cards):
    """Lay out graph cards two per row."""
    return [
        dbc.Row([dbc.Col(card, lg=6) for card in cards[i:i + 2]], class_name='g-3 mb-3')
        for i in range(0, len(cards), 2)
    ]


def summary_table_card(title, table_body_id, columns, subtitle=None):
    """
    Card with a compact table. Rows are filled by a callback targeting `table_body_id`.
    The subtitle has id f'{table_body_id}-subtitle' so callbacks can update it.
    """
    header = [html.H6(title, className='graph-card-title')]
    if subtitle:
        header.append(html.Small(subtitle, id=f'{table_body_id}-subtitle', className='graph-card-subtitle'))
    return dbc.Card(
        dbc.CardBody([
            html.Div(header, className='graph-card-header mb-2'),
            dbc.Table(
                [html.Thead(html.Tr([html.Th(c) for c in columns])), html.Tbody(id=table_body_id)],
                hover=True,
                size='sm',
                class_name='summary-table mb-0',
            ),
        ]),
        class_name='graph-card mb-3',
    )


def table_rows(rows):
    return [html.Tr([html.Td(cell) for cell in row]) for row in rows]


def yield_analysis_tab():
    # Each graph owns its controls: snapshots always span the full curve,
    # tenor and lookback selection only drive the time series.
    cards = [
        graph_card(
            'Par Yield Curve', 'yc-snapshot-graph', 'All maturities', 'Maturity', 'Yield (%)',
            controls=control('Snapshot dates', button_checklist('yc-snapshots', SNAPSHOT_OPTIONS, [0, 21, 252])),
        ),
        graph_card(
            'Par Yield Time Series', 'yc-timeseries-graph', 'Selected tenors', 'Date', 'Yield (%)',
            controls=[
                control('Tenors', dcc.Dropdown(
                    id='yc-tenors',
                    options=tenor_options(ALL_TENORS),
                    value=['5', '10', '30'],
                    multi=True,
                    placeholder='Select tenors...',
                )),
                control('Lookback', button_radio('yc-lookback', LOOKBACK_OPTIONS, 252)),
            ],
        ),
    ]
    return html.Div(graph_grid(cards), className='tab-body')


def trade_controls(prefix, lookback_label, default_preset):
    """Trade type toggle, preset buttons for that type, and lookback. Shared by PCA and Mean Reversion."""
    structure = structure_of(default_preset)
    return control_panel(
        dbc.Row([
            dbc.Col(control('Trade type', button_radio(f'{prefix}-structure', STRUCTURE_OPTIONS, structure)),
                    xs='auto'),
            dbc.Col(control('Trade', button_radio(
                f'{prefix}-preset', preset_options(STRUCTURES[structure]['presets']), default_preset,
            )), xs='auto'),
            dbc.Col(control(lookback_label, button_radio(f'{prefix}-lookback', LOOKBACK_OPTIONS, 504)), xs='auto'),
        ], class_name='g-4'),
    )


def pca_analysis_tab():
    controls = trade_controls('pca', 'Lookback', '3s10s30s')
    hedge_table = summary_table_card(
        'Hedge Ratios', 'pca-hedge-table', HEDGE_TABLE_COLUMNS,
        subtitle='Notional per unit of belly, DV01 and PC1/PC2 neutral',
    )
    cards = [graph_card(title, gid, sub, x, y) for gid, title, sub, x, y in PCA_GRAPHS]
    cards.append(graph_card(
        'Hedge Check', 'pca-hedge-check-graph',
        'Trade vs factor, flat for the hedged factors when the hedge holds', 'PC1', 'Butterfly (bp)',
        controls=control('Factor', button_radio('pca-hedge-factor', FACTOR_OPTIONS, 0)),
    ))
    return html.Div([controls, hedge_table, *graph_grid(cards)], className='tab-body')


def mean_reversion_tab():
    controls = trade_controls('mr', 'Calibration window', '2s5s10s')
    ou_table = summary_table_card(
        'OU Parameters', 'mr-ou-table', OU_TABLE_COLUMNS,
        subtitle='dS = λ(μ − S)dt + σ dW, fitted by AR(1) regression S(i+1) = aS(i) + b + ε with δ = 1',
    )
    cards = [
        graph_card(
            'OU Process', 'mr-ou-graph', f'Expected future path (mean) with 2σ bands, {FUTURE_N} days ahead',
            'Date', 'Trade (bp)',
        ),
        graph_card(
            'First Passage Time Density', 'mr-fpt-graph',
            f'Monte Carlo hitting times to halfway between current level and μ ({FPT_N_PATHS:,} paths)',
            'Days after final observation', 'Nonparametric density',
        ),
    ]
    return html.Div([controls, ou_table, *graph_grid(cards)], className='tab-body')


def build_layout():
    return html.Div([
        html.Header([
            html.H1('Treasury Yield Curve Decomposition'),
            html.P('Par yields, principal component analysis and mean reversion of curve trades'),
        ], className='app-header'),
        dbc.Container([
            dbc.Tabs([
                dbc.Tab(yield_analysis_tab(), label='Yield Analysis', tab_id='tab-yields'),
                dbc.Tab(pca_analysis_tab(), label='PCA Analysis', tab_id='tab-pca'),
                dbc.Tab(mean_reversion_tab(), label='Mean Reversion', tab_id='tab-mean-reversion'),
            ], id='main-tabs', active_tab='tab-yields', class_name='main-tabs'),
        ], fluid=True, class_name='app-body'),
    ])


# ---------------------------------------------------------------------------
# Figures: Yield Analysis
# ---------------------------------------------------------------------------

def yield_curve_figure(snapshots):
    x_title, y_title = 'Maturity', 'Yield (%)'
    yields = get_yields()
    offsets = [s for s in sorted(snapshots or []) if s < len(yields)]
    if not offsets:
        return placeholder_figure('', x_title, y_title, 'Select at least one snapshot date')

    maturities = [TENOR_YEARS[t] for t in ALL_TENORS]
    labels = [TENOR_LABELS[t] for t in ALL_TENORS]
    fig = go.Figure()
    for offset in offsets:
        row = yields.iloc[-1 - offset]
        fig.add_trace(go.Scatter(
            x=maturities, y=row[ALL_TENORS].values, mode='lines+markers',
            name=f'{option_label(offset, SNAPSHOT_OPTIONS)} ({row.name:%d %b %Y})',
            customdata=labels, hovertemplate='%{customdata}: %{y:.2f}%<extra></extra>',
        ))
    style_figure(fig, '', x_title, y_title)
    # Log maturity axis keeps the 1M-1Y points from bunching up at the left
    fig.update_xaxes(type='log', tickvals=maturities, ticktext=labels)
    return fig


def yield_timeseries_figure(tenors, lookback):
    x_title, y_title = 'Date', 'Yield (%)'
    if not tenors:
        return placeholder_figure('', x_title, y_title, 'Select at least one tenor')

    data = lookback_slice(get_yields(), lookback)
    fig = go.Figure()
    for tenor in sorted(tenors, key=TENOR_YEARS.get):
        fig.add_trace(go.Scatter(x=data.index, y=data[tenor], mode='lines', name=TENOR_LABELS[tenor]))
    style_figure(fig, '', x_title, y_title)
    fig.update_layout(hovermode='x unified')
    return fig


def eigenvalues_figure(result):
    explained = result.eigenvalues / result.eigenvalues.sum() * 100
    components = [f'PC{i + 1}' for i in range(len(explained))]
    fig = go.Figure(go.Bar(
        x=components, y=explained, marker_color=[BRAND] * 3 + [MUTED] * (len(explained) - 3),
        text=[f'{v:.1f}%' for v in explained], textposition='outside',
    ))
    style_figure(fig, f'PC1–PC3 explain {explained[:3].sum():.1f}%', 'Component', 'Variance (%)')
    fig.update_yaxes(range=[0, 110])
    return fig


def eigenvectors_figure(result):
    eig_v = result.model.eig_v
    labels = [TENOR_LABELS[t] for t in eig_v.index]
    fig = go.Figure([
        go.Scatter(x=labels, y=eig_v.iloc[:, k], mode='lines+markers', name=f'PC{k + 1}',
                   line={'color': FACTOR_COLORS[k]})
        for k in range(3)
    ])
    return style_figure(fig, '', 'Maturity', 'Loading')


def factors_figure(result):
    fig = go.Figure([
        go.Scatter(x=factor.index, y=factor, mode='lines', name=f'PC{k + 1}', line={'color': FACTOR_COLORS[k]})
        for k, factor in enumerate(result.model.factors)
    ])
    style_figure(fig, '', 'Date', 'Factor')
    fig.update_layout(hovermode='x unified')
    return fig


def hedged_factors_text(structure):
    return '/'.join(f'PC{k + 1}' for k in STRUCTURES[structure]['hedged_factors'])


def residuals_figure(result):
    # Butterfly uses a 2-factor residual, Steepener overrides _factor_resid with a 1-factor residual
    residuals = result.model.residuals * 100  # yields are in %, show bp
    fig = go.Figure(go.Bar(
        x=[TENOR_LABELS[t] for t in residuals.index], y=residuals,
        marker_color=[BRAND if t in result.legs else MUTED for t in residuals.index],
        hovertemplate='%{x}: %{y:.2f} bp<extra></extra>',
    ))
    return style_figure(fig, 'Trade legs highlighted', 'Maturity', 'Residual (bp)')


def trade_figure(result):
    series = result.series
    fig = go.Figure(go.Scatter(x=series.index, y=series, mode='lines', line={'color': BRAND},
                               name=format_tenors(result.legs, sep='-')))
    fig.add_hline(y=0, line={'color': MUTED, 'width': 1})
    return style_figure(fig, f'Latest {series.iloc[-1]:.1f} bp', 'Date', STRUCTURES[result.structure]['axis'])


def hedge_check_figure(result, factor):
    """Notebook regression plots: trade on PC1, PC2 or PC3 with OLS slope and R²."""
    factor_series = result.model.factors[factor]
    series = result.series
    factor_label = option_label(factor, FACTOR_OPTIONS)
    fit = stats.linregress(factor_series, series)  # same as the notebook's regression cells
    x_line = np.array([factor_series.min(), factor_series.max()])
    hedged = factor in STRUCTURES[result.structure]['hedged_factors']

    fig = go.Figure([
        go.Scatter(x=factor_series, y=series, mode='markers', name='Daily obs',
                   marker={'size': 4, 'color': FACTOR_COLORS[factor], 'opacity': 0.5}),
        go.Scatter(x=x_line, y=fit.intercept + fit.slope * x_line, mode='lines', name='OLS fit',
                   line={'color': '#c0392b', 'dash': 'dash'}),
    ])
    title = (f'β = {fit.slope:.3f}, R² = {fit.rvalue ** 2:.3f}  |  '
             f'{"hedged, should be flat" if hedged else "not hedged"}')
    return style_figure(fig, title, factor_label, STRUCTURES[result.structure]['axis'])


# Notebook colours: tab:blue series and mean path, red upper band, yellow lower band
# (yellow darkened to gold so it stays visible on a white background)
NB_BLUE, NB_RED, NB_GOLD, NB_ORANGE = '#1f77b4', '#d62728', '#d4a017', '#ff7f0e'


def first_passage_stats(hitting_times):
    """Mean and mode of hitting times, as in the notebook (scipy.stats.mode)."""
    # squeeze: older SciPy returns the mode as a length-1 array, newer as a scalar
    return hitting_times.mean(), float(np.squeeze(stats.mode(hitting_times)[0]))


def ou_figure(trade_result, mr_result):
    """Notebook plot: trade history, expected future path (mean) and upper/lower sigma bands."""
    series = trade_result.series
    model = mr_result.model
    name = format_tenors(trade_result.legs, sep='-')
    y_title = STRUCTURES[trade_result.structure]['axis']
    fig = go.Figure(go.Scatter(x=series.index, y=series, mode='lines', name=name, line={'color': NB_BLUE}))

    if mr_result.error:
        return style_figure(fig, mr_result.error, 'Date', y_title)

    paths = model.future_paths_df
    projected = paths.iloc[1:-1]  # notebook plots iloc[1:-1]
    for column, label, color in [
        ('future_path', 'Expected Future Path (Mean)', NB_BLUE),
        ('upper_2', 'Upper Sigma Band', NB_RED),
        ('lower_2', 'Lower Sigma Band', NB_GOLD),
    ]:
        fig.add_trace(go.Scatter(x=projected.index, y=projected[column], mode='lines', name=label,
                                 line={'color': color, 'dash': 'dash'}))

    style_figure(fig, name, 'Date', y_title)
    start = series.index[min(HISTORY_START, len(series) - 1)]
    fig.update_xaxes(range=[start, paths.index[-1]])
    return fig


def first_passage_kde(hitting_times, grid):
    """
    Gaussian KDE of hitting times, reflected at 0 and FUTURE_N.

    Hitting times only exist in [0, FUTURE_N), so a plain Gaussian KDE would leak density below
    day 0 and past the horizon. Reflecting the kernel at both boundaries keeps all mass inside.
    Returns None if a KDE can't be fitted (fewer than 2 points, or all hitting times identical).
    """
    try:
        kde = stats.gaussian_kde(hitting_times)  # bandwidth by Scott's rule
    except (ValueError, np.linalg.LinAlgError):
        return None
    return kde(grid) + kde(-grid) + kde(2 * FUTURE_N - grid)


def fpt_figure(mr_result):
    """Density histogram of hitting times (100 bins) with a KDE and mean/mode lines."""
    x_title, y_title = 'Days after final observation', 'Nonparametric density'
    if mr_result.error:
        return placeholder_figure('', x_title, y_title, mr_result.error)

    hitting_times = mr_result.hitting_times
    if len(hitting_times) == 0:
        return placeholder_figure('', x_title, y_title, f'No simulated paths hit the target within {FUTURE_N} days')

    # Same binning as matplotlib's ax.hist(hitting_times, density=True, bins=100)
    density, edges = np.histogram(hitting_times, bins=FPT_BINS, density=True)
    mean_fp, mode_fp = first_passage_stats(hitting_times)
    grid = np.linspace(0, FUTURE_N, 300)
    kde_density = first_passage_kde(hitting_times, grid)
    y_max = max(density.max(), kde_density.max() if kde_density is not None else 0) * 1.05

    traces = [
        go.Bar(x=(edges[:-1] + edges[1:]) / 2, y=density, width=np.diff(edges),
               marker_color=NB_BLUE, opacity=0.35, name='Hitting Times'),
    ]
    if kde_density is not None:
        traces.append(go.Scatter(x=grid, y=kde_density, mode='lines', name='Kernel density',
                                 line={'color': NB_BLUE, 'width': 2.5}))

    fig = go.Figure([
        *traces,
        go.Scatter(x=[mean_fp, mean_fp], y=[0, y_max], mode='lines', name=f'Mean: {np.round(mean_fp)}',
                   line={'color': NB_ORANGE, 'dash': 'dash'}),
        go.Scatter(x=[mode_fp, mode_fp], y=[0, y_max], mode='lines', name=f'Mode: {mode_fp}',
                   line={'color': NB_RED, 'dash': 'dash'}),
    ])
    style_figure(fig, f'Target {mr_result.model.target:.1f} bp', x_title, y_title)
    fig.update_layout(bargap=0)
    fig.update_xaxes(range=[0, FUTURE_N])
    return fig


def ou_table_rows(mr_result):
    model = mr_result.model
    if mr_result.error or len(mr_result.hitting_times) == 0:
        return table_rows([
            ('Mean reversion speed λ (per day)', fmt(model.lamda, '.4f')),
            ('Long-run mean μ (bp)', fmt(model.mu, '.2f')),
            ('Half-life ln(2)/λ (days)', PLACEHOLDER_VALUE),
            ('Mean first passage (days)', PLACEHOLDER_VALUE),
            ('Mode first passage (days)', PLACEHOLDER_VALUE),
            ('P(first passage ≤ half-life)', PLACEHOLDER_VALUE),
        ])

    hitting_times = mr_result.hitting_times
    mean_fp, mode_fp = first_passage_stats(hitting_times)
    # (hitting_times <= half_life).sum() / len(hitting_times), i.e. among paths that hit
    p_half_life = (hitting_times <= model.half_life).sum() / len(hitting_times)
    return table_rows([
        ('Mean reversion speed λ (per day)', fmt(model.lamda, '.4f')),
        ('Long-run mean μ (bp)', fmt(model.mu, '.2f')),
        ('Half-life ln(2)/λ (days)', fmt(model.half_life, '.1f')),
        ('Mean first passage (days)', fmt(mean_fp, '.0f')),
        ('Mode first passage (days)', fmt(mode_fp, '.0f')),
        ('P(first passage ≤ half-life)', f'{p_half_life:.1%}'),
    ])

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], title='Yield Curve Dashboard')
app.layout = build_layout()
server = app.server

@app.callback(
    Output('yc-snapshot-graph', 'figure'),
    Output('yc-timeseries-graph', 'figure'),
    Input('yc-snapshots', 'value'),
    Input('yc-tenors', 'value'),
    Input('yc-lookback', 'value'),
)
def update_yield_tab(snapshots, tenors, lookback):
    return yield_curve_figure(snapshots), yield_timeseries_figure(tenors, lookback)


def register_structure_switch(prefix):
    """Swap the trade buttons when the trade type changes (PCA and Mean Reversion tabs)."""
    @app.callback(
        Output(f'{prefix}-preset', 'options'),
        Output(f'{prefix}-preset', 'value'),
        Input(f'{prefix}-structure', 'value'),
        prevent_initial_call=True,
    )
    def switch_structure(structure):
        spec = STRUCTURES[structure]
        return preset_options(spec['presets']), spec['default']


register_structure_switch('pca')
register_structure_switch('mr')

# Graph callbacks only depend on the preset: its structure is inferred from the number of legs,
# so there's no mismatch while the preset buttons are being swapped.


@app.callback(
    *[Output(gid, 'figure') for gid, *_ in PCA_GRAPHS],
    Output('pca-hedge-table', 'children'),
    Output('pca-hedge-table-subtitle', 'children'),
    Output('pca-residuals-graph-subtitle', 'children'),
    Output('pca-trade-graph-subtitle', 'children'),
    Input('pca-preset', 'value'),
    Input('pca-lookback', 'value'),
)
def update_pca_tab(preset, lookback):
    result = get_trade(preset, lookback)
    spec = STRUCTURES[result.structure]
    hedged = hedged_factors_text(result.structure)

    figures = [
        eigenvalues_figure(result),
        eigenvectors_figure(result),
        factors_figure(result),
        residuals_figure(result),
        trade_figure(result),
    ]
    hedge_rows = [
        (name, TENOR_LABELS[tenor], f'{ratio:.3f}', f'{result.model.dv01[tenor].iloc[-1]:.4f}')
        for name, tenor, ratio in zip(spec['leg_names'], result.legs, result.hedge_ratios)
    ]
    return (
        *figures,
        table_rows(hedge_rows),
        f'Notional per unit of {spec["reference_leg"]}, DV01 and {hedged} neutral',
        f'{spec["residual_model"]}, latest observation',
        f'{spec["label"]}, DV01 and {hedged} neutral',
    )


@app.callback(
    Output('pca-hedge-check-graph', 'figure'),
    Input('pca-preset', 'value'),
    Input('pca-lookback', 'value'),
    Input('pca-hedge-factor', 'value'),
)
def update_hedge_check(preset, lookback, factor):
    return hedge_check_figure(get_trade(preset, lookback), factor)


@app.callback(
    Output('mr-ou-graph', 'figure'),
    Output('mr-fpt-graph', 'figure'),
    Output('mr-ou-table', 'children'),
    Input('mr-preset', 'value'),
    Input('mr-lookback', 'value'),
)
def update_mean_reversion_tab(preset, lookback):
    trade_result = get_trade(preset, lookback)
    mr_result = get_mean_reversion(trade_result)  # vectorised Monte Carlo, recomputed on each selection
    return ou_figure(trade_result, mr_result), fpt_figure(mr_result), ou_table_rows(mr_result)


if __name__ == '__main__':
    app.run(debug=True)
