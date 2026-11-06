# FlowCast

Short-horizon traffic forecasting for the Northline Corridor: 25 road segments, 30-minute
windows. FlowCast forecasts volume, congestion level, travel time and accident risk for the next
30 to 120 minutes, and shows them in a web dashboard for the corridor operations team.

## Run it

You need Python 3.11 or newer and the three raw files in `data/raw/`:

```
data/raw/traffic_sensor_log.csv
data/raw/weather_observations.csv
data/raw/calendar_events.csv
```

```bash
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python -m src.pipeline --data-dir data/raw                  # cleans, trains, forecasts (about 30 min)
streamlit run dashboard/app.py                              # http://localhost:8501
```

Run the pipeline again whenever new raw files arrive. The dashboard picks up the new results by
itself. You can also start a run from **Data > Data quality**; it runs in the background.

## Run it with Docker

```bash
docker build -t flowcast .
docker run --rm -v "$PWD/data:/app/data" -v "$PWD/models:/app/models" -v "$PWD/reports:/app/reports" \
    flowcast python -m src.pipeline --data-dir data/raw
docker run -d -p 8501:8501 -v "$PWD/data:/app/data" -v "$PWD/models:/app/models" \
    -v "$PWD/reports:/app/reports" --name flowcast flowcast
```

The container runs as a non-root user and answers health checks at `/_stcore/health`. To put it on
the internet, run it behind a reverse proxy that handles HTTPS and sign-in (for example nginx with
your organisation's single sign-on). Streamlit has no user accounts of its own.

## Pages

Five pages along the top, each with its own address so a team can bookmark the one it starts its
shift on. The PRD's nine dashboard views are all here, grouped by who uses them.

| Address | Page | Who it is for | What is on it |
| --- | --- | --- | --- |
| `/` | Now | Control room | Status band, map of every road (click a dot to select it), three headline numbers against a usual week, the roads that need attention, and every road in corridor order. Switch to Look back to replay any past window. |
| `/today` | Today | Control room, shift leads | The day on the map with Play and a time slider, the corridor total against a usual day, slow roads through the day, and every road by half-hour. |
| `/roads` | Roads | Traffic engineers | Tabs: one road in detail (next two hours with likely ranges, any past forecast), all roads compared, trends, weather, and busy times over a date range. |
| `/accuracy` | Accuracy | Reviewers, managers | How accurate the forecasts are in plain numbers. Tabs: targets, every model compared (confusion matrix, error by horizon, training curves), forecast against recorded, likely ranges, and the inputs the models rely on. |
| `/data` | Data | Data team | Tabs: upload new files and forecast without retraining, download a report for any period, and data checks with retraining. |

"Usual" means the median recorded value for the same road, weekday and half-hour.

## Map

The map shows street tiles from CARTO when the server can reach the internet. Without internet it
draws Delhi ward outlines bundled in `dashboard/static/delhi_wards.geojson` (DataMeet,
CC BY-SA 2.5 IN), so the page still works on a closed network. Force one or the other with:

```bash
FLOWCAST_BASEMAP=streets streamlit run dashboard/app.py    # or FLOWCAST_BASEMAP=offline
```

## Project layout

```
config.yaml          rules, split dates and hyperparameters
src/                 pipeline: ingest, clean, features, eda, ml_models, dl_model, evaluate, predict
dashboard/app.py     pages
dashboard/ui.py      colours, type, components and chart style (the only place styles live)
dashboard/store.py   read-only access to the pipeline outputs
dashboard/static/    IBM Plex Sans (SIL Open Font License), the logo and ward outlines
.streamlit/          theme (light and dark) and server settings
notebooks/           01_eda, 02_cleaning, 03_modelling: step-by-step walkthroughs of the pipeline
tests/               pytest suite, including leakage tests
```

## Interface notes

- Grey is the default colour. Colour appears only where something needs attention: amber,
  orange and red for Moderate, Heavy and Severe, and a red dot where a patrol is suggested.
- IBM Plex Sans throughout. Its digits all have the same width, so numbers line up in columns.
  Sizes: 12 px labels, 13 px tables, 14 px text, 22 px page titles.
- Light and dark themes follow the device setting. You can switch under Settings in the
  top-right menu.
- Forms use one column, with labels above the fields and help next to each label.
- The layout reflows on phones: columns stack and each road becomes a two-line block.

## Notebooks

Run the pipeline once, then open the walkthroughs (EDA, cleaning, modelling). Each cell calls the same
functions in `src/`, and the cell above it says what output to expect.

```bash
pip install -r requirements-dev.txt
jupyter lab notebooks/
```

## What to commit

`.gitignore` leaves out everything the pipeline rebuilds (processed data, trained model files) and the
two large raw CSVs. Model cards (`models/*.md`, `models/*.json`), reports and figures are small and are
committed, so a reviewer can read the results before rerunning anything.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```
