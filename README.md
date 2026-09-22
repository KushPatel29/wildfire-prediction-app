# live-data

`forecast.json` is the seven-day forecast in a form a web page can read: every
cell's calibrated risk for each day (integers in units of 1/`scale`), its normal
rate for the month, the day's satellite hotspots, and when and from what the
forecast was built.

`.github/workflows/forecast.yml` on `main` rewrites it twice a day, after the
`live-forecast` release. It is on a branch of its own so those commits stay out
of `main`'s history, and raw.githubusercontent.com serves it to any origin,
which GitHub release files are not. Built by `pipelines/web_snapshot.py`.
