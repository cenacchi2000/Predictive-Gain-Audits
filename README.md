# Reproducibility package: predictive-gain information audits

The complete code and saved experimental records are in [complete/](complete/).

- [Run instructions and scope](complete/README.txt)
- [Reproduction entry point](complete/reproduce.py)
- [Pinned requirements](complete/requirements.txt)
- [Foundation code and records](complete/foundation/)
- [Time-series application code and records](complete/application/)
- [All three rebuttal experiment scripts](complete/rebuttal/code/)
- [Complete rebuttal results and protocols](complete/rebuttal/results/)
- [Verification record](complete/rebuttal/results/verification.json)

To run the three additional studies in a new output folder:

```sh
cd complete
python -m pip install -r requirements.txt
python reproduce.py --output fresh_replay
```

The package includes all available experiment code and saved records, including negative results. Raw time-series archives and fitted classifier objects are not included; manifests identify external source data. The Bayesian witness is model-relative, and the categorical comparisons use a controlled oracle candidate. No new validation is implied by publishing this package.

SHA256SUMS.json records the files in complete/ for integrity checks.
