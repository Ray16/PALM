"""PALM dataset configuration + loaders.

``sources.py`` exposes a small registry mapping a dataset name to a loader that
returns a :class:`~PALM.data.sources.DatasetBundle` (featurized and ready to hand
to ``PALM.splitters``). ``data_source.csv`` (this directory) records the
provenance and download link for every configured database.
"""
