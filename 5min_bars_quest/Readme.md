# Market Structure Feature Research

This repository contains research code for building and analyzing feature families derived from market structure, path geometry, build behavior, and release transitions on time series price data.

## Scope

The project focuses on creating reusable research features from bar data while keeping the source price dataframe unchanged. Feature families are appended to a separate research dataframe so the workflow stays modular and extensible.

## Current Feature Groups

### Family A
Side relative support and damage geometry, normalized by local scale.

### Family B
Ordered path and cumulative path shape features over trailing windows.

### Family B Addon
Additional ordered path features such as half window balance, quartile balance, bend behavior, and late path dominance.

### Family C
Pre release build features that describe persistence, clustering, balance progression, strengthening, and efficiency before expansion.

### Family D
Release transition features that describe the shift from organized build into directional movement.

## Design Principles

- Keep source market data unchanged
- Build features into a separate research dataframe
- Use only frozen upstream inputs for downstream families
- Preserve causal, window based construction
- Append new feature blocks in one pass for consistency

## Repository Purpose

This repository is intended for exploratory research, feature engineering, and model preparation around market structure driven behavior in financial time series.

## Status

Work in progress. Additional feature families, validation steps, and research notebooks may be added over time.