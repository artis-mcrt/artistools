# Intructions for LLMs
- Use type hints and python syntax must support 3.12 or above
- Always use polars DataFrames (`import polars as pl`) instead of pandas.
- Use `.group_by()` in polars (not `.groupby()` as in pandas).
- Polars doesn't have `.group_by().filter()`, use agg first, then do any filtering.
- When referencing string columns in expressions, wrap them with `pl.col("colname")` or use `pl.lit()` for literal strings.
- polars distinguishes between missing values: use `None` for missing, `float('nan')` for NaN.
- polars does not support multi-indexing; use columns for grouping or sorting instead.
- Use `.map_elements()` in polars instead of `.apply()` for element-wise operations on columns, and always specify the `return_dtype` parameter.
- Prefer lazy evaluation (`pl.LazyFrame`) for large data or complex pipelines.
- For joining DataFrames, use `.join()` with explicit `on` arguments.
- For type conversions, use `.cast()` (e.g., `df.with_columns(pl.col("a").cast(pl.Int64))`).
- Avoid inplace operations; polars methods return new DataFrames.
