use polars::prelude::*;
use std::str::FromStr;

/// Build an error describing a line that could not be understood
pub fn malformed(msg: String) -> PolarsError {
    PolarsError::ComputeError(msg.into())
}

/// Parse a token taken from a line, naming what was expected if it doesn't parse
pub fn parse_field<T: FromStr>(token: &str, expected: &str) -> PolarsResult<T> {
    token
        .parse()
        .map_err(|_| malformed(format!("could not parse {token:?} as {expected}")))
}

/// Take the next token of a line and parse it, failing if the line ends first
pub fn next_field<'a, T: FromStr>(
    tokens: &mut impl Iterator<Item = &'a str>,
    expected: &str,
) -> PolarsResult<T> {
    let token = tokens
        .next()
        .ok_or_else(|| malformed(format!("line ended where {expected} was expected")))?;

    parse_field(token, expected)
}
