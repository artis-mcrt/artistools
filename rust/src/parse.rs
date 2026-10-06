use autocompress::io::ProcessorReader;
use autocompress::xz::XzDecompress;
use autocompress::{FileFormat, PlainProcessor, Processor};
use polars::prelude::*;
use std::io::{BufReader, Read};
use std::path::Path;
use std::str::FromStr;

/// Build an error describing a line that could not be understood
pub fn malformed(msg: String) -> PolarsError {
    PolarsError::ComputeError(msg.into())
}

/// Construct a decompressor for the given format with no xz decoder memory limit.
///
/// autocompress's own `autodetect_open` allows liblzma only 10 MB, which fails on files
/// compressed with the higher presets (`xz -9` needs 64 MiB to decode).
pub fn unlimited_decompressor(
    format: FileFormat,
) -> std::io::Result<Box<dyn Processor + Send + Unpin>> {
    Ok(match format {
        FileFormat::Xz => Box::new(XzDecompress::new(u64::MAX).map_err(std::io::Error::from)?),
        other => other.decompressor(),
    })
}

/// Return an I/O error with the place in a file before its text, e.g. `estimators_0000.out:12`
///
/// pyo3-polars gives Python the text of the `io::Error` and not the message of the `PolarsError`. Thus the place
/// must be in the `io::Error` itself.
pub fn io_error_at(err: &std::io::Error, place: &str) -> std::io::Error {
    std::io::Error::new(err.kind(), format!("{place}: {err}"))
}

/// Open a file with transparent decompression detected from its content (not the file name),
/// placing no memory limit on the xz decoder. An error names the file.
pub fn open_decompressed(filepath: &Path) -> std::io::Result<impl Read> {
    let open = || -> std::io::Result<_> {
        let mut reader = BufReader::new(std::fs::File::open(filepath)?);
        let decompressor: Box<dyn Processor + Send + Unpin> =
            match FileFormat::from_buf_reader(&mut reader)? {
                Some(format) => unlimited_decompressor(format)?,
                None => Box::new(PlainProcessor::new()),
            };

        Ok(ProcessorReader::with_processor(decompressor, reader))
    };

    open().map_err(|err| io_error_at(&err, &filepath.display().to_string()))
}

/// Add the file and the line number to the message of an error
pub fn error_at_line(err: &PolarsError, filepath: &Path, linenum: usize) -> PolarsError {
    err.wrap_msg(|msg| format!("{}:{linenum}: {msg}", filepath.display()))
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
