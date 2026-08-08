use crate::parse::open_decompressed;
use pyo3::prelude::*;
use std::io::Read as _;
use std::path::PathBuf;

/// Truncate a line at the first occurrence of the comment prefix
fn strip_comment<'a>(line: &'a [u8], comment_prefix: Option<&str>) -> &'a [u8] {
    match comment_prefix {
        Some(prefix) if !prefix.is_empty() => {
            let prefix = prefix.as_bytes();
            line.windows(prefix.len())
                .position(|window| window == prefix)
                .map_or(line, |pos| &line[..pos])
        }
        _ => line,
    }
}

/// Collapse every run of whitespace to a single space, dropping blank lines.
///
/// The first `skip_rows` lines are dropped, and text from `comment_prefix` to the end of each
/// line is removed. The surviving lines are joined by single newlines with no trailing newline,
/// so the result can stream straight into a CSV parser with a single-space separator.
fn normalize_whitespace(data: &[u8], skip_rows: usize, comment_prefix: Option<&str>) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    for line in data.split(|&byte| byte == b'\n').skip(skip_rows) {
        let line = strip_comment(line, comment_prefix);

        let mut pos = 0;
        let mut line_started = false;
        while pos < line.len() {
            while pos < line.len() && line[pos].is_ascii_whitespace() {
                pos += 1;
            }
            let tokenstart = pos;
            while pos < line.len() && !line[pos].is_ascii_whitespace() {
                pos += 1;
            }
            if tokenstart < pos {
                if line_started {
                    out.push(b' ');
                } else {
                    if !out.is_empty() {
                        out.push(b'\n');
                    }
                    line_started = true;
                }
                out.extend_from_slice(&line[tokenstart..pos]);
            }
        }
    }

    out
}

/// Read a whitespace-separated text file into bytes with each run of whitespace collapsed to a single space.
///
/// The file may be zstd/gzip/xz compressed (detected from the content, not the file name). The
/// first `skip_rows` lines are dropped, text from `comment_prefix` to the end of a line is
/// removed, and blank lines are omitted, matching what `pandas.read_csv(sep=r"\s+")` would parse.
#[pyfunction]
#[pyo3(signature = (filepath, skip_rows=0, comment_prefix=None))]
#[expect(clippy::needless_pass_by_value)]
pub fn read_wsv_normalized(
    filepath: PathBuf,
    skip_rows: usize,
    comment_prefix: Option<&str>,
) -> PyResult<Vec<u8>> {
    let mut data = Vec::new();
    open_decompressed(&filepath)?.read_to_end(&mut data)?;

    Ok(normalize_whitespace(&data, skip_rows, comment_prefix))
}

#[cfg(test)]
mod tests {
    use super::normalize_whitespace;

    #[test]
    fn collapses_whitespace_and_drops_blank_lines() {
        let text = b"  colA \t colB\r\n1    2.5\n\n 4\t5 \n";
        assert_eq!(
            normalize_whitespace(text, 0, None),
            b"colA colB\n1 2.5\n4 5"
        );
    }

    #[test]
    fn skips_rows_and_strips_comments() {
        let text = b"# file comment\nx y\n1 2 # trailing comment\n# only comment\n3   4\n";
        assert_eq!(normalize_whitespace(text, 1, Some("#")), b"x y\n1 2\n3 4");
        // skip_rows applies before comment handling, so the comment line still counts as a row
        assert_eq!(normalize_whitespace(text, 2, Some("#")), b"1 2\n3 4");
    }

    #[test]
    fn empty_input_gives_empty_output() {
        assert_eq!(normalize_whitespace(b"", 0, Some("#")), b"");
        assert_eq!(normalize_whitespace(b"# all comments\n", 0, Some("#")), b"");
    }
}
