use autocompress::io::ProcessorReader;
use autocompress::xz::XzDecompress;
use autocompress::{FileFormat, Processor};
use polars::prelude::*;
use std::io::{BufReader, Cursor, Read};
use std::num::NonZero;
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

/// A sequence of blocks of decompressed text, in the order of the file
pub type TextBlocks = Box<dyn Iterator<Item = std::io::Result<Vec<u8>>> + Send>;

/// The size of a block of text from a file that is not zstd
const STREAM_TEXTBLOCK_BYTES: usize = 16 * 1024 * 1024;

/// Open a file and return its decompressed text in blocks
///
/// The content of the file, and not its name, gives the compression format. The xz decoder has no memory limit. An
/// error names the file. A block from a zstd file holds whole frames, and ARTIS ends each frame of an estimator file at the end of a cell.
pub fn open_text_blocks(filepath: &Path) -> std::io::Result<TextBlocks> {
    let open = || -> std::io::Result<TextBlocks> {
        let mut reader = BufReader::new(std::fs::File::open(filepath)?);
        Ok(match FileFormat::from_buf_reader(&mut reader)? {
            Some(FileFormat::Zstd) => parallel_zstd_text_blocks(reader),
            Some(format) => Box::new(StreamTextBlocks {
                reader: ProcessorReader::with_processor(unlimited_decompressor(format)?, reader),
                finished: false,
            }),
            None => Box::new(StreamTextBlocks {
                reader,
                finished: false,
            }),
        })
    };

    open().map_err(|err| io_error_at(&err, &filepath.display().to_string()))
}

/// Open a file as `open_text_blocks` does, and return a reader of its decompressed text
pub fn open_decompressed(filepath: &Path) -> std::io::Result<TextBlocksReader> {
    Ok(TextBlocksReader {
        blocks: open_text_blocks(filepath)?,
        block: Vec::new(),
        blockpos: 0,
    })
}

/// The blocks of text of a reader
struct StreamTextBlocks<R: Read> {
    reader: R,
    finished: bool,
}

impl<R: Read> Iterator for StreamTextBlocks<R> {
    type Item = std::io::Result<Vec<u8>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        let mut block = Vec::with_capacity(STREAM_TEXTBLOCK_BYTES);
        match (&mut self.reader)
            .take(STREAM_TEXTBLOCK_BYTES as u64)
            .read_to_end(&mut block)
        {
            Ok(bytecount) => {
                self.finished = bytecount < STREAM_TEXTBLOCK_BYTES;
                (!block.is_empty()).then_some(Ok(block))
            }
            Err(err) => {
                self.finished = true;
                Some(Err(err))
            }
        }
    }
}

/// A reader of the text of blocks
pub struct TextBlocksReader {
    blocks: TextBlocks,
    /// the block that a read takes now, and the position of its next byte
    block: Vec<u8>,
    blockpos: usize,
}

impl Read for TextBlocksReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        while self.blockpos == self.block.len() {
            match self.blocks.next() {
                Some(block) => {
                    self.block = block?;
                    self.blockpos = 0;
                }
                None => return Ok(0),
            }
        }
        let count = buf.len().min(self.block.len() - self.blockpos);
        buf[..count].copy_from_slice(&self.block[self.blockpos..self.blockpos + count]);
        self.blockpos += count;
        Ok(count)
    }
}

/// The magic number at the start of a zstd frame
const ZSTD_MAGIC: u32 = 0xFD2F_B528;

/// The minimum compressed size of a group of zstd frames that one thread decompresses
const MIN_FRAMEGROUP_BYTES: usize = 4 * 1024 * 1024;

/// The maximum compressed size of a zstd frame that the parallel reader decompresses. The zstd command writes one large
/// frame, and the streaming decoder then reads the file, thus the memory of the reader stays small
const MAX_FRAME_BYTES: usize = 64 * 1024 * 1024;

/// The maximum number of threads that decompress the groups of zstd frames
const MAX_DECOMPRESS_THREADS: usize = 16;

/// The result of a read of one zstd frame
enum FrameRead {
    /// the buffer holds the complete frame
    Complete,
    /// the file ends before the frame starts
    End,
    /// the frame is cut, larger than `MAX_FRAME_BYTES`, or of a form that the reader does not know. The buffer holds
    /// the bytes of the frame that the reader read
    Unusual,
}

/// Append `count` bytes of `reader` to `buffer`, and return false if the file ends before them
///
/// The buffer keeps the bytes before the end of a cut file, thus the streaming decoder can read their complete zstd
/// blocks.
fn append_bytes(
    reader: &mut impl Read,
    buffer: &mut Vec<u8>,
    count: usize,
) -> std::io::Result<bool> {
    let end = buffer.len() + count;
    let mut filled = buffer.len();
    buffer.resize(end, 0);
    while filled < end {
        match reader.read(&mut buffer[filled..]) {
            Ok(0) => break,
            Ok(bytecount) => filled += bytecount,
            Err(err) if err.kind() == std::io::ErrorKind::Interrupted => {}
            Err(err) => return Err(err),
        }
    }
    buffer.truncate(filled);
    Ok(filled == end)
}

/// Return the little-endian integer of the last `count` bytes of `buffer`
fn last_bytes_as_integer(buffer: &[u8], count: usize) -> usize {
    buffer[buffer.len() - count..]
        .iter()
        .rev()
        .fold(0, |value, &byte| (value << 8) | usize::from(byte))
}

/// Append the next zstd frame of `reader` to `buffer`
///
/// The header of the frame and of each zstd block give the size of the frame, thus the reader finds the end of the
/// frame without a decompression. Request for Comments (RFC) 8878 gives the format.
fn read_zstd_frame(reader: &mut impl Read, buffer: &mut Vec<u8>) -> std::io::Result<FrameRead> {
    let framestart = buffer.len();
    if !append_bytes(reader, buffer, 4)? {
        return Ok(if buffer.len() == framestart {
            FrameRead::End
        } else {
            FrameRead::Unusual
        });
    }
    let magic = last_bytes_as_integer(buffer, 4);
    if magic & 0xFFFF_FFF0 == 0x184D_2A50 {
        // a skippable frame, which the decoder ignores
        if !append_bytes(reader, buffer, 4)? {
            return Ok(FrameRead::Unusual);
        }
        let size = last_bytes_as_integer(buffer, 4);
        if size > MAX_FRAME_BYTES || !append_bytes(reader, buffer, size)? {
            return Ok(FrameRead::Unusual);
        }
        return Ok(FrameRead::Complete);
    }
    if magic != ZSTD_MAGIC as usize || !append_bytes(reader, buffer, 1)? {
        return Ok(FrameRead::Unusual);
    }

    let descriptor = buffer[buffer.len() - 1];
    let singlesegment = descriptor & 0x20 != 0;
    let windowbytes = usize::from(!singlesegment);
    let dictidbytes = [0, 1, 2, 4][usize::from(descriptor & 3)];
    let contentsizebytes = [usize::from(singlesegment), 2, 4, 8][usize::from(descriptor >> 6)];
    if !append_bytes(reader, buffer, windowbytes + dictidbytes + contentsizebytes)? {
        return Ok(FrameRead::Unusual);
    }

    loop {
        if !append_bytes(reader, buffer, 3)? {
            return Ok(FrameRead::Unusual);
        }
        let blockheader = last_bytes_as_integer(buffer, 3);
        let blocksize = match (blockheader >> 1) & 3 {
            // a raw zstd block or a compressed zstd block
            0 | 2 => blockheader >> 3,
            // a run-length encoding (RLE) block holds one byte
            1 => 1,
            _ => return Ok(FrameRead::Unusual),
        };
        if buffer.len() - framestart + blocksize > MAX_FRAME_BYTES
            || !append_bytes(reader, buffer, blocksize)?
        {
            return Ok(FrameRead::Unusual);
        }
        if blockheader & 1 != 0 {
            break;
        }
    }

    let checksumbytes = if descriptor & 4 == 0 { 0 } else { 4 };
    if !append_bytes(reader, buffer, checksumbytes)? {
        return Ok(FrameRead::Unusual);
    }

    Ok(FrameRead::Complete)
}

/// Decompress a group of complete zstd frames
fn decompress_framegroup(framegroup: &[u8]) -> std::io::Result<Vec<u8>> {
    let mut text = Vec::new();
    ProcessorReader::with_processor(FileFormat::Zstd.decompressor(), Cursor::new(framegroup))
        .read_to_end(&mut text)?;
    Ok(text)
}

/// Return the text of a zstd file in blocks that other threads decompress while the caller parses them
///
/// ARTIS writes the estimator file of all ranks as many small frames, and one thread decompresses approximately 1 GB
/// each second. The parser is faster than that. One thread reads several groups of frames, and a different thread
/// decompresses each group. Each group gives one block. The channel holds the blocks of one round of groups, thus the memory stays
/// small while the caller parses the round before.
fn parallel_zstd_text_blocks(source: BufReader<std::fs::File>) -> TextBlocks {
    let threadcount = std::thread::available_parallelism()
        .map_or(1, NonZero::get)
        .min(MAX_DECOMPRESS_THREADS);
    let (sender, receiver) = std::sync::mpsc::sync_channel(threadcount);
    std::thread::spawn(move || send_zstd_text_blocks(source, threadcount, &sender));
    Box::new(receiver.into_iter())
}

/// Read the frames of a zstd file, and send the decompressed text of each group of frames in the order of the file
///
/// An unusual frame and the rest of the file go to the streaming decoder. The decoder gives the complete zstd blocks
/// of a cut frame, as for a file of one frame. The thread stops when the receiver stops.
fn send_zstd_text_blocks(
    mut source: BufReader<std::fs::File>,
    threadcount: usize,
    sender: &std::sync::mpsc::SyncSender<std::io::Result<Vec<u8>>>,
) {
    loop {
        let mut framegroups: Vec<Vec<u8>> = Vec::new();
        let mut unusualframe: Option<Vec<u8>> = None;
        let mut sourceended = false;
        while framegroups.len() < threadcount && unusualframe.is_none() && !sourceended {
            let mut framegroup = Vec::new();
            while framegroup.len() < MIN_FRAMEGROUP_BYTES {
                let framestart = framegroup.len();
                match read_zstd_frame(&mut source, &mut framegroup) {
                    Ok(FrameRead::Complete) => {}
                    Ok(FrameRead::End) => {
                        sourceended = true;
                        break;
                    }
                    Ok(FrameRead::Unusual) => {
                        unusualframe = Some(framegroup.split_off(framestart));
                        break;
                    }
                    Err(err) => {
                        let _ = sender.send(Err(err));
                        return;
                    }
                }
            }
            if !framegroup.is_empty() {
                framegroups.push(framegroup);
            }
        }

        let texts: Vec<std::io::Result<Vec<u8>>> = std::thread::scope(|scope| {
            let handles: Vec<_> = framegroups
                .iter()
                .map(|framegroup| scope.spawn(|| decompress_framegroup(framegroup)))
                .collect();
            handles
                .into_iter()
                .map(|handle| {
                    handle
                        .join()
                        .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
                })
                .collect()
        });
        for text in texts {
            let failed = text.is_err();
            if sender.send(text).is_err() || failed {
                return;
            }
        }

        if let Some(unusualframe) = unusualframe {
            let streaming = StreamTextBlocks {
                reader: ProcessorReader::with_processor(
                    FileFormat::Zstd.decompressor(),
                    Cursor::new(unusualframe).chain(source),
                ),
                finished: false,
            };
            for block in streaming {
                if sender.send(block).is_err() {
                    return;
                }
            }
            return;
        }
        if sourceended {
            return;
        }
    }
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
