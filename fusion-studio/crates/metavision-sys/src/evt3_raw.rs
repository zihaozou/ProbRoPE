use std::io;
use std::path::Path;


#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Evt3Header {
    pub width: u32,
    pub height: u32,

    pub evt_version: String,
    pub raw_lines: Vec<String>,
}


pub fn parse_header(data: &[u8]) -> io::Result<(Evt3Header, usize)> {
    let mut offset = 0usize;
    let mut raw_lines = Vec::new();
    while offset < data.len() && data[offset] == b'%' {
        let nl = data[offset..].iter().position(|&b| b == b'\n');
        let line_end = match nl {
            Some(p) => offset + p,
            None => data.len(),
        };
        let line = String::from_utf8_lossy(&data[offset..line_end]);
        let content = line.trim_start_matches('%').trim().to_string();
        raw_lines.push(content);
        offset = match nl {
            Some(_) => line_end + 1,
            None => line_end,
        };
    }

    let mut width = 0u32;
    let mut height = 0u32;
    let mut evt_version = String::new();
    for l in &raw_lines {
        if let Some(rest) = l.strip_prefix("format ") {
            for kv in rest.split(';') {
                if let Some(v) = kv.strip_prefix("width=") {
                    width = v.trim().parse().unwrap_or(width);
                } else if let Some(v) = kv.strip_prefix("height=") {
                    height = v.trim().parse().unwrap_or(height);
                }
            }
        } else if let Some(rest) = l.strip_prefix("geometry ") {
            if let Some((w, h)) = rest.trim().split_once('x') {
                if width == 0 {
                    width = w.parse().unwrap_or(0);
                }
                if height == 0 {
                    height = h.parse().unwrap_or(0);
                }
            }
        } else if let Some(rest) = l.strip_prefix("evt ") {
            evt_version = rest.trim().to_string();
        }
    }

    if width == 0 || height == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "evt3 header missing width/height (no 'format' or 'geometry' line found)",
        ));
    }

    Ok((
        Evt3Header { width, height, evt_version, raw_lines },
        offset,
    ))
}


pub struct Evt3RawFile {
    pub header: Evt3Header,
    pub words: Vec<u16>,
}

impl Evt3RawFile {
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let data = std::fs::read(path)?;
        let (header, offset) = parse_header(&data)?;
        let payload = &data[offset..];
        let mut words = Vec::with_capacity(payload.len() / 2);
        let mut chunks = payload.chunks_exact(2);
        for pair in &mut chunks {
            words.push(u16::from_le_bytes([pair[0], pair[1]]));
        }
        if !chunks.remainder().is_empty() {
            log_truncated_tail();
        }
        Ok(Self { header, words })
    }


    pub fn chunks(&self, chunk_words: usize) -> std::slice::Chunks<'_, u16> {
        self.words.chunks(chunk_words.max(1))
    }
}

#[cold]
fn log_truncated_tail() {
    eprintln!("evt3_raw: warning: payload length is not a multiple of 2 bytes; ignoring trailing byte");
}

pub fn bytes_to_words(words: &mut Vec<u16>, carry: &mut Option<u8>, bytes: &[u8]) {
    let mut bytes = bytes;
    if let Some(lo) = carry.take() {
        if let Some((&hi, rest)) = bytes.split_first() {
            words.push(u16::from_le_bytes([lo, hi]));
            bytes = rest;
        } else {
            *carry = Some(lo);
            return;
        }
    }
    words.reserve(bytes.len() / 2);
    let mut pairs = bytes.chunks_exact(2);
    for pair in &mut pairs {
        words.push(u16::from_le_bytes([pair[0], pair[1]]));
    }
    if let &[last] = pairs.remainder() {
        *carry = Some(last);
    }
}


pub fn time_shift_us(words: &[u16]) -> Option<i64> {
    words.iter().find(|&&w| w >> 12 == 0x8).map(|&w| {
        let mut t = (w & 0xFFF) as i64;
        if t > 0 {
            t -= 1;
        }
        t << 12
    })
}

#[cfg(test)]
mod tests {
    use super::*;


    const DANCER_HEADER: &[u8] = b"% camera_integrator_name Prophesee\n\
% date 2026-08-17 14:55:44\n\
% evt 3.0\n\
% format EVT3;height=720;width=1280\n\
% generation 4.2\n\
% geometry 1280x720\n\
% plugin_integrator_name Prophesee\n\
% plugin_name hal_plugin_prophesee\n\
% sensor_generation 4.2\n\
% sensor_name IMX636\n\
% serial_number 00051618\n\
% end\n";

    #[test]
    fn parses_dancer_header_fields() {
        let (header, offset) = parse_header(DANCER_HEADER).expect("header should parse");
        assert_eq!(header.width, 1280);
        assert_eq!(header.height, 720);
        assert_eq!(header.evt_version, "3.0");
        assert_eq!(offset, DANCER_HEADER.len(), "offset should land exactly after the header");
        assert!(header.raw_lines.contains(&"sensor_name IMX636".to_string()));
        assert_eq!(header.raw_lines.last().unwrap(), "end");
    }

    #[test]
    fn offset_points_past_header_into_binary_payload() {
        let mut data = DANCER_HEADER.to_vec();
        data.extend_from_slice(&[0x56u8, 0x85u8]);
        let (_header, offset) = parse_header(&data).unwrap();
        assert_eq!(offset, DANCER_HEADER.len());
        assert_eq!(&data[offset..], &[0x56, 0x85]);
    }

    #[test]
    fn header_without_trailing_newline_on_last_line_still_parses() {
        let data = b"% evt 3.0\n% format EVT3;height=2;width=3\n% end".to_vec();
        let (header, offset) = parse_header(&data).unwrap();
        assert_eq!((header.width, header.height), (3, 2));
        assert_eq!(offset, data.len());
    }

    #[test]
    fn missing_width_height_is_an_error() {
        let data = b"% evt 3.0\n% end\n".to_vec();
        assert!(parse_header(&data).is_err());
    }

    #[test]
    fn geometry_line_used_as_fallback_when_format_omits_dims() {
        let data = b"% evt 3.0\n% geometry 640x480\n% end\n".to_vec();
        let (header, _) = parse_header(&data).unwrap();
        assert_eq!((header.width, header.height), (640, 480));
    }

    #[test]
    fn bytes_to_words_carries_odd_byte_across_calls() {
        let stream: Vec<u8> = vec![0x56, 0x85, 0x01, 0x62, 0x03, 0x00, 0x10, 0x22];
        let whole: Vec<u16> = stream.chunks_exact(2).map(|p| u16::from_le_bytes([p[0], p[1]])).collect();
        for split in 0..=stream.len() {
            let mut words = Vec::new();
            let mut carry = None;
            bytes_to_words(&mut words, &mut carry, &stream[..split]);
            bytes_to_words(&mut words, &mut carry, &stream[split..]);
            assert_eq!(words, whole, "split at byte {split}");
            assert!(carry.is_none(), "even-length stream must leave no carry (split {split})");
        }
        let mut words = Vec::new();
        let mut carry = None;
        bytes_to_words(&mut words, &mut carry, &stream[..3]);
        assert_eq!(words, whole[..1]);
        assert_eq!(carry, Some(0x01));
    }

    #[test]
    fn time_shift_matches_sdk_formula() {
        let words = [0xE014u16, 0x8556, 0x6001, 0x0002];
        assert_eq!(time_shift_us(&words), Some(5_591_040));
        assert_eq!(time_shift_us(&[0x8000u16, 0x6001]), Some(0));
        assert_eq!(time_shift_us(&[0x6001u16, 0x0002]), None);
    }
}
