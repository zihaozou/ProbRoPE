use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

const HEADER_LEN: u64 = 24;

const T_BITS: u32 = 39;
const COORD_MAX: u16 = 0xFFF;

pub struct EventsBinWriter {
    out: BufWriter<File>,
    size: (u32, u32),

    t0_us: Option<i64>,
    late_dropped: u64,
    written: u64,
    bytes_written: u64,
}

impl EventsBinWriter {
    pub fn create(path: &Path, size: (u32, u32)) -> io::Result<Self> {
        let out = BufWriter::with_capacity(4 << 20, File::create(path)?);
        Ok(Self { out, size, t0_us: None, late_dropped: 0, written: 0, bytes_written: 0 })
    }

    fn write_header(&mut self, t0_us: i64) -> io::Result<()> {
        let mut h = [0u8; HEADER_LEN as usize];
        h[0..4].copy_from_slice(b"FSEV");
        h[4..8].copy_from_slice(&1u32.to_le_bytes());
        h[8..12].copy_from_slice(&self.size.0.to_le_bytes());
        h[12..16].copy_from_slice(&self.size.1.to_le_bytes());
        h[16..24].copy_from_slice(&t0_us.to_le_bytes());
        self.out.write_all(&h)?;
        self.bytes_written += HEADER_LEN;
        Ok(())
    }

    pub fn push(&mut self, t_us: i64, x: u16, y: u16, p: u8) -> io::Result<()> {
        if x > COORD_MAX || y > COORD_MAX {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("事件坐标超出 12 位: x={x} y={y}(上限 4095)"),
            ));
        }
        let t0 = match self.t0_us {
            Some(t0) => t0,
            None => {
                self.write_header(t_us)?;
                self.t0_us = Some(t_us);
                t_us
            }
        };
        let dt = t_us - t0;
        if dt < 0 {
            self.late_dropped += 1;
            return Ok(());
        }
        if dt >= 1 << T_BITS {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("t - t0 = {dt} µs 溢出 39 位(上限 ≈6.4 天)"),
            ));
        }
        let word = dt as u64
            | (x as u64) << T_BITS
            | (y as u64) << (T_BITS + 12)
            | ((p != 0) as u64) << 63;
        self.out.write_all(&word.to_le_bytes())?;
        self.written += 1;
        self.bytes_written += 8;
        Ok(())
    }


    pub fn late_dropped(&self) -> u64 { self.late_dropped }


    pub fn events_written(&self) -> u64 { self.written }


    pub fn bytes_written(&self) -> u64 { self.bytes_written }

    pub fn finalize(mut self) -> io::Result<()> {
        if self.t0_us.is_none() {
            self.write_header(0)?;
        }
        self.out.flush()
    }
}

pub struct EventsBinFile {
    pub width: u32,
    pub height: u32,
    pub t0_us: i64,

    pub events: Vec<(i64, u16, u16, u8)>,
}

pub fn read_events_bin(path: &Path) -> io::Result<EventsBinFile> {
    let bytes = std::fs::read(path)?;
    let bad = |msg: String| io::Error::new(io::ErrorKind::InvalidData, msg);
    if bytes.len() < HEADER_LEN as usize {
        return Err(bad(format!("events.bin 头部不完整: {} 字节 < 24", bytes.len())));
    }
    if &bytes[0..4] != b"FSEV" {
        return Err(bad(format!("magic 不是 FSEV: {:?}", &bytes[0..4])));
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if version != 1 {
        return Err(bad(format!("不认识的 events.bin 版本: {version}")));
    }
    let body = &bytes[HEADER_LEN as usize..];
    if body.len() % 8 != 0 {
        return Err(bad(format!("记录区长度 {} 不是 8 的倍数(尾部有撕裂记录)", body.len())));
    }
    let t0_us = i64::from_le_bytes(bytes[16..24].try_into().unwrap());
    let events = body
        .chunks_exact(8)
        .map(|c| {
            let w = u64::from_le_bytes(c.try_into().unwrap());
            let t = t0_us + (w & ((1 << T_BITS) - 1)) as i64;
            let x = (w >> T_BITS) as u16 & COORD_MAX;
            let y = (w >> (T_BITS + 12)) as u16 & COORD_MAX;
            (t, x, y, (w >> 63) as u8)
        })
        .collect();
    Ok(EventsBinFile {
        width: u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
        height: u32::from_le_bytes(bytes[12..16].try_into().unwrap()),
        t0_us,
        events,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct XorShift64(u64);
    impl XorShift64 {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
    }


    fn raw_word(bytes: &[u8], i: usize) -> u64 {
        let off = 24 + i * 8;
        u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap())
    }

    #[test]
    fn header_fields_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("events.bin");
        let mut w = EventsBinWriter::create(&path, (1280, 720)).unwrap();
        w.push(5_591_040, 7, 9, 1).unwrap();
        w.finalize().unwrap();

        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[0..4], b"FSEV");
        assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 1);
        assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), 1280);
        assert_eq!(u32::from_le_bytes(bytes[12..16].try_into().unwrap()), 720);
        assert_eq!(i64::from_le_bytes(bytes[16..24].try_into().unwrap()), 5_591_040);

        let f = read_events_bin(&path).unwrap();
        assert_eq!(f.width, 1280);
        assert_eq!(f.height, 720);
        assert_eq!(f.t0_us, 5_591_040);
        assert_eq!(f.events, vec![(5_591_040, 7, 9, 1)]);
    }

    #[test]
    fn single_event_bit_packing_at_the_boundaries() {
        let t0 = 1_000_i64;
        let max_dt = (1_i64 << 39) - 1;
        let cases: [(i64, u16, u16, u8, u64); 5] = [
            (t0, 0, 0, 0, 0),
            (t0 + max_dt, 4095, 4095, 1, u64::MAX),
            (t0, 4095, 0, 0, 0x0007_FF80_0000_0000),
            (t0, 0, 4095, 0, 0x7FF8_0000_0000_0000),
            (t0 + 1, 2, 3, 1, 0x8018_0100_0000_0001),
        ];
        for (t, x, y, p, want) in cases {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("events.bin");
            let mut w = EventsBinWriter::create(&path, (1280, 720)).unwrap();
            if t != t0 {
                w.push(t0, 0, 0, 0).unwrap();
            }
            w.push(t, x, y, p).unwrap();
            w.finalize().unwrap();

            let bytes = std::fs::read(&path).unwrap();
            let i = if t != t0 { 1 } else { 0 };
            assert_eq!(raw_word(&bytes, i), want, "打包字不符: t={t} x={x} y={y} p={p}");

            let f = read_events_bin(&path).unwrap();
            assert_eq!(f.events[i], (t, x, y, p), "读取端往返不符");
        }
    }

    #[test]
    fn round_trips_10k_random_events() {
        let mut rng = XorShift64(0x9E37_79B9_7F4A_7C15);
        let t_base = 5_591_040_i64;
        let mut input: Vec<(i64, u16, u16, u8)> = (0..10_000)
            .map(|_| {
                let t = t_base + (rng.next() % (1 << 38)) as i64;
                let x = (rng.next() % 4096) as u16;
                let y = (rng.next() % 4096) as u16;
                let p = (rng.next() & 1) as u8;
                (t, x, y, p)
            })
            .collect();
        input.sort_by_key(|e| e.0);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("events.bin");
        let mut w = EventsBinWriter::create(&path, (1280, 720)).unwrap();
        for &(t, x, y, p) in &input {
            w.push(t, x, y, p).unwrap();
        }
        assert_eq!(w.late_dropped(), 0);
        w.finalize().unwrap();

        let f = read_events_bin(&path).unwrap();
        assert_eq!(f.t0_us, input[0].0);
        assert_eq!(f.events.len(), 10_000);
        assert_eq!(f.events, input);
    }

    #[test]
    fn late_events_are_dropped_and_counted() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("events.bin");
        let mut w = EventsBinWriter::create(&path, (1280, 720)).unwrap();
        w.push(1_000, 1, 1, 0).unwrap();
        w.push(999, 2, 2, 1).unwrap();
        w.push(500, 3, 3, 0).unwrap();
        assert_eq!(w.late_dropped(), 2);
        assert_eq!(w.events_written(), 1, "迟到事件不许进 events_written");
        assert_eq!(w.bytes_written(), 24 + 8, "迟到事件不许占字节");
        w.push(1_001, 4, 4, 1).unwrap();
        w.finalize().unwrap();

        let f = read_events_bin(&path).unwrap();
        assert_eq!(f.events, vec![(1_000, 1, 1, 0), (1_001, 4, 4, 1)]);
    }

    #[test]
    fn overflowing_39_bits_errs_and_leaves_a_valid_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("events.bin");
        let mut w = EventsBinWriter::create(&path, (1280, 720)).unwrap();
        w.push(0, 1, 1, 1).unwrap();
        w.push((1 << 39) - 1, 2, 2, 0).unwrap();
        assert!(w.push(1 << 39, 3, 3, 0).is_err());
        assert_eq!(w.bytes_written(), 24 + 16, "溢出事件不许留下半条记录");
        w.finalize().unwrap();

        let f = read_events_bin(&path).unwrap();
        assert_eq!(f.events, vec![(0, 1, 1, 1), ((1 << 39) - 1, 2, 2, 0)]);
    }

    #[test]
    fn zero_event_take_finalizes_as_a_pure_header() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("events.bin");
        let w = EventsBinWriter::create(&path, (640, 480)).unwrap();
        w.finalize().unwrap();

        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes.len(), 24);
        assert_eq!(&bytes[0..4], b"FSEV");
        assert_eq!(i64::from_le_bytes(bytes[16..24].try_into().unwrap()), 0);

        let f = read_events_bin(&path).unwrap();
        assert_eq!(f.t0_us, 0);
        assert!(f.events.is_empty());
    }

    #[test]
    fn oversize_coordinates_err_without_corrupting_state() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("events.bin");
        let mut w = EventsBinWriter::create(&path, (1280, 720)).unwrap();
        assert!(w.push(50, 4096, 0, 0).is_err());
        assert!(w.push(50, 0, 4096, 0).is_err());
        assert_eq!(w.bytes_written(), 0, "非法事件不许触发头部落盘");
        w.push(100, 4095, 4095, 1).unwrap();
        w.finalize().unwrap();

        let f = read_events_bin(&path).unwrap();
        assert_eq!(f.t0_us, 100, "t0 必须来自首个合法事件,而不是被拒绝的那个");
        assert_eq!(f.events, vec![(100, 4095, 4095, 1)]);
    }

    #[test]
    fn bytes_written_is_exactly_header_plus_records() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("events.bin");
        let mut w = EventsBinWriter::create(&path, (1280, 720)).unwrap();
        assert_eq!(w.bytes_written(), 0, "首个事件之前头部不存在,不许虚报");
        for i in 0..5 {
            w.push(1_000 + i, i as u16, i as u16, 0).unwrap();
            assert_eq!(w.bytes_written(), 24 + (i as u64 + 1) * 8);
            assert_eq!(w.events_written(), i as u64 + 1, "events_written 与落盘记录数必须逐条一致");
        }
        w.finalize().unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().len(), 24 + 5 * 8);
    }

    #[test]
    fn reader_rejects_malformed_files() {
        let dir = tempfile::tempdir().unwrap();

        let short = dir.path().join("short.bin");
        std::fs::write(&short, b"FSEV").unwrap();
        assert!(read_events_bin(&short).is_err());

        let bad_magic = dir.path().join("bad_magic.bin");
        let mut bytes = vec![0u8; 24];
        bytes[0..4].copy_from_slice(b"EVT3");
        std::fs::write(&bad_magic, &bytes).unwrap();
        assert!(read_events_bin(&bad_magic).is_err());

        let torn = dir.path().join("torn.bin");
        let mut bytes = vec![0u8; 24 + 13];
        bytes[0..4].copy_from_slice(b"FSEV");
        bytes[4..8].copy_from_slice(&1u32.to_le_bytes());
        std::fs::write(&torn, &bytes).unwrap();
        assert!(read_events_bin(&torn).is_err());

        let bad_version = dir.path().join("bad_version.bin");
        let mut bytes = vec![0u8; 24];
        bytes[0..4].copy_from_slice(b"FSEV");
        bytes[4..8].copy_from_slice(&2u32.to_le_bytes());
        std::fs::write(&bad_version, &bytes).unwrap();
        assert!(read_events_bin(&bad_version).is_err());
    }
}
