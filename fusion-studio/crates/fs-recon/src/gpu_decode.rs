
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

const KERNELS_SRC: &str = include_str!("gpu_decode.cu");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodedEvent {
    pub x: u16,
    pub y: u16,
    pub p: u8,
    pub t_us: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodedTrigger {
    pub t_us: i64,
    pub polarity: u8,
    pub id: u8,
}


pub struct DecodedEventsDevice {
    pub x: CudaSlice<u16>,
    pub y: CudaSlice<u16>,
    pub p: CudaSlice<u8>,
    pub t_us: CudaSlice<i64>,
    pub len: usize,
}

impl DecodedEventsDevice {

    pub fn download(&self, stream: &Arc<CudaStream>) -> Result<Vec<DecodedEvent>, String> {
        let xs = stream.clone_dtoh(&self.x).map_err(|e| format!("download x failed: {e}"))?;
        let ys = stream.clone_dtoh(&self.y).map_err(|e| format!("download y failed: {e}"))?;
        let ps = stream.clone_dtoh(&self.p).map_err(|e| format!("download p failed: {e}"))?;
        let ts = stream.clone_dtoh(&self.t_us).map_err(|e| format!("download t_us failed: {e}"))?;
        Ok((0..self.len)
            .map(|i| DecodedEvent { x: xs[i], y: ys[i], p: ps[i], t_us: ts[i] })
            .collect())
    }


    pub fn download_range(&self, stream: &Arc<CudaStream>, start: usize, count: usize) -> Result<Vec<DecodedEvent>, String> {
        let end = (start + count).min(self.len);
        if start >= end {
            return Ok(Vec::new());
        }
        let xs = stream
            .clone_dtoh(&self.x.slice(start..end))
            .map_err(|e| format!("download x range failed: {e}"))?;
        let ys = stream
            .clone_dtoh(&self.y.slice(start..end))
            .map_err(|e| format!("download y range failed: {e}"))?;
        let ps = stream
            .clone_dtoh(&self.p.slice(start..end))
            .map_err(|e| format!("download p range failed: {e}"))?;
        let ts = stream
            .clone_dtoh(&self.t_us.slice(start..end))
            .map_err(|e| format!("download t_us range failed: {e}"))?;
        Ok((0..xs.len())
            .map(|i| DecodedEvent { x: xs[i], y: ys[i], p: ps[i], t_us: ts[i] })
            .collect())
    }
}


#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CarryState {

    epoch_us: i64,
    time_high: u16,
    time_low: u16,
    y: u16,
    base_x: u16,
    polarity: u8,
}

impl CarryState {

    pub fn time_us(&self) -> i64 {
        self.epoch_us + ((self.time_high as i64) << 12) + self.time_low as i64
    }
}


#[derive(Clone, Copy, Debug, Default)]
struct ChunkDelta {
    cd_count: u32,
    trig_count: u32,
    has_time_high: bool,
    first_time_high: u16,
    last_time_high: u16,
    internal_wraps: u32,
    has_time_low: bool,
    last_time_low: u16,
    has_y: bool,
    last_y: u16,
    has_base_x: bool,
    last_base_x: u16,
    leading_advance: u32,
    last_polarity: u8,
}

fn resolve_carry_states(
    deltas: &[ChunkDelta],
    init: CarryState,
) -> (Vec<CarryState>, Vec<u32>, Vec<u32>, u64, u64, CarryState) {
    let n = deltas.len();
    let mut carry_in = Vec::with_capacity(n);
    let mut cd_offsets = Vec::with_capacity(n);
    let mut trig_offsets = Vec::with_capacity(n);

    let mut state = init;
    let mut cd_total: u64 = 0;
    let mut trig_total: u64 = 0;

    for d in deltas {
        carry_in.push(state);
        cd_offsets.push(cd_total as u32);
        trig_offsets.push(trig_total as u32);

        if d.has_time_high {
            if d.first_time_high < state.time_high {
                state.epoch_us += 1i64 << 24;
            }
            state.epoch_us += (d.internal_wraps as i64) << 24;
            state.time_high = d.last_time_high;
        }
        if d.has_time_low {
            state.time_low = d.last_time_low;
        }
        if d.has_y {
            state.y = d.last_y;
        }
        if d.has_base_x {
            state.base_x = d.last_base_x;
            state.polarity = d.last_polarity;
        } else if d.leading_advance != 0 {
            state.base_x = state.base_x.wrapping_add(d.leading_advance as u16);
        }

        cd_total += d.cd_count as u64;
        trig_total += d.trig_count as u64;
    }

    (carry_in, cd_offsets, trig_offsets, cd_total, trig_total, state)
}


#[derive(Clone, Copy, Debug, Default)]
pub struct DecodeStats {
    pub n_words: usize,
    pub n_chunks: usize,
    pub cd_events: usize,
    pub triggers: usize,
    pub upload_words_ms: f64,
    pub upload_chunk_meta_ms: f64,
    pub pass1_kernel_ms: f64,
    pub download_pass1_ms: f64,
    pub host_scan_ms: f64,
    pub upload_carry_ms: f64,
    pub alloc_output_ms: f64,
    pub pass2_kernel_ms: f64,
    pub download_triggers_ms: f64,
    pub total_ms: f64,
}

pub struct GpuEventDecoder {
    stream: Arc<CudaStream>,
    scan_fn: CudaFunction,
    decode_fn: CudaFunction,
}

impl GpuEventDecoder {
    pub fn new() -> Result<Self, String> {
        let ctx = CudaContext::new(0).map_err(|e| format!("cuda device init failed: {e}"))?;
        let stream = ctx.default_stream();
        let ptx = compile_ptx(KERNELS_SRC).map_err(|e| format!("nvrtc compile failed: {e}"))?;
        let module = ctx.load_module(ptx).map_err(|e| format!("cuda module load failed: {e}"))?;
        let scan_fn = module
            .load_function("evt3_scan_chunks")
            .map_err(|e| format!("cuda function load failed: {e}"))?;
        let decode_fn = module
            .load_function("evt3_decode_chunks")
            .map_err(|e| format!("cuda function load failed: {e}"))?;
        Ok(Self { stream, scan_fn, decode_fn })
    }


    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }


    pub fn decode(
        &mut self,
        words: &[u16],
        chunk_words: usize,
    ) -> Result<(DecodedEventsDevice, Vec<DecodedTrigger>, DecodeStats), String> {
        let mut state = CarryState::default();
        self.decode_with_state(words, chunk_words, &mut state)
    }

    pub fn decode_with_state(
        &mut self,
        words: &[u16],
        chunk_words: usize,
        state: &mut CarryState,
    ) -> Result<(DecodedEventsDevice, Vec<DecodedTrigger>, DecodeStats), String> {
        let t_total0 = Instant::now();
        let chunk_words = chunk_words.max(1);
        let n_words = words.len();
        let n_chunks = if n_words == 0 { 0 } else { n_words.div_ceil(chunk_words) };

        let mut chunk_starts = Vec::with_capacity(n_chunks);
        let mut chunk_lens = Vec::with_capacity(n_chunks);
        for i in 0..n_chunks {
            let start = i * chunk_words;
            let len = chunk_words.min(n_words - start);
            chunk_starts.push(start as u32);
            chunk_lens.push(len as u32);
        }

        let t0 = Instant::now();
        let d_words = self.stream.clone_htod(words).map_err(|e| format!("upload words failed: {e}"))?;
        let upload_words_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = Instant::now();
        let d_chunk_starts =
            self.stream.clone_htod(&chunk_starts).map_err(|e| format!("upload chunk_starts failed: {e}"))?;
        let d_chunk_lens = self.stream.clone_htod(&chunk_lens).map_err(|e| format!("upload chunk_lens failed: {e}"))?;
        let upload_chunk_meta_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let cap = n_chunks.max(1);
        let mut d_cd_count = self.stream.alloc_zeros::<u32>(cap).map_err(|e| format!("alloc failed: {e}"))?;
        let mut d_trig_count = self.stream.alloc_zeros::<u32>(cap).map_err(|e| format!("alloc failed: {e}"))?;
        let mut d_has_th = self.stream.alloc_zeros::<u8>(cap).map_err(|e| format!("alloc failed: {e}"))?;
        let mut d_first_th = self.stream.alloc_zeros::<u16>(cap).map_err(|e| format!("alloc failed: {e}"))?;
        let mut d_last_th = self.stream.alloc_zeros::<u16>(cap).map_err(|e| format!("alloc failed: {e}"))?;
        let mut d_internal_wraps = self.stream.alloc_zeros::<u32>(cap).map_err(|e| format!("alloc failed: {e}"))?;
        let mut d_has_tl = self.stream.alloc_zeros::<u8>(cap).map_err(|e| format!("alloc failed: {e}"))?;
        let mut d_last_tl = self.stream.alloc_zeros::<u16>(cap).map_err(|e| format!("alloc failed: {e}"))?;
        let mut d_has_y = self.stream.alloc_zeros::<u8>(cap).map_err(|e| format!("alloc failed: {e}"))?;
        let mut d_last_y = self.stream.alloc_zeros::<u16>(cap).map_err(|e| format!("alloc failed: {e}"))?;
        let mut d_has_bx = self.stream.alloc_zeros::<u8>(cap).map_err(|e| format!("alloc failed: {e}"))?;
        let mut d_last_bx = self.stream.alloc_zeros::<u16>(cap).map_err(|e| format!("alloc failed: {e}"))?;
        let mut d_leading_adv = self.stream.alloc_zeros::<u32>(cap).map_err(|e| format!("alloc failed: {e}"))?;
        let mut d_last_pol = self.stream.alloc_zeros::<u8>(cap).map_err(|e| format!("alloc failed: {e}"))?;

        let t0 = Instant::now();
        if n_chunks > 0 {
            let cfg = LaunchConfig::for_num_elems(n_chunks as u32);
            let n_chunks_i32 = n_chunks as i32;
            let mut args = self.stream.launch_builder(&self.scan_fn);
            args.arg(&d_words);
            args.arg(&d_chunk_starts);
            args.arg(&d_chunk_lens);
            args.arg(&n_chunks_i32);
            args.arg(&mut d_cd_count);
            args.arg(&mut d_trig_count);
            args.arg(&mut d_has_th);
            args.arg(&mut d_first_th);
            args.arg(&mut d_last_th);
            args.arg(&mut d_internal_wraps);
            args.arg(&mut d_has_tl);
            args.arg(&mut d_last_tl);
            args.arg(&mut d_has_y);
            args.arg(&mut d_last_y);
            args.arg(&mut d_has_bx);
            args.arg(&mut d_last_bx);
            args.arg(&mut d_leading_adv);
            args.arg(&mut d_last_pol);
            unsafe { args.launch(cfg) }.map_err(|e| format!("pass1 launch failed: {e}"))?;
            self.stream.synchronize().map_err(|e| format!("pass1 sync failed: {e}"))?;
        }
        let pass1_kernel_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = Instant::now();
        let h_cd_count = self.stream.clone_dtoh(&d_cd_count).map_err(|e| format!("download failed: {e}"))?;
        let h_trig_count = self.stream.clone_dtoh(&d_trig_count).map_err(|e| format!("download failed: {e}"))?;
        let h_has_th = self.stream.clone_dtoh(&d_has_th).map_err(|e| format!("download failed: {e}"))?;
        let h_first_th = self.stream.clone_dtoh(&d_first_th).map_err(|e| format!("download failed: {e}"))?;
        let h_last_th = self.stream.clone_dtoh(&d_last_th).map_err(|e| format!("download failed: {e}"))?;
        let h_internal_wraps = self.stream.clone_dtoh(&d_internal_wraps).map_err(|e| format!("download failed: {e}"))?;
        let h_has_tl = self.stream.clone_dtoh(&d_has_tl).map_err(|e| format!("download failed: {e}"))?;
        let h_last_tl = self.stream.clone_dtoh(&d_last_tl).map_err(|e| format!("download failed: {e}"))?;
        let h_has_y = self.stream.clone_dtoh(&d_has_y).map_err(|e| format!("download failed: {e}"))?;
        let h_last_y = self.stream.clone_dtoh(&d_last_y).map_err(|e| format!("download failed: {e}"))?;
        let h_has_bx = self.stream.clone_dtoh(&d_has_bx).map_err(|e| format!("download failed: {e}"))?;
        let h_last_bx = self.stream.clone_dtoh(&d_last_bx).map_err(|e| format!("download failed: {e}"))?;
        let h_leading_adv = self.stream.clone_dtoh(&d_leading_adv).map_err(|e| format!("download failed: {e}"))?;
        let h_last_pol = self.stream.clone_dtoh(&d_last_pol).map_err(|e| format!("download failed: {e}"))?;
        let download_pass1_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = Instant::now();
        let deltas: Vec<ChunkDelta> = (0..n_chunks)
            .map(|i| ChunkDelta {
                cd_count: h_cd_count[i],
                trig_count: h_trig_count[i],
                has_time_high: h_has_th[i] != 0,
                first_time_high: h_first_th[i],
                last_time_high: h_last_th[i],
                internal_wraps: h_internal_wraps[i],
                has_time_low: h_has_tl[i] != 0,
                last_time_low: h_last_tl[i],
                has_y: h_has_y[i] != 0,
                last_y: h_last_y[i],
                has_base_x: h_has_bx[i] != 0,
                last_base_x: h_last_bx[i],
                leading_advance: h_leading_adv[i],
                last_polarity: h_last_pol[i],
            })
            .collect();
        let (carry_in, cd_offsets, trig_offsets, total_cd, total_trig, final_state) =
            resolve_carry_states(&deltas, *state);
        let host_scan_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = Instant::now();
        let carry_epoch: Vec<i64> = carry_in.iter().map(|s| s.epoch_us).collect();
        let carry_th: Vec<u16> = carry_in.iter().map(|s| s.time_high).collect();
        let carry_tl: Vec<u16> = carry_in.iter().map(|s| s.time_low).collect();
        let carry_y: Vec<u16> = carry_in.iter().map(|s| s.y).collect();
        let carry_bx: Vec<u16> = carry_in.iter().map(|s| s.base_x).collect();
        let carry_pol: Vec<u8> = carry_in.iter().map(|s| s.polarity).collect();

        let d_carry_epoch = self.stream.clone_htod(&carry_epoch).map_err(|e| format!("upload carry failed: {e}"))?;
        let d_carry_th = self.stream.clone_htod(&carry_th).map_err(|e| format!("upload carry failed: {e}"))?;
        let d_carry_tl = self.stream.clone_htod(&carry_tl).map_err(|e| format!("upload carry failed: {e}"))?;
        let d_carry_y = self.stream.clone_htod(&carry_y).map_err(|e| format!("upload carry failed: {e}"))?;
        let d_carry_bx = self.stream.clone_htod(&carry_bx).map_err(|e| format!("upload carry failed: {e}"))?;
        let d_carry_pol = self.stream.clone_htod(&carry_pol).map_err(|e| format!("upload carry failed: {e}"))?;
        let d_cd_offsets = self.stream.clone_htod(&cd_offsets).map_err(|e| format!("upload offsets failed: {e}"))?;
        let d_trig_offsets = self.stream.clone_htod(&trig_offsets).map_err(|e| format!("upload offsets failed: {e}"))?;
        let upload_carry_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = Instant::now();
        let n_cd = total_cd as usize;
        let n_trig = total_trig as usize;
        let mut out_x = self.stream.alloc_zeros::<u16>(n_cd.max(1)).map_err(|e| format!("alloc failed: {e}"))?;
        let mut out_y = self.stream.alloc_zeros::<u16>(n_cd.max(1)).map_err(|e| format!("alloc failed: {e}"))?;
        let mut out_p = self.stream.alloc_zeros::<u8>(n_cd.max(1)).map_err(|e| format!("alloc failed: {e}"))?;
        let mut out_t = self.stream.alloc_zeros::<i64>(n_cd.max(1)).map_err(|e| format!("alloc failed: {e}"))?;
        let mut out_trig_t = self.stream.alloc_zeros::<i64>(n_trig.max(1)).map_err(|e| format!("alloc failed: {e}"))?;
        let mut out_trig_p = self.stream.alloc_zeros::<u8>(n_trig.max(1)).map_err(|e| format!("alloc failed: {e}"))?;
        let mut out_trig_id = self.stream.alloc_zeros::<u8>(n_trig.max(1)).map_err(|e| format!("alloc failed: {e}"))?;
        let alloc_output_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = Instant::now();
        if n_chunks > 0 {
            let cfg = LaunchConfig::for_num_elems(n_chunks as u32);
            let n_chunks_i32 = n_chunks as i32;
            let mut args = self.stream.launch_builder(&self.decode_fn);
            args.arg(&d_words);
            args.arg(&d_chunk_starts);
            args.arg(&d_chunk_lens);
            args.arg(&n_chunks_i32);
            args.arg(&d_carry_epoch);
            args.arg(&d_carry_th);
            args.arg(&d_carry_tl);
            args.arg(&d_carry_y);
            args.arg(&d_carry_bx);
            args.arg(&d_carry_pol);
            args.arg(&d_cd_offsets);
            args.arg(&d_trig_offsets);
            args.arg(&mut out_x);
            args.arg(&mut out_y);
            args.arg(&mut out_p);
            args.arg(&mut out_t);
            args.arg(&mut out_trig_t);
            args.arg(&mut out_trig_p);
            args.arg(&mut out_trig_id);
            unsafe { args.launch(cfg) }.map_err(|e| format!("pass2 launch failed: {e}"))?;
            self.stream.synchronize().map_err(|e| format!("pass2 sync failed: {e}"))?;
        }
        let pass2_kernel_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = Instant::now();
        let triggers = if n_trig > 0 {
            let ts = self.stream.clone_dtoh(&out_trig_t).map_err(|e| format!("download triggers failed: {e}"))?;
            let ps = self.stream.clone_dtoh(&out_trig_p).map_err(|e| format!("download triggers failed: {e}"))?;
            let ids = self.stream.clone_dtoh(&out_trig_id).map_err(|e| format!("download triggers failed: {e}"))?;
            (0..n_trig).map(|i| DecodedTrigger { t_us: ts[i], polarity: ps[i], id: ids[i] }).collect()
        } else {
            Vec::new()
        };
        let download_triggers_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let total_ms = t_total0.elapsed().as_secs_f64() * 1000.0;

        let events = DecodedEventsDevice { x: out_x, y: out_y, p: out_p, t_us: out_t, len: n_cd };
        let stats = DecodeStats {
            n_words,
            n_chunks,
            cd_events: n_cd,
            triggers: n_trig,
            upload_words_ms,
            upload_chunk_meta_ms,
            pass1_kernel_ms,
            download_pass1_ms,
            host_scan_ms,
            upload_carry_ms,
            alloc_output_ms,
            pass2_kernel_ms,
            download_triggers_ms,
            total_ms,
        };
        *state = final_state;
        Ok((events, triggers, stats))
    }
}


pub fn scan_triggers_cpu(words: &[u16]) -> Vec<DecodedTrigger> {
    let mut scanner = CpuTriggerScanner::default();
    scanner.push(words)
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CpuTriggerScanner {
    epoch_us: i64,
    time_high: u16,
    time_low: u16,
}

impl CpuTriggerScanner {

    pub fn push(&mut self, words: &[u16]) -> Vec<DecodedTrigger> {
        let mut triggers = Vec::new();
        for &w in words {
            let ty = w >> 12;
            match ty {
                0x6 => self.time_low = w & 0xFFF,
                0x8 => {
                    let new_th = w & 0xFFF;
                    if new_th < self.time_high {
                        self.epoch_us += 1i64 << 24;
                    }
                    self.time_high = new_th;
                }
                0xA => {
                    let polarity = (w & 1) as u8;
                    let id = ((w >> 8) & 0xF) as u8;
                    let t_us = self.epoch_us + ((self.time_high as i64) << 12) + self.time_low as i64;
                    triggers.push(DecodedTrigger { t_us, polarity, id });
                }
                _ => {}
            }
        }
        triggers
    }


    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(ty: u16, payload: u16) -> u16 {
        (ty << 12) | (payload & 0x0FFF)
    }
    fn time_high(th: u16) -> u16 {
        word(0x8, th)
    }
    fn time_low(tl: u16) -> u16 {
        word(0x6, tl)
    }
    fn addr_y(y: u16) -> u16 {
        word(0x0, y)
    }
    fn addr_x(x: u16, pol: u16) -> u16 {
        word(0x2, (pol << 11) | x)
    }
    fn vect_base_x(base_x: u16, pol: u16) -> u16 {
        word(0x3, (pol << 11) | base_x)
    }
    fn vect_12(mask: u16) -> u16 {
        word(0x4, mask)
    }
    fn vect_8(mask: u16) -> u16 {
        word(0x5, mask & 0xFF)
    }
    fn ext_trigger(id: u16, pol: u16) -> u16 {
        word(0xA, (id << 8) | pol)
    }
    fn others(subtype: u16) -> u16 {
        word(0xE, subtype)
    }
    fn continued_12(payload: u16) -> u16 {
        word(0xF, payload)
    }
    fn continued_4(payload: u16) -> u16 {
        word(0x7, payload)
    }

    fn decode_and_check_chunk_invariance(
        dec: &mut GpuEventDecoder,
        words: &[u16],
    ) -> (Vec<DecodedEvent>, Vec<DecodedTrigger>) {
        let (dev_whole, trig_whole, _stats) = dec.decode(words, words.len().max(1)).expect("decode (whole) failed");
        let events_whole = dev_whole.download(dec.stream()).expect("download failed");

        for &cw in &[1usize, 2, 3] {
            let (dev_c, trig_c, _stats_c) = dec.decode(words, cw).expect("decode (chunked) failed");
            let events_c = dev_c.download(dec.stream()).expect("download failed");
            assert_eq!(
                events_c, events_whole,
                "chunk_words={cw} decode should match the single-chunk decode"
            );
            assert_eq!(trig_c, trig_whole, "chunk_words={cw} triggers should match the single-chunk decode");
        }

        (events_whole, trig_whole)
    }

    #[test]
    fn decodes_single_addr_x_event() {
        let Ok(mut dec) = GpuEventDecoder::new() else {
            eprintln!("skip: no CUDA");
            return;
        };
        let words = [time_high(5), time_low(100), addr_y(10), addr_x(20, 1)];
        let (events, triggers) = decode_and_check_chunk_invariance(&mut dec, &words);
        assert_eq!(triggers.len(), 0);
        assert_eq!(events, vec![DecodedEvent { x: 20, y: 10, p: 1, t_us: 5 * 4096 + 100 }]);
    }

    #[test]
    fn decodes_vect_12_events() {
        let Ok(mut dec) = GpuEventDecoder::new() else {
            eprintln!("skip: no CUDA");
            return;
        };
        let words = [time_high(1), time_low(0), addr_y(50), vect_base_x(100, 0), vect_12(0b0000_0000_0101)];
        let (events, triggers) = decode_and_check_chunk_invariance(&mut dec, &words);
        assert_eq!(triggers.len(), 0);
        assert_eq!(
            events,
            vec![
                DecodedEvent { x: 100, y: 50, p: 0, t_us: 4096 },
                DecodedEvent { x: 102, y: 50, p: 0, t_us: 4096 },
            ]
        );
    }

    #[test]
    fn decodes_vect_8_events() {
        let Ok(mut dec) = GpuEventDecoder::new() else {
            eprintln!("skip: no CUDA");
            return;
        };
        let words = [time_high(2), time_low(0), addr_y(7), vect_base_x(30, 1), vect_8(0b0000_0011)];
        let (events, _triggers) = decode_and_check_chunk_invariance(&mut dec, &words);
        assert_eq!(
            events,
            vec![
                DecodedEvent { x: 30, y: 7, p: 1, t_us: 2 * 4096 },
                DecodedEvent { x: 31, y: 7, p: 1, t_us: 2 * 4096 },
            ]
        );
    }

    #[test]
    fn vector_run_spanning_multiple_words_advances_base_x_across_chunks() {
        let Ok(mut dec) = GpuEventDecoder::new() else {
            eprintln!("skip: no CUDA");
            return;
        };
        let words = [
            time_high(1),
            time_low(0),
            addr_y(9),
            vect_base_x(0, 1),
            vect_12(0b1),
            vect_12(0b1),
            vect_8(0b1),
        ];
        let (events, _triggers) = decode_and_check_chunk_invariance(&mut dec, &words);
        let xs: Vec<u16> = events.iter().map(|e| e.x).collect();
        assert_eq!(xs, vec![0, 12, 24], "base_x must keep advancing across chunk boundaries with no new VECT_BASE_X");
        assert!(events.iter().all(|e| e.y == 9 && e.p == 1));
    }

    #[test]
    fn decodes_time_high_rollover() {
        let Ok(mut dec) = GpuEventDecoder::new() else {
            eprintln!("skip: no CUDA");
            return;
        };
        let words = [
            time_high(4095),
            time_low(50),
            addr_y(5),
            addr_x(1, 0),
            time_high(0),
            time_low(10),
            addr_x(2, 1),
        ];
        let (events, _triggers) = decode_and_check_chunk_invariance(&mut dec, &words);
        assert_eq!(
            events,
            vec![
                DecodedEvent { x: 1, y: 5, p: 0, t_us: 4095 * 4096 + 50 },
                DecodedEvent { x: 2, y: 5, p: 1, t_us: (1i64 << 24) + 10 },
            ]
        );
    }

    #[test]
    fn decodes_trigger_word() {
        let Ok(mut dec) = GpuEventDecoder::new() else {
            eprintln!("skip: no CUDA");
            return;
        };
        let words = [time_high(2), time_low(200), ext_trigger(0, 1)];
        let (events, triggers) = decode_and_check_chunk_invariance(&mut dec, &words);
        assert_eq!(events.len(), 0);
        assert_eq!(triggers, vec![DecodedTrigger { t_us: 2 * 4096 + 200, polarity: 1, id: 0 }]);

        let cpu_triggers = scan_triggers_cpu(&words);
        assert_eq!(cpu_triggers, triggers);
    }

    #[test]
    fn others_and_continued_words_are_skipped_without_state_change() {
        let Ok(mut dec) = GpuEventDecoder::new() else {
            eprintln!("skip: no CUDA");
            return;
        };
        let words = [
            time_high(1),
            time_low(0),
            others(20),
            continued_12(0),
            continued_12(0),
            continued_4(0),
            addr_y(3),
            addr_x(4, 0),
        ];
        let (events, triggers) = decode_and_check_chunk_invariance(&mut dec, &words);
        assert_eq!(triggers.len(), 0);
        assert_eq!(events, vec![DecodedEvent { x: 4, y: 3, p: 0, t_us: 4096 }]);
    }


    #[test]
    fn push_events_device_default_impl_feeds_accumulator() {
        use crate::{Accumulator, Reconstructor};
        use fs_core::GrayImage;

        let Ok(mut dec) = GpuEventDecoder::new() else {
            eprintln!("skip: no CUDA");
            return;
        };
        let mut recon = Accumulator::new(64, 64);

        let words = [time_high(0), time_low(0), addr_y(10), addr_x(10, 1)];
        let (dev, triggers, _stats) = dec.decode(&words, 1024).expect("decode failed");
        assert_eq!(dev.len, 1);
        assert_eq!(triggers.len(), 0);

        recon.push_events_device(&dev, dec.stream());

        let mut img = GrayImage::new(64, 64);
        recon.render_at(1, &mut img);
        let px = img.data[10 * 64 + 10] as i32;
        let bg = img.data[0] as i32;
        assert!(px > bg, "push_events_device's default (download + push_events) should integrate the decoded ON event, got px={px} bg={bg}");
    }

    #[test]
    fn empty_input_decodes_to_nothing() {
        let Ok(mut dec) = GpuEventDecoder::new() else {
            eprintln!("skip: no CUDA");
            return;
        };
        let (dev, triggers, stats) = dec.decode(&[], 1024).expect("decode failed");
        assert_eq!(dev.len, 0);
        assert_eq!(triggers.len(), 0);
        assert_eq!(stats.n_chunks, 0);
    }


    fn assert_streaming_matches_whole(dec: &mut GpuEventDecoder, words: &[u16], splits: &[usize]) {
        let (dev_whole, trig_whole, _stats) = dec.decode(words, words.len().max(1)).expect("whole decode failed");
        let events_whole = dev_whole.download(dec.stream()).expect("download failed");

        for &chunk_words in &[1usize, 2, 1024] {
            let mut state = CarryState::default();
            let mut events_stream = Vec::new();
            let mut trig_stream = Vec::new();
            let mut prev = 0usize;
            let mut boundaries: Vec<usize> = splits.to_vec();
            boundaries.push(words.len());
            for &b in &boundaries {
                assert!(b >= prev && b <= words.len(), "bad split spec {b}");
                let (dev, trig, _s) =
                    dec.decode_with_state(&words[prev..b], chunk_words, &mut state).expect("streaming decode failed");
                events_stream.extend(dev.download(dec.stream()).expect("download failed"));
                trig_stream.extend(trig);
                prev = b;
            }
            assert_eq!(
                events_stream, events_whole,
                "streaming decode (splits={splits:?}, chunk_words={chunk_words}) must match whole-stream decode"
            );
            assert_eq!(trig_stream, trig_whole, "streaming triggers must match whole-stream decode");
            if let Some(last) = events_whole.last() {
                assert!(
                    state.time_us() >= last.t_us,
                    "carry-out time_us ({}) must not lag the last decoded event ({})",
                    state.time_us(),
                    last.t_us
                );
            }
        }
    }


    #[test]
    fn streaming_split_mid_vector_run() {
        let Ok(mut dec) = GpuEventDecoder::new() else {
            eprintln!("skip: no CUDA");
            return;
        };
        let words = [
            time_high(1),
            time_low(0),
            addr_y(9),
            vect_base_x(0, 1),
            vect_12(0b1),
            vect_12(0b1),
            vect_8(0b1),
        ];
        assert_streaming_matches_whole(&mut dec, &words, &[5]);
        assert_streaming_matches_whole(&mut dec, &words, &[4]);
    }

    #[test]
    fn streaming_split_between_time_high_and_time_low() {
        let Ok(mut dec) = GpuEventDecoder::new() else {
            eprintln!("skip: no CUDA");
            return;
        };
        let words = [
            time_high(7),
            time_low(123),
            addr_y(3),
            addr_x(4, 1),
            ext_trigger(0, 1),
        ];
        assert_streaming_matches_whole(&mut dec, &words, &[1]);
    }

    #[test]
    fn streaming_single_word_batches() {
        let Ok(mut dec) = GpuEventDecoder::new() else {
            eprintln!("skip: no CUDA");
            return;
        };
        let words = [
            time_high(4095),
            time_low(50),
            addr_y(5),
            vect_base_x(100, 0),
            vect_12(0b101),
            time_high(0),
            time_low(10),
            addr_x(2, 1),
            ext_trigger(1, 0),
            vect_8(0b11),
        ];
        let splits: Vec<usize> = (1..words.len()).collect();
        assert_streaming_matches_whole(&mut dec, &words, &splits);
    }

    #[test]
    fn streaming_rollover_and_empty_batches_across_calls() {
        let Ok(mut dec) = GpuEventDecoder::new() else {
            eprintln!("skip: no CUDA");
            return;
        };
        let words = [
            time_high(4000),
            time_low(1),
            addr_y(1),
            addr_x(1, 1),
            time_high(3),
            time_low(2),
            addr_x(2, 0),
        ];
        assert_streaming_matches_whole(&mut dec, &words, &[4]);

        let mut state = CarryState::default();
        let (d1, _t1, _) = dec.decode_with_state(&words[..4], 1024, &mut state).expect("decode failed");
        let mid_state = state;
        let (d_empty, t_empty, s_empty) = dec.decode_with_state(&[], 1024, &mut state).expect("decode failed");
        assert_eq!((d_empty.len, t_empty.len(), s_empty.n_chunks), (0, 0, 0));
        assert_eq!(state, mid_state, "empty batch must not change the carry state");
        let (d2, _t2, _) = dec.decode_with_state(&words[4..], 1024, &mut state).expect("decode failed");
        let e1 = d1.download(dec.stream()).expect("download failed");
        let e2 = d2.download(dec.stream()).expect("download failed");
        assert_eq!(e1.len() + e2.len(), 2);
        assert_eq!(
            e2,
            vec![DecodedEvent { x: 2, y: 1, p: 0, t_us: (1i64 << 24) + 3 * 4096 + 2 }],
            "rollover must be detected against the carry-in TIME_HIGH from the previous call"
        );
    }

    #[test]
    fn streaming_state_reset_restarts_epoch() {
        let Ok(mut dec) = GpuEventDecoder::new() else {
            eprintln!("skip: no CUDA");
            return;
        };
        let pre_reset = [time_high(3000), time_low(0), addr_y(1), addr_x(1, 1)];
        let post_reset = [time_high(1), time_low(5), addr_y(2), addr_x(3, 0)];

        let mut state = CarryState::default();
        let _ = dec.decode_with_state(&pre_reset, 1024, &mut state).expect("decode failed");
        let mut stale = state;
        let (dev_stale, _, _) = dec.decode_with_state(&post_reset, 1024, &mut stale).expect("decode failed");
        let stale_events = dev_stale.download(dec.stream()).expect("download failed");
        assert_eq!(stale_events[0].t_us, (1i64 << 24) + 4096 + 5);
        state = CarryState::default();
        let (dev_fresh, _, _) = dec.decode_with_state(&post_reset, 1024, &mut state).expect("decode failed");
        let fresh_events = dev_fresh.download(dec.stream()).expect("download failed");
        assert_eq!(fresh_events, vec![DecodedEvent { x: 3, y: 2, p: 0, t_us: 4096 + 5 }]);
    }


    #[test]
    fn cpu_trigger_scanner_streams_like_whole_scan() {
        let words = [
            time_high(4095),
            time_low(9),
            ext_trigger(0, 1),
            time_high(1),
            ext_trigger(2, 0),
            time_low(7),
            ext_trigger(0, 1),
        ];
        let whole = scan_triggers_cpu(&words);
        for split in 1..words.len() {
            let mut scanner = CpuTriggerScanner::default();
            let mut streamed = scanner.push(&words[..split]);
            streamed.extend(scanner.push(&words[split..]));
            assert_eq!(streamed, whole, "split at {split} must not change scanned triggers");
        }
        let mut scanner = CpuTriggerScanner::default();
        let _ = scanner.push(&words);
        scanner.reset();
        assert_eq!(scanner.push(&[ext_trigger(0, 1)])[0].t_us, 0, "reset must clear the time state");
    }
}
