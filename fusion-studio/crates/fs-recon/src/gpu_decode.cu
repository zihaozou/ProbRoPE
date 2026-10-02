
extern "C" __global__ void evt3_scan_chunks(
    const unsigned short* words,
    const unsigned int* chunk_starts,
    const unsigned int* chunk_lens,
    int n_chunks,
    unsigned int* out_cd_count,
    unsigned int* out_trig_count,
    unsigned char* out_has_th,
    unsigned short* out_first_th,
    unsigned short* out_last_th,
    unsigned int* out_internal_wraps,
    unsigned char* out_has_tl,
    unsigned short* out_last_tl,
    unsigned char* out_has_y,
    unsigned short* out_last_y,
    unsigned char* out_has_bx,
    unsigned short* out_last_bx,
    unsigned int* out_leading_advance,
    unsigned char* out_last_pol)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_chunks) return;

    unsigned int start = chunk_starts[c];
    unsigned int len = chunk_lens[c];

    unsigned int cd_count = 0;
    unsigned int trig_count = 0;

    unsigned char has_th = 0;
    unsigned short first_th = 0, last_th = 0;
    unsigned int internal_wraps = 0;

    unsigned char has_tl = 0;
    unsigned short last_tl = 0;

    unsigned char has_y = 0;
    unsigned short last_y = 0;

    unsigned char has_bx = 0;
    unsigned short bx = 0;
    unsigned char pol = 0;
    unsigned int leading_advance = 0;

    for (unsigned int i = 0; i < len; ++i) {
        unsigned short w = words[start + i];
        unsigned int type = (unsigned int)(w >> 12);
        switch (type) {
            case 0x0:
                last_y = (unsigned short)(w & 0x7FFu);
                has_y = 1;
                break;
            case 0x2:
                cd_count += 1;
                break;
            case 0x3:
                pol = (unsigned char)((w >> 11) & 1u);
                bx = (unsigned short)(w & 0x7FFu);
                has_bx = 1;
                break;
            case 0x4: {
                unsigned int mask = w & 0xFFFu;
                cd_count += (unsigned int)__popc(mask);
                if (has_bx) { bx = (unsigned short)(bx + 12); } else { leading_advance += 12; }
                break;
            }
            case 0x5: {
                unsigned int mask = w & 0xFFu;
                cd_count += (unsigned int)__popc(mask);
                if (has_bx) { bx = (unsigned short)(bx + 8); } else { leading_advance += 8; }
                break;
            }
            case 0x6:
                last_tl = (unsigned short)(w & 0xFFFu);
                has_tl = 1;
                break;
            case 0x8: {
                unsigned short th = (unsigned short)(w & 0xFFFu);
                if (has_th) {
                    if (th < last_th) internal_wraps += 1;
                } else {
                    first_th = th;
                }
                last_th = th;
                has_th = 1;
                break;
            }
            case 0xA:
                trig_count += 1;
                break;
            default:
                break;
        }
    }

    out_cd_count[c] = cd_count;
    out_trig_count[c] = trig_count;
    out_has_th[c] = has_th;
    out_first_th[c] = first_th;
    out_last_th[c] = last_th;
    out_internal_wraps[c] = internal_wraps;
    out_has_tl[c] = has_tl;
    out_last_tl[c] = last_tl;
    out_has_y[c] = has_y;
    out_last_y[c] = last_y;
    out_has_bx[c] = has_bx;
    out_last_bx[c] = bx;
    out_leading_advance[c] = leading_advance;
    out_last_pol[c] = pol;
}

extern "C" __global__ void evt3_decode_chunks(
    const unsigned short* words,
    const unsigned int* chunk_starts,
    const unsigned int* chunk_lens,
    int n_chunks,
    const long long* carry_epoch_us,
    const unsigned short* carry_time_high,
    const unsigned short* carry_time_low,
    const unsigned short* carry_y,
    const unsigned short* carry_base_x,
    const unsigned char* carry_polarity,
    const unsigned int* cd_offsets,
    const unsigned int* trig_offsets,
    unsigned short* out_x,
    unsigned short* out_y,
    unsigned char* out_p,
    long long* out_t_us,
    long long* out_trig_t_us,
    unsigned char* out_trig_p,
    unsigned char* out_trig_id)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_chunks) return;

    unsigned int start = chunk_starts[c];
    unsigned int len = chunk_lens[c];

    long long epoch_us = carry_epoch_us[c];
    unsigned short th = carry_time_high[c];
    unsigned short tl = carry_time_low[c];
    unsigned short y = carry_y[c];
    unsigned short bx = carry_base_x[c];
    unsigned char pol = carry_polarity[c];

    unsigned int cd_cursor = cd_offsets[c];
    unsigned int trig_cursor = trig_offsets[c];

    for (unsigned int i = 0; i < len; ++i) {
        unsigned short w = words[start + i];
        unsigned int type = (unsigned int)(w >> 12);
        switch (type) {
            case 0x0:
                y = (unsigned short)(w & 0x7FFu);
                break;
            case 0x2: {
                unsigned char p = (unsigned char)((w >> 11) & 1u);
                unsigned short x = (unsigned short)(w & 0x7FFu);
                long long t = epoch_us + (((long long)th) << 12) + (long long)tl;
                out_x[cd_cursor] = x;
                out_y[cd_cursor] = y;
                out_p[cd_cursor] = p;
                out_t_us[cd_cursor] = t;
                cd_cursor += 1;
                break;
            }
            case 0x3:
                pol = (unsigned char)((w >> 11) & 1u);
                bx = (unsigned short)(w & 0x7FFu);
                break;
            case 0x4: {
                unsigned int mask = w & 0xFFFu;
                long long t = epoch_us + (((long long)th) << 12) + (long long)tl;
                for (unsigned int k = 0; k < 12; ++k) {
                    if (mask & (1u << k)) {
                        out_x[cd_cursor] = (unsigned short)(bx + k);
                        out_y[cd_cursor] = y;
                        out_p[cd_cursor] = pol;
                        out_t_us[cd_cursor] = t;
                        cd_cursor += 1;
                    }
                }
                bx = (unsigned short)(bx + 12);
                break;
            }
            case 0x5: {
                unsigned int mask = w & 0xFFu;
                long long t = epoch_us + (((long long)th) << 12) + (long long)tl;
                for (unsigned int k = 0; k < 8; ++k) {
                    if (mask & (1u << k)) {
                        out_x[cd_cursor] = (unsigned short)(bx + k);
                        out_y[cd_cursor] = y;
                        out_p[cd_cursor] = pol;
                        out_t_us[cd_cursor] = t;
                        cd_cursor += 1;
                    }
                }
                bx = (unsigned short)(bx + 8);
                break;
            }
            case 0x6:
                tl = (unsigned short)(w & 0xFFFu);
                break;
            case 0x8: {
                unsigned short new_th = (unsigned short)(w & 0xFFFu);
                if (new_th < th) epoch_us += (1LL << 24);
                th = new_th;
                break;
            }
            case 0xA: {
                unsigned char tp = (unsigned char)(w & 1u);
                unsigned char tid = (unsigned char)((w >> 8) & 0xFu);
                long long t = epoch_us + (((long long)th) << 12) + (long long)tl;
                out_trig_t_us[trig_cursor] = t;
                out_trig_p[trig_cursor] = tp;
                out_trig_id[trig_cursor] = tid;
                trig_cursor += 1;
                break;
            }
            default:
                break;
        }
    }
}
