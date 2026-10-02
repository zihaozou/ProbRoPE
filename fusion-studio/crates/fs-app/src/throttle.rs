pub const SLIDER_MIN_INTERVAL_MS: u64 = 100;

#[derive(Default)]
pub struct SendThrottle {
    last_sent_ms: Option<u64>,
    pending: bool,
}

impl SendThrottle {

    pub fn update(&mut self, changed: bool, force: bool, now_ms: u64) -> bool {
        if changed {
            self.pending = true;
        }
        if !self.pending {
            return false;
        }
        let due = self.last_sent_ms.is_none_or(|t| now_ms.saturating_sub(t) >= SLIDER_MIN_INTERVAL_MS);
        if force || due {
            self.last_sent_ms = Some(now_ms);
            self.pending = false;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_change_sends_immediately() {
        let mut t = SendThrottle::default();
        assert!(t.update(true, false, 1_000), "第一次变化应立刻发送");
    }

    #[test]
    fn changes_inside_the_window_are_held_then_flushed() {
        let mut t = SendThrottle::default();
        assert!(t.update(true, false, 1_000));
        assert!(!t.update(true, false, 1_050), "50ms 内不再发");
        assert!(t.update(false, false, 1_100), "窗口到期必须把最后的值补发出去");
        assert!(!t.update(false, false, 1_500), "已经没有 pending 了,不再发");
    }

    #[test]
    fn force_sends_and_rearms_the_window() {
        let mut t = SendThrottle::default();
        assert!(t.update(true, false, 1_000));
        assert!(t.update(true, true, 1_010), "松手(force)必须立刻发最终值");
        assert!(!t.update(true, false, 1_050), "force 之后窗口重新计时");
    }

    #[test]
    fn force_without_pending_sends_nothing() {
        let mut t = SendThrottle::default();
        assert!(!t.update(false, true, 1_000), "没有变化就没有要发的东西");
    }
}
