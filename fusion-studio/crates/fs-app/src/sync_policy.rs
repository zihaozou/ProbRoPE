use fs_core::health::{SyncHealth, SyncState};

pub const COOLDOWN_MS: u64 = 10_000;

pub const RELAPSE_MS: u64 = 5_000;

pub const MAX_ATTEMPTS: u32 = 3;


#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Activity {
    pub calibrating: bool,
    pub recording: bool,
}

impl Activity {
    fn busy(self) -> bool {
        self.calibrating || self.recording
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Action {
    Nothing,

    AutoResync,

    Freeze,
    Unfreeze,

    GiveUp,
}


pub struct SyncPolicy {
    last_attempt_ms: Option<u64>,
    failures: u32,
    gave_up: bool,
    frozen: bool,
}

impl SyncPolicy {
    pub fn new() -> Self {
        SyncPolicy { last_attempt_ms: None, failures: 0, gave_up: false, frozen: false }
    }


    pub fn decide(&mut self, h: SyncHealth, a: Activity, now_ms: u64) -> Action {
        if h.state != SyncState::Lost {
            if self.last_attempt_ms.is_some_and(|t| now_ms.saturating_sub(t) > RELAPSE_MS) {
                self.failures = 0;
                self.gave_up = false;
                self.last_attempt_ms = None;
            }
            if self.frozen {
                self.frozen = false;
                return Action::Unfreeze;
            }
            return Action::Nothing;
        }

        if a.busy() {
            if self.frozen {
                return Action::Nothing;
            }
            self.frozen = true;
            return Action::Freeze;
        }

        if self.frozen {
            self.frozen = false;
            return Action::Unfreeze;
        }

        if self.gave_up {
            return Action::GiveUp;
        }
        match self.last_attempt_ms {
            Some(t) if now_ms.saturating_sub(t) < COOLDOWN_MS => Action::Nothing,
            Some(_) => {
                self.failures += 1;
                if self.failures >= MAX_ATTEMPTS {
                    self.gave_up = true;
                    return Action::GiveUp;
                }
                self.last_attempt_ms = Some(now_ms);
                Action::AutoResync
            }
            None => {
                self.last_attempt_ms = Some(now_ms);
                Action::AutoResync
            }
        }
    }

    pub fn manual_resync(&mut self, now_ms: u64) {
        self.failures = 0;
        self.gave_up = false;
        self.last_attempt_ms = Some(now_ms);
    }


    pub fn frozen(&self) -> bool {
        self.frozen
    }

    pub fn gave_up(&self) -> bool {
        self.gave_up
    }
}

impl Default for SyncPolicy {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fs_core::health::{SyncHealth, SyncState};

    fn health(state: SyncState) -> SyncHealth {
        SyncHealth {
            state,
            match_rate_window: if state == SyncState::Healthy { Some(1.0) } else { Some(0.2) },
            consecutive_non_matched: if state == SyncState::Lost { 20 } else { 0 },
            window_samples: 30,
            since_ms: 0,
            desynced: false,
        }
    }
    const IDLE: Activity = Activity { calibrating: false, recording: false };
    const CALIB: Activity = Activity { calibrating: true, recording: false };
    const REC: Activity = Activity { calibrating: false, recording: true };

    #[test]
    fn healthy_does_nothing() {
        let mut p = SyncPolicy::new();
        assert_eq!(p.decide(health(SyncState::Healthy), IDLE, 1_000), Action::Nothing);
        assert_eq!(p.decide(health(SyncState::Degraded), IDLE, 1_000), Action::Nothing);
    }

    #[test]
    fn lost_while_idle_auto_resyncs() {
        let mut p = SyncPolicy::new();
        assert_eq!(p.decide(health(SyncState::Lost), IDLE, 1_000), Action::AutoResync);
    }


    #[test]
    fn lost_while_busy_freezes_instead() {
        let mut p = SyncPolicy::new();
        assert_eq!(p.decide(health(SyncState::Lost), CALIB, 1_000), Action::Freeze);
        let mut p = SyncPolicy::new();
        assert_eq!(p.decide(health(SyncState::Lost), REC, 1_000), Action::Freeze);
    }

    #[test]
    fn cooldown_blocks_a_second_attempt() {
        let mut p = SyncPolicy::new();
        assert_eq!(p.decide(health(SyncState::Lost), IDLE, 1_000), Action::AutoResync);
        assert_eq!(p.decide(health(SyncState::Lost), IDLE, 1_500), Action::Nothing);
        assert_eq!(p.decide(health(SyncState::Lost), IDLE, 1_000 + COOLDOWN_MS), Action::AutoResync);
    }

    #[test]
    fn three_relapses_give_up() {
        let mut p = SyncPolicy::new();
        let mut t = 1_000;
        for _ in 0..3 {
            assert_eq!(p.decide(health(SyncState::Lost), IDLE, t), Action::AutoResync);
            t += COOLDOWN_MS;
        }
        assert_eq!(p.decide(health(SyncState::Lost), IDLE, t), Action::GiveUp);
        assert_eq!(p.decide(health(SyncState::Lost), IDLE, t + COOLDOWN_MS), Action::GiveUp,
                   "放弃之后不再自动动手");
    }


    #[test]
    fn a_healthy_stretch_clears_the_failure_count() {
        let mut p = SyncPolicy::new();
        let mut t = 1_000;
        for _ in 0..2 {
            assert_eq!(p.decide(health(SyncState::Lost), IDLE, t), Action::AutoResync);
            t += COOLDOWN_MS;
        }
        p.decide(health(SyncState::Healthy), IDLE, t + RELAPSE_MS + 1);
        t += RELAPSE_MS + COOLDOWN_MS + 2;
        assert_eq!(p.decide(health(SyncState::Lost), IDLE, t), Action::AutoResync,
                   "健康过一段之后应重新获得三次机会");
    }


    #[test]
    fn manual_resync_clears_cooldown_and_giveup() {
        let mut p = SyncPolicy::new();
        let mut t = 1_000;
        for _ in 0..3 {
            p.decide(health(SyncState::Lost), IDLE, t);
            t += COOLDOWN_MS;
        }
        assert_eq!(p.decide(health(SyncState::Lost), IDLE, t), Action::GiveUp);
        p.manual_resync(t);
        assert_eq!(p.decide(health(SyncState::Lost), IDLE, t + COOLDOWN_MS), Action::AutoResync);
    }


    #[test]
    fn recovering_after_a_freeze_unfreezes_once() {
        let mut p = SyncPolicy::new();
        assert_eq!(p.decide(health(SyncState::Lost), CALIB, 1_000), Action::Freeze);
        assert_eq!(p.decide(health(SyncState::Healthy), CALIB, 2_000), Action::Unfreeze);
        assert_eq!(p.decide(health(SyncState::Healthy), CALIB, 2_100), Action::Nothing,
                   "解冻只发一次");
    }


    #[test]
    fn freeze_is_not_repeated_while_it_holds() {
        let mut p = SyncPolicy::new();
        assert_eq!(p.decide(health(SyncState::Lost), CALIB, 1_000), Action::Freeze);
        assert_eq!(p.decide(health(SyncState::Lost), CALIB, 1_100), Action::Nothing);
    }


    #[test]
    fn frozen_reflects_the_freeze_decision_not_raw_lost() {
        let mut p = SyncPolicy::new();
        p.decide(health(SyncState::Lost), IDLE, 1_000);
        assert!(!p.frozen(), "空闲失步是自动恢复,不该冻结面板");
        let mut p = SyncPolicy::new();
        p.decide(health(SyncState::Lost), CALIB, 1_000);
        assert!(p.frozen());
    }


    #[test]
    fn dropping_out_of_busy_while_lost_unfreezes_then_resyncs() {
        let mut p = SyncPolicy::new();
        assert_eq!(p.decide(health(SyncState::Lost), CALIB, 1_000), Action::Freeze);
        assert_eq!(p.decide(health(SyncState::Lost), IDLE, 1_100), Action::Unfreeze);
        assert!(!p.frozen());
        assert_eq!(p.decide(health(SyncState::Lost), IDLE, 1_200), Action::AutoResync);
    }


    #[test]
    fn giving_up_while_idle_still_freezes_when_work_starts() {
        let mut p = SyncPolicy::new();
        let mut t = 1_000;
        for _ in 0..3 {
            assert_eq!(p.decide(health(SyncState::Lost), IDLE, t), Action::AutoResync);
            t += COOLDOWN_MS;
        }
        assert_eq!(p.decide(health(SyncState::Lost), IDLE, t), Action::GiveUp);

        t += 100;
        assert_eq!(p.decide(health(SyncState::Lost), CALIB, t), Action::Freeze,
                   "已经放弃自动重试,不代表正在进行的标定可以不受保护");

        t += 100;
        assert_eq!(p.decide(health(SyncState::Lost), IDLE, t), Action::Unfreeze);

        t += 100;
        assert_eq!(p.decide(health(SyncState::Lost), IDLE, t), Action::GiveUp,
                   "解冻之后,放弃状态本身还在,应当重新浮现,不会被那次解冻抹掉");
    }


    #[test]
    fn manual_resync_is_idempotent_and_does_not_touch_the_freeze() {
        let mut p = SyncPolicy::new();
        p.manual_resync(1_000);
        p.manual_resync(1_000);
        assert_eq!(p.decide(health(SyncState::Lost), IDLE, 1_000 + COOLDOWN_MS), Action::AutoResync);

        let mut p = SyncPolicy::new();
        assert_eq!(p.decide(health(SyncState::Lost), CALIB, 1_000), Action::Freeze);
        p.manual_resync(1_100);
        assert!(p.frozen(), "manual_resync 不解冻;解冻只走 decide 里的两条路径");

        let mut p = SyncPolicy::new();
        p.manual_resync(1_000);
        assert_eq!(p.decide(health(SyncState::Healthy), IDLE, 1_000), Action::Nothing);
    }
}
